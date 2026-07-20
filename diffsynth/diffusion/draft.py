"""Direct Reward Fine-Tuning (DRaFT), isolated from existing training paths.

Implements DRaFT, DRaFT-K, and DRaFT-LV from Clark et al., ICLR 2024
(arXiv:2309.17400v2). A ``DRaFTPipelineAdapter`` supplies the small,
model-specific scheduler and decoder details; the optimization algorithm itself
is shared by every supported image pipeline.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Number
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class DRaFTConfig:
    num_inference_steps: int = 50
    truncated_backprop_steps: int = 1
    low_variance_samples: int = 1
    low_variance_timestep: int = 20
    cfg_scale: float = 7.5

    def validate(self):
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not 1 <= self.truncated_backprop_steps <= self.num_inference_steps:
            raise ValueError("truncated_backprop_steps must be in [1, num_inference_steps]")
        if self.low_variance_samples < 1:
            raise ValueError("low_variance_samples must be positive")
        if self.low_variance_timestep < 0:
            raise ValueError("low_variance_timestep must be non-negative")


class DRaFTPipelineAdapter:
    """Hooks needed to run the shared DRaFT loss on a DiffSynth image pipe.

    The defaults target Stable Diffusion/DDIM. Other pipelines override only
    the hooks whose scheduler or VAE differs; they do not duplicate DRaFT-K or
    DRaFT-LV's sampling and reward-gradient logic.
    """

    def configure_scheduler(self, pipe, inputs_shared, config: DRaFTConfig):
        pipe.scheduler.set_timesteps(config.num_inference_steps)

    def before_sampling(self, pipe):
        pass

    def before_decode(self, pipe):
        pass

    def decode(self, pipe, latents):
        image = pipe.vae.decode(latents / pipe.vae.scaling_factor)
        return ((image + 1) / 2).clamp(0, 1)

    def low_variance_point(self, pipe, config: DRaFTConfig):
        """Return `(timestep, progress_id)` for one DRaFT-LV denoise pass."""
        timestep = torch.tensor([config.low_variance_timestep], device=pipe.device, dtype=pipe.torch_dtype)
        return timestep, len(pipe.scheduler.timesteps) - 1


class DRaFTLoss(torch.nn.Module):
    """Minimizable negative reward loss for a DiffSynth latent-diffusion pipe.

    ``reward_fn`` receives a ``[B, 3, H, W]`` float tensor in [0, 1] and must
    return a differentiable score per image. Its parameters should be frozen.
    """

    def __init__(
        self,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        config: DRaFTConfig = None,
        pipeline_adapter: DRaFTPipelineAdapter = None,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.config = config or DRaFTConfig()
        self.config.validate()
        self.pipeline_adapter = pipeline_adapter or DRaFTPipelineAdapter()

    def reward(self, images):
        value = self.reward_fn(images)
        if not torch.is_tensor(value):
            raise TypeError("reward_fn must return a torch.Tensor")
        return value.reshape(-1).mean()

    def _denoise(self, pipe, shared, inputs_posi, inputs_nega, models, latents, timestep, progress_id):
        shared["latents"] = latents
        prediction = pipe.cfg_guided_model_fn(
            pipe.model_fn, self.config.cfg_scale, shared, inputs_posi, inputs_nega,
            **models, timestep=timestep, progress_id=progress_id,
        )
        return pipe.step(
            pipe.scheduler, latents, progress_id, prediction,
            **{name: value for name, value in shared.items() if name != "latents"},
        )

    def forward(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        cfg = self.config
        shared = dict(inputs_shared)
        adapter = self.pipeline_adapter
        adapter.configure_scheduler(pipe, shared, cfg)
        adapter.before_sampling(pipe)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        latents = shared["latents"]
        detach_at = cfg.num_inference_steps - cfg.truncated_backprop_steps

        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            if progress_id < detach_at:
                # DRaFT-K must avoid *building* a graph for the prefix, not
                # merely detach it later. Otherwise K=1 still reaches the
                # memory peak of full unrolled backpropagation.
                with torch.no_grad():
                    latents = self._denoise(pipe, shared, inputs_posi, inputs_nega, models, latents, timestep, progress_id)
            else:
                if progress_id == detach_at:
                    latents = latents.detach()
                latents = self._denoise(pipe, shared, inputs_posi, inputs_nega, models, latents, timestep, progress_id)

        reward_latents = [latents]
        if cfg.low_variance_samples > 1:
            timestep, progress_id = adapter.low_variance_point(pipe, cfg)
            for _ in range(cfg.low_variance_samples - 1):
                noisy = pipe.scheduler.add_noise(latents, torch.randn_like(latents), timestep)
                reward_latents.append(self._denoise(pipe, shared, inputs_posi, inputs_nega, models, noisy, timestep, progress_id))

        adapter.before_decode(pipe)
        rewards = [self.reward(adapter.decode(pipe, sample)) for sample in reward_latents]
        return -torch.stack(rewards).mean()


class DifferentiableAestheticReward(torch.nn.Module):
    """Autograd-preserving adapter for DiffSynth's frozen AestheticMetric."""

    def __init__(self, aesthetic_model):
        super().__init__()
        self.model = aesthetic_model.eval().requires_grad_(False)

    @staticmethod
    def _spatial_size(size):
        """Extract `(height, width)` from Transformers dict/SizeDict variants."""
        if isinstance(size, Number):
            value = int(size)
            return value, value
        height, width = getattr(size, "height", None), getattr(size, "width", None)
        if isinstance(height, Number) and isinstance(width, Number):
            return int(height), int(width)
        if hasattr(size, "to_dict"):
            size = size.to_dict()
        if isinstance(size, Mapping):
            height, width = size.get("height"), size.get("width")
            if isinstance(height, Number) and isinstance(width, Number):
                return int(height), int(width)
            # Some processor revisions nest SizeDict objects under both keys.
            for value in size.values():
                try:
                    return DifferentiableAestheticReward._spatial_size(value)
                except (TypeError, ValueError):
                    pass
        raise TypeError(f"Cannot extract an image size from processor crop_size={size!r}")

    def forward(self, images):
        processor = self.model.processor
        if processor is None:
            raise RuntimeError("Aesthetic model has no image processor")
        size = self._spatial_size(processor.crop_size)
        images = F.interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)
        mean = torch.as_tensor(processor.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(processor.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        pixel_values = (images - mean) / std
        features = self.model.vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
        features = self.model.visual_projection(features)
        features = F.normalize(features, dim=-1)
        return self.model.layers(features).squeeze(-1)
