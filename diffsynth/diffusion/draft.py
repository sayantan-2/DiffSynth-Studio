"""Direct Reward Fine-Tuning (DRaFT), isolated from existing training paths.

Implements DRaFT, DRaFT-K, and DRaFT-LV from Clark et al., ICLR 2024
(arXiv:2309.17400v2). The caller supplies an already prepared latent-diffusion
pipeline and a frozen, differentiable reward function.
"""
from dataclasses import dataclass
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


class DRaFTLoss(torch.nn.Module):
    """Minimizable negative reward loss for a DiffSynth latent-diffusion pipe.

    ``reward_fn`` receives a ``[B, 3, H, W]`` float tensor in [0, 1] and must
    return a differentiable score per image. Its parameters should be frozen.
    """

    def __init__(self, reward_fn: Callable[[torch.Tensor], torch.Tensor], config: DRaFTConfig = None):
        super().__init__()
        self.reward_fn = reward_fn
        self.config = config or DRaFTConfig()
        self.config.validate()

    @staticmethod
    def decode(pipe, latents):
        image = pipe.vae.decode(latents / pipe.vae.scaling_factor)
        return ((image + 1) / 2).clamp(0, 1)

    def reward(self, images):
        value = self.reward_fn(images)
        if not torch.is_tensor(value):
            raise TypeError("reward_fn must return a torch.Tensor")
        return value.reshape(-1).mean()

    def forward(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        cfg = self.config
        pipe.scheduler.set_timesteps(cfg.num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        shared = dict(inputs_shared)
        latents = shared["latents"]
        detach_at = cfg.num_inference_steps - cfg.truncated_backprop_steps

        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            if progress_id == detach_at:
                # This stop-gradient is DRaFT-K. K == number of inference
                # steps retains the full DRaFT graph.
                latents = latents.detach()
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            shared["latents"] = latents
            prediction = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg.cfg_scale, shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            latents = pipe.step(pipe.scheduler, latents, progress_id, prediction, **{name: value for name, value in shared.items() if name != "latents"})

        rewards = [self.reward(self.decode(pipe, latents))]
        if cfg.low_variance_samples > 1:
            # DRaFT-LV: forward diffuse the generated sample and denoise the
            # extra variants, which reduces gradient-estimator variance without
            # regenerating complete sampling trajectories.
            timestep = torch.tensor([cfg.low_variance_timestep], device=pipe.device, dtype=pipe.torch_dtype)
            for _ in range(cfg.low_variance_samples - 1):
                noisy = pipe.scheduler.add_noise(latents, torch.randn_like(latents), timestep)
                shared["latents"] = noisy
                prediction = pipe.cfg_guided_model_fn(
                    pipe.model_fn, cfg.cfg_scale, shared, inputs_posi, inputs_nega,
                    **models, timestep=timestep, progress_id=len(pipe.scheduler.timesteps) - 1,
                )
                denoised = pipe.scheduler.step(prediction, timestep, noisy, to_final=True)
                rewards.append(self.reward(self.decode(pipe, denoised)))
        return -torch.stack(rewards).mean()


class DifferentiableAestheticReward(torch.nn.Module):
    """Autograd-preserving adapter for DiffSynth's frozen AestheticMetric."""

    def __init__(self, aesthetic_model):
        super().__init__()
        self.model = aesthetic_model.eval().requires_grad_(False)

    def forward(self, images):
        processor = self.model.processor
        if processor is None:
            raise RuntimeError("Aesthetic model has no image processor")
        size = processor.crop_size
        size = (size["height"], size["width"]) if isinstance(size, dict) else (size, size)
        images = F.interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)
        mean = torch.as_tensor(processor.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(processor.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        pixel_values = (images - mean) / std
        features = self.model.vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
        features = self.model.visual_projection(features)
        features = F.normalize(features, dim=-1)
        return self.model.layers(features).squeeze(-1)


