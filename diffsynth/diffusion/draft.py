"""Direct Reward Fine-Tuning (DRaFT), isolated from existing training paths.

Implements DRaFT, DRaFT-K, and DRaFT-LV from Clark et al., ICLR 2024
(arXiv:2309.17400v2). A ``DRaFTPipelineAdapter`` supplies the small,
model-specific scheduler and decoder details; the optimization algorithm itself
is shared by every supported image pipeline.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
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


class DifferentiableFaceIdentityReward(torch.nn.Module):
    """ArcFace identity reward with static or per-row reference faces.

    Antelopev2 detects faces on detached CPU images. The selected generated-face
    crop uses differentiable ROI Align before a frozen PyTorch ArcFace encoder
    computes cosine similarity to the active reference embedding.
    """

    def __init__(
        self,
        reference_image_path: str | None = None,
        insightface_root: str = "models/insightface",
        detection_size: int = 640,
        device="cuda",
    ):
        super().__init__()
        try:
            import cv2
            import numpy as np
            from facexlib.recognition import init_recognition_model
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except ImportError as error:
            raise ImportError(
                "Face identity DRaFT requires facexlib, insightface, onnxruntime, and opencv-python. "
                "See docs/en/Training/DRaFT.md for setup."
            ) from error

        self._cv2 = cv2
        self._np = np
        self._face_align = face_align
        self._detector = self._load_antelope_detector(FaceAnalysis, insightface_root, detection_size)
        self.recognizer = init_recognition_model("arcface", device=device).eval().requires_grad_(False)
        self.register_buffer("reference_embedding", torch.empty(0), persistent=True)
        self._dataset_reference_image = None
        if reference_image_path is not None:
            self._set_static_reference(reference_image_path)

    @staticmethod
    def _load_antelope_detector(face_analysis_class, root, detection_size):
        """Load Antelopev2 and repair its occasionally nested zip layout."""
        kwargs = {"name": "antelopev2", "root": root, "providers": ["CPUExecutionProvider"]}
        try:
            detector = face_analysis_class(**kwargs)
        except AssertionError:
            model_dir = Path(root) / "models" / "antelopev2"
            nested_dir = model_dir / "antelopev2"
            nested_models = list(nested_dir.glob("*.onnx")) if nested_dir.is_dir() else []
            if not nested_models:
                raise
            for source in nested_models:
                destination = model_dir / source.name
                if destination.exists():
                    raise RuntimeError(f"Cannot normalize Antelopev2 layout: destination already exists: {destination}")
                source.replace(destination)
            if not any(nested_dir.iterdir()):
                nested_dir.rmdir()
            detector = face_analysis_class(**kwargs)
        detector.prepare(ctx_id=-1, det_size=(detection_size, detection_size))
        return detector

    @staticmethod
    def _largest_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda face: float((face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])))

    def _features(self, faces):
        features = self.recognizer(faces)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return F.normalize(features, dim=-1)

    def _reference_tensor(self, image):
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image_bgr = self._cv2.cvtColor(self._np.array(image.convert("RGB")), self._cv2.COLOR_RGB2BGR)
        face = self._largest_face(self._detector.get(image_bgr))
        if face is None:
            raise ValueError("No face detected in face-identity reference image.")
        aligned = self._face_align.norm_crop(image_bgr, landmark=face["kps"], image_size=112)
        return torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1

    def _set_static_reference(self, image):
        device = next(self.recognizer.parameters()).device
        with torch.no_grad():
            embedding = self._features(self._reference_tensor(image).to(device=device))
        self.reference_embedding = embedding.squeeze(0).detach()

    def set_reference_image(self, image):
        """Select the PIL image for the next prompt/reference dataset row."""
        self._dataset_reference_image = image

    def _active_reference_embedding(self, device, dtype):
        if self.reference_embedding.numel() != 0:
            return self.reference_embedding.to(device=device, dtype=dtype)
        if self._dataset_reference_image is None:
            raise RuntimeError("Face identity reward needs --draft_face_reference_image or a dataset `image` column.")
        with torch.no_grad():
            embedding = self._features(self._reference_tensor(self._dataset_reference_image).to(device=device))
        return embedding.squeeze(0).to(dtype=dtype)

    def _detect_rois(self, images):
        arrays = (images.detach().float().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
        rois, indices = [], []
        for index, image in enumerate(arrays):
            face = self._largest_face(self._detector.get(self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)))
            if face is None:
                continue
            x0, y0, x1, y1 = (float(value) for value in face["bbox"])
            padding_x, padding_y = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            height, width = image.shape[:2]
            rois.append([index, max(0, x0 - padding_x), max(0, y0 - padding_y), min(width, x1 + padding_x), min(height, y1 + padding_y)])
            indices.append(index)
        if not rois:
            return None, None
        return torch.tensor(rois, device=images.device, dtype=torch.float32), torch.tensor(indices, device=images.device, dtype=torch.long)

    def forward(self, images):
        from torchvision.ops import roi_align

        rois, indices = self._detect_rois(images)
        zero_scores = images.float().mean(dim=(1, 2, 3)) * 0
        if rois is None:
            return zero_scores
        faces = roi_align(images[:, [2, 1, 0]].float(), rois, output_size=(112, 112), spatial_scale=1.0, aligned=True)
        embeddings = self._features(faces * 2 - 1)
        reference = self._active_reference_embedding(embeddings.device, embeddings.dtype)
        scores = (embeddings * reference.unsqueeze(0)).sum(dim=-1).clamp(-1, 1)
        return zero_scores.index_copy(0, indices, scores)

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
