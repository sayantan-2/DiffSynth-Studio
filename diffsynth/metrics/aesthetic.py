import os
import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.aesthetic import AestheticModel
from .base import Metric
from transformers import CLIPImageProcessor

class AestheticMetric(Metric):
    def __init__(self, model: AestheticModel):
        super().__init__()
        self.model = model

    @classmethod
    def _get_model_id(cls):
        download_source = os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
        if download_source.lower() == "huggingface":
            return "achiru/DiffSynth-Studio-ImageMetrics"
        return "DiffSynth-Studio/ImageMetrics"

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = None,
        processor_config: ModelConfig = None,
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):
        model_id = cls._get_model_id()
        if model_config is None:
            model_config = ModelConfig(model_id=model_id, origin_file_pattern="Aesthetic/model.safetensors")
        if processor_config is None:
            processor_config = ModelConfig(model_id=model_id, origin_file_pattern="Aesthetic/")

        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_aesthetic")
        processor_config.download_if_necessary()
        model.processor = CLIPImageProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model.layers = model.layers.float()
        model = model.eval()
        return cls(model)

    @torch.no_grad()
    def score(self, images):
        scores = self.model(images)
        return self.tensor_to_list(scores)

    def compute(self, images):
        return self.score(images)

    def forward(self, images):
        return self.score(images)
