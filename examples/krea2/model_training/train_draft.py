"""Train a Krea-2 Raw LoRA with DRaFT using prompt-only metadata."""
import argparse
import json
import os

import accelerate
import torch

from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, add_general_config, add_image_size_config, launch_training_task
from diffsynth.diffusion.draft import DRaFTConfig, DRaFTLoss, DRaFTPipelineAdapter, DifferentiableAestheticReward, DifferentiableFaceIdentityReward
from diffsynth.metrics.aesthetic import AestheticMetric
from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
from diffsynth.utils.lora.krea2 import Krea2LoRAConverter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Krea2DRaFTAdapter(DRaFTPipelineAdapter):
    """Krea-2's scheduler/VAE details; DRaFT algorithm stays shared."""

    def configure_scheduler(self, pipe, inputs_shared, config):
        dynamic_shift_len = (inputs_shared["height"] // 16) * (inputs_shared["width"] // 16)
        pipe.scheduler.set_timesteps(config.num_inference_steps, denoising_strength=1.0, dynamic_shift_len=dynamic_shift_len)

    def before_sampling(self, pipe):
        pipe.load_models_to_device(["dit"])

    def before_decode(self, pipe):
        pipe.load_models_to_device(["vae"])

    def decode(self, pipe, latents):
        # Qwen Image VAE performs latent normalization internally.
        return ((pipe.vae.decode(latents) + 1) / 2).clamp(0, 1)

    def low_variance_point(self, pipe, config):
        # Krea's continuous scheduler interprets this argument as a trajectory
        # index, whereas the default adapter uses a DDIM training timestep.
        progress_id = min(config.low_variance_timestep, len(pipe.scheduler.timesteps) - 1)
        timestep = pipe.scheduler.timesteps[progress_id].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        return timestep, progress_id


class Krea2DRaFTTrainingModule(DiffusionTrainingModule):
    def __init__(self, args, device):
        super().__init__()
        if args.height is None or args.width is None:
            raise ValueError("DRaFT requires explicit --height and --width; it uses prompt-only metadata.")
        configs = self.parse_model_configs(args.model_paths, args.model_id_with_origin_paths, fp8_models=args.fp8_models, offload_models=args.offload_models, device=device)
        tokenizer = self.parse_path_or_model_id(args.tokenizer_path, ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""))
        self.pipe = Krea2Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=configs, tokenizer_config=tokenizer)
        self.switch_pipe_to_training_mode(self.pipe, args.trainable_models, args.lora_base_model, args.lora_target_modules, args.lora_rank, args.lora_checkpoint, args.preset_lora_path, args.preset_lora_model, task="draft")
        if args.draft_reward == "aesthetic":
            metric = AestheticMetric.from_pretrained(torch_dtype=torch.float32, device=device)
            self.reward = DifferentiableAestheticReward(metric.model)
        else:
            if args.draft_face_reference_image is None:
                raise ValueError("--draft_face_reference_image is required with --draft_reward face_identity.")
            self.reward = DifferentiableFaceIdentityReward(
                args.draft_face_reference_image,
                insightface_root=args.draft_insightface_root,
                detection_size=args.draft_face_detection_size,
                device=device,
            )
        self.loss_fn = DRaFTLoss(self.reward, DRaFTConfig(args.draft_num_inference_steps, args.draft_truncated_backprop_steps, args.draft_low_variance_samples, args.draft_low_variance_timestep, args.draft_cfg_scale), Krea2DRaFTAdapter())
        self.height, self.width = args.height, args.width
        self.use_gradient_checkpointing = args.use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = args.use_gradient_checkpointing_offload

    def get_pipeline_inputs(self, data):
        shared = {
            "input_image": None, "height": self.height, "width": self.width,
            "cfg_scale": self.loss_fn.config.cfg_scale, "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        return shared, {"prompt": json.dumps(data["prompt"]), "context_pre_compute": True}, {"negative_prompt": "", "context_pre_compute": True}

    def forward(self, data):
        inputs = self.transfer_data_to_device(self.get_pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.loss_fn(self.pipe, *inputs)


def parser():
    p = argparse.ArgumentParser(description="Krea-2 Raw DRaFT aesthetic LoRA training")
    add_general_config(p)
    add_image_size_config(p)
    for action in p._actions:
        if action.dest == "dataset_base_path":
            action.required, action.default = False, ""
        elif action.dest == "dataset_metadata_path":
            action.required = True
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--align_to_opensource_format", default=False, action="store_true")
    p.add_argument("--draft_num_inference_steps", type=int, default=28)
    p.add_argument("--draft_truncated_backprop_steps", type=int, default=1)
    p.add_argument("--draft_low_variance_samples", type=int, default=1)
    p.add_argument("--draft_low_variance_timestep", type=int, default=12, help="Krea-2 sampler index used for DRaFT-LV resampling.")
    p.add_argument("--draft_cfg_scale", type=float, default=3.5)
    p.add_argument("--draft_reward", choices=["aesthetic", "face_identity"], default="aesthetic")
    p.add_argument("--draft_face_reference_image", type=str, default=None, help="Reference face image for face_identity reward.")
    p.add_argument("--draft_insightface_root", type=str, default="models/insightface", help="InsightFace root containing models/antelopev2/.")
    p.add_argument("--draft_face_detection_size", type=int, default=640, help="Antelopev2 face detector resolution.")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)])
    dataset = UnifiedDataset(base_path="", metadata_path=args.dataset_metadata_path, repeat=args.dataset_repeat, data_file_keys=tuple(), main_data_operator=lambda value: value)
    model = Krea2DRaFTTrainingModule(args, "cpu" if args.enable_model_cpu_offload else accelerator.device)
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, state_dict_converter=Krea2LoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda value: value, enable_tensorboard_log=args.enable_tensorboard_log, enable_swanlab_log=args.enable_swanlab_log, swanlab_project=args.swanlab_project, enable_wandb_log=args.enable_wandb_log, wandb_project=args.wandb_project)
    launch_training_task(accelerator, dataset, model, logger, args=args)
