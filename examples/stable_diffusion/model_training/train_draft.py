"""Train a Stable Diffusion LoRA with DRaFT using prompt-only metadata.

The metadata file must contain a `prompt` column/key. No reference images are
loaded or used: DRaFT generates its own image before evaluating the reward.
"""
import argparse
import os

import accelerate
import torch

from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, add_general_config, add_image_size_config, launch_training_task
from diffsynth.metrics.aesthetic import AestheticMetric
from diffsynth.pipelines.stable_diffusion import ModelConfig, StableDiffusionPipeline
from diffsynth.diffusion.draft import DRaFTConfig, DRaFTLoss, DifferentiableAestheticReward

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StableDiffusionDRaFTTrainingModule(DiffusionTrainingModule):
    def __init__(self, args, device):
        super().__init__()
        if args.height is None or args.width is None:
            raise ValueError("DRaFT requires explicit --height and --width; there is no reference image to infer them from.")
        configs = self.parse_model_configs(args.model_paths, args.model_id_with_origin_paths, fp8_models=args.fp8_models, offload_models=args.offload_models, device=device)
        tokenizer = self.parse_path_or_model_id(args.tokenizer_path, ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="tokenizer/"))
        self.pipe = StableDiffusionPipeline.from_pretrained(torch_dtype=torch.float32, device=device, model_configs=configs, tokenizer_config=tokenizer)
        self.switch_pipe_to_training_mode(self.pipe, args.trainable_models, args.lora_base_model, args.lora_target_modules, args.lora_rank, args.lora_checkpoint, args.preset_lora_path, args.preset_lora_model, task="draft")
        metric = AestheticMetric.from_pretrained(torch_dtype=torch.float32, device=device)
        self.reward = DifferentiableAestheticReward(metric.model)
        self.loss_fn = DRaFTLoss(self.reward, DRaFTConfig(args.draft_num_inference_steps, args.draft_truncated_backprop_steps, args.draft_low_variance_samples, args.draft_low_variance_timestep, args.draft_cfg_scale))
        self.height, self.width = args.height, args.width
        self.extra_inputs = args.extra_inputs.split(",") if args.extra_inputs else []
        self.use_gradient_checkpointing = args.use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = args.use_gradient_checkpointing_offload

    def get_pipeline_inputs(self, data):
        shared = {
            "input_image": None, "height": self.height, "width": self.width,
            "cfg_scale": self.loss_fn.config.cfg_scale, "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        return self.parse_extra_inputs(data, self.extra_inputs, shared), {"prompt": data["prompt"]}, {"negative_prompt": ""}

    def forward(self, data):
        inputs = self.transfer_data_to_device(self.get_pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.loss_fn(self.pipe, *inputs)


def parser():
    p = argparse.ArgumentParser(description="Stable Diffusion DRaFT aesthetic LoRA training")
    add_general_config(p)
    add_image_size_config(p)
    # DRaFT is prompt-only; relax the inherited image-training base-path flag.
    for action in p._actions:
        if action.dest == "dataset_base_path":
            action.required, action.default = False, ""
        elif action.dest == "dataset_metadata_path":
            action.required = True
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--draft_num_inference_steps", type=int, default=50)
    p.add_argument("--draft_truncated_backprop_steps", type=int, default=1)
    p.add_argument("--draft_low_variance_samples", type=int, default=1)
    p.add_argument("--draft_low_variance_timestep", type=int, default=20)
    p.add_argument("--draft_cfg_scale", type=float, default=7.5)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)])
    # No data_file_keys means UnifiedDataset leaves the prompt metadata intact
    # and does not attempt to resolve or load image/video paths.
    dataset = UnifiedDataset(base_path="", metadata_path=args.dataset_metadata_path, repeat=args.dataset_repeat, data_file_keys=tuple(), main_data_operator=lambda value: value)
    model = StableDiffusionDRaFTTrainingModule(args, "cpu" if args.enable_model_cpu_offload else accelerator.device)
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, enable_tensorboard_log=args.enable_tensorboard_log, enable_swanlab_log=args.enable_swanlab_log, swanlab_project=args.swanlab_project, enable_wandb_log=args.enable_wandb_log, wandb_project=args.wandb_project)
    launch_training_task(accelerator, dataset, model, logger, args=args)
