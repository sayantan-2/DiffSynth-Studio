import json, math, os
from contextlib import contextmanager

import pandas
import torch
from PIL import Image


class TrainingImageSampler:
    def __init__(
        self,
        prompts_path=None,
        output_path=None,
        sample_steps=None,
        sample_epochs=None,
    ):
        self.prompts_path = prompts_path
        self.output_path = output_path
        self.sample_steps = sample_steps
        self.sample_epochs = sample_epochs
        self.enabled = sample_steps is not None or sample_epochs is not None
        self.samples = []
        self._validate_config()
        if self.enabled:
            self.samples = self.load_prompt_metadata(prompts_path)

    def _validate_config(self):
        for name, value in (("sample_steps", self.sample_steps), ("sample_epochs", self.sample_epochs)):
            if value is not None and value <= 0:
                raise ValueError(f"--{name} must be a positive integer when provided.")
        if self.enabled and self.prompts_path is None:
            raise ValueError("--sample_prompts_path is required when --sample_steps or --sample_epochs is provided.")
        if self.enabled and self.output_path is None:
            raise ValueError("sample output path is required when sampling is enabled.")

    @staticmethod
    def load_prompt_metadata(prompts_path):
        if prompts_path.endswith(".json"):
            with open(prompts_path, "r") as f:
                metadata = json.load(f)
            if isinstance(metadata, dict):
                metadata = [metadata]
        elif prompts_path.endswith(".jsonl"):
            metadata = []
            with open(prompts_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metadata.append(json.loads(line))
        else:
            dataframe = pandas.read_csv(prompts_path)
            metadata = [dataframe.iloc[i].to_dict() for i in range(len(dataframe))]

        if not isinstance(metadata, list):
            raise ValueError("Sampling prompt metadata must be a list of rows, or a single JSON object with a prompt key.")
        samples = [TrainingImageSampler.clean_sample_row(row, row_id + 1) for row_id, row in enumerate(metadata)]
        if len(samples) == 0:
            raise ValueError("Sampling prompt metadata is empty.")
        return samples

    @staticmethod
    def clean_sample_row(row, row_id):
        if not isinstance(row, dict):
            raise ValueError(f"Sampling prompt row {row_id} must be an object/dictionary.")
        sample = {}
        for key, value in row.items():
            if TrainingImageSampler.is_missing_value(value):
                continue
            sample[str(key)] = TrainingImageSampler.to_python_value(value)
        if "prompt" not in sample:
            raise ValueError(f"Sampling prompt row {row_id} must contain a non-empty prompt value.")
        return sample

    @staticmethod
    def is_missing_value(value):
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, (list, tuple, dict)):
            return False
        try:
            missing = pandas.isna(value)
        except TypeError:
            return False
        try:
            return bool(missing)
        except ValueError:
            return False

    @staticmethod
    def to_python_value(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    def on_step_end(self, accelerator, model, step):
        if self.enabled and self.sample_steps is not None and step % self.sample_steps == 0:
            self.sample(accelerator, model, "step", step)

    def on_epoch_end(self, accelerator, model, epoch):
        if self.enabled and self.sample_epochs is not None and epoch % self.sample_epochs == 0:
            self.sample(accelerator, model, "epoch", epoch)

    def sample(self, accelerator, model, prefix, number):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            self.sample_model(unwrapped_model, prefix, number)
        accelerator.wait_for_everyone()

    def sample_model(self, model, prefix, number):
        if not hasattr(model, "pipe"):
            raise ValueError("Training-time sampling requires the training module to expose a `pipe` attribute.")
        pipe = model.pipe
        os.makedirs(self.output_path, exist_ok=True)
        with self.inference_context(pipe):
            for row_id, sample_kwargs in enumerate(self.samples, start=1):
                with torch.no_grad():
                    output = pipe(**sample_kwargs)
                self.save_image_output(output, prefix, number, row_id)

    @contextmanager
    def inference_context(self, pipe):
        training_units = getattr(pipe, "units", None)
        inference_units = getattr(pipe, "_diffsynth_inference_units", training_units)
        module_states = [(module, module.training) for module in pipe.modules()]
        scheduler_state = self.capture_scheduler_state(getattr(pipe, "scheduler", None))
        try:
            if inference_units is not None:
                pipe.units = inference_units
            pipe.eval()
            yield
        finally:
            if training_units is not None:
                pipe.units = training_units
            self.restore_scheduler_state(getattr(pipe, "scheduler", None), scheduler_state)
            for module, training in module_states:
                module.train(training)

    @staticmethod
    def capture_scheduler_state(scheduler):
        if scheduler is None:
            return None
        state = {}
        for name in ("sigmas", "timesteps", "linear_timesteps_weights", "training"):
            if hasattr(scheduler, name):
                value = getattr(scheduler, name)
                if isinstance(value, torch.Tensor):
                    value = value.clone()
                state[name] = value
        return state

    @staticmethod
    def restore_scheduler_state(scheduler, state):
        if scheduler is None or state is None:
            return
        for name, value in state.items():
            setattr(scheduler, name, value)

    def save_image_output(self, output, prefix, number, row_id):
        if isinstance(output, Image.Image):
            image = output
        elif isinstance(output, list) and len(output) == 1 and isinstance(output[0], Image.Image):
            image = output[0]
        elif isinstance(output, list) and all(isinstance(item, Image.Image) for item in output):
            raise ValueError("Training-time sampling supports one image per prompt row; this pipeline returned multiple images.")
        else:
            raise ValueError(f"Training-time sampling only supports image outputs, but received {type(output).__name__}.")
        file_name = f"{prefix}_{number}_{row_id}.png"
        image.save(os.path.join(self.output_path, file_name))
