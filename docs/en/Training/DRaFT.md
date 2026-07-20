# Direct Reward Fine-Tuning (DRaFT)

This is an isolated implementation of [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400v2) (Clark et al., ICLR 2024). It adds no changes to an existing DiffSynth pipeline, metric, or training script.

DRaFT optimizes a diffusion model **online**: for each prompt it samples a fresh image, evaluates a frozen differentiable reward, and backpropagates that reward through the VAE decoder and selected sampling steps into the trainable LoRA weights. It does not fit reference images.

## Files

- `diffsynth/diffusion/draft.py`: reusable image DRaFT objective.
- `examples/stable_diffusion/model_training/train_draft.py`: Stable Diffusion v1.5 prompt-only LoRA runner.
- `examples/stable_diffusion/model_training/lora/stable-diffusion-v1-5-draft-aesthetic.sh`: minimal runnable smoke test.

## Prompt-only data

The runner requires `--dataset_metadata_path`, pointing to `.csv`, `.json`, or `.jsonl` data with a `prompt` field. Images, an `image` field, and `--dataset_base_path` are not needed or read. `--height` and `--width` are required because there is no reference image from which to infer resolution.

Example CSV:

```csv
prompt
a cinematic photograph of an astronaut in a tulip field
a studio product photo of a ceramic teapot
```

The included shell script creates a four-prompt CSV when `PROMPTS_CSV` does not already exist. It is only a wiring/smoke test; use a broad, representative prompt set for actual training.

## DRaFT arguments

| Argument | Meaning | Effect / recommended starting value |
| --- | --- | --- |
| `--draft_num_inference_steps 50` | Number of DDIM denoising calls used to make each training image. | Defines the sample trajectory and the maximum allowed K. Higher values cost more. The paper uses 50. |
| `--draft_truncated_backprop_steps 1` | **K** in DRaFT-K: number of *final* sampler calls retained for autograd. | The preceding `N-K` calls still generate the latent but are detached. `1` is the cheapest, most practical default; `50` with N=50 is original full DRaFT and is much more memory-intensive. |
| `--draft_low_variance_samples 2` | Number of reward trajectories averaged per prompt. | `1` disables LV. Values above 1 enable DRaFT-LV. `2` is the paper's small-scale choice and normally adds modest compute. |
| `--draft_low_variance_timestep 20` | Forward-diffusion training timestep used to make each extra LV variant from the generated final latent. | Only used when LV samples >1. It controls how much noise is injected before one extra denoise/reward pass. `20` is a conservative starting point; tune it with the reward and sampler. |
| `--draft_cfg_scale 7.5` | Classifier-free guidance weight used for every denoiser evaluation inside DRaFT. | Must generally match the CFG setting you plan to use at inference. The paper uses 7.5 for Stable Diffusion 1.4. |

## K and LV: separate, composable controls

They solve different problems and are not alternatives in the code:

- **DRaFT-K** chooses *how far through the main sampling trajectory to differentiate*. It reduces activation memory and backward compute by cutting the graph before the last K denoising steps.
- **DRaFT-LV** chooses *how many reward-gradient estimates to average after a main image is generated*. It forward-noises the generated latent, denoises those additional variants, scores them, then averages their rewards. This reduces gradient-estimator variance.

So they are independent axes and can be combined. Typical configurations are:

| Variant | N | K | LV samples |
| --- | ---: | ---: | ---: |
| Full DRaFT | 50 | 50 | 1 |
| DRaFT-K | 50 | 1, 5, or 10 | 1 |
| DRaFT-LV (recommended paper regime) | 50 | 1 | 2 |

The paper introduces LV specifically as an improvement to the K=1 regime, because K=1 had the best reward-versus-compute trade-off in its experiments. It is still technically valid to use LV with K>1; it simply costs more and is not the paper's main operating point.

## Reward currently maximized

`train_draft.py` maximizes the scalar output of the frozen LAION aesthetic predictor shipped through `AestheticMetric`: CLIP image features followed by a trained MLP. Its parameters do not receive gradients or optimizer updates. The reward's **input gradient** flows through a Torch-only resize/normalization adapter, VAE decode, sampling graph, and LoRA weights.

Metric methods that convert tensors to PIL/NumPy or use `torch.no_grad()` cannot be used directly in DRaFT because they sever this path.

## Using another reward

`DRaFTLoss` accepts any callable matching:

```python
reward_fn(images: torch.Tensor) -> torch.Tensor
```

Input images are float `[batch, 3, height, width]` tensors in `[0, 1]`. Return one scalar per image (or one scalar overall) without detaching the tensor. Freeze reward parameters with:

```python
reward = reward.eval().requires_grad_(False)
```

Useful choices include PickScore/HPS/CLIP preference scores, a detector confidence, a classifier logit, or a custom differentiable visual objective. If a model's published processor expects PIL images, reimplement its resize, crop, and normalization in Torch—the included `DifferentiableAestheticReward` is the template.

Text-conditioned rewards need the prompt passed to the adapter as well; the generic `DRaFTLoss` is image-reward-shaped by design, so make a small closure or adapter that captures the prompt batch.

## Extending to another image model

The objective is not tied to Stable Diffusion. An adapter needs a pipeline with:

1. differentiable sampler scheduling (`set_timesteps`, `add_noise`, `step`);
2. model/CFG hooks (`model_fn`, `cfg_guided_model_fn`, pipeline `step`);
3. initial latent and conditioning dictionaries;
4. differentiable VAE decode and scaling factor.

Make a new model-specific example runner to prepare the target pipeline's prompt, noise, and conditioning inputs, then feed those dictionaries to `DRaFTLoss`. Video/audio models need a corresponding decode and reward adapter, so they are intentionally outside this image implementation's contract.

## Run

```bash
bash examples/stable_diffusion/model_training/lora/stable-diffusion-v1-5-draft-aesthetic.sh
```

`PROMPTS_CSV=/path/to/prompts.csv bash ...` supplies your own prompt set. Model components and the aesthetic reward download on first use. Start with LoRA, gradient checkpointing, K=1, LV=1 or 2, then validate on held-out prompts with independent metrics and visual review: reward optimization can overfit reward-model artifacts or reduce diversity.
