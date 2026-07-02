# Z-Image with Flux2 VAE projection training

This folder contains experimental training scripts for running Z-Image with the Flux2 VAE from `black-forest-labs/FLUX.2-dev`.

Z-Image was trained with the original Flux VAE latent layout:

- `16` latent channels
- `8x` spatial compression
- DiT patch size `2`

Flux2 VAE can be used here in two layouts:

- `packed`: `128` latent channels, `16x` spatial compression, DiT patch size `1`
- `unpacked`: `32` latent channels, `8x` spatial compression, DiT patch size `2`

The projection layers bridge the VAE latent layout and the existing Z-Image DiT hidden size:

- `dit.all_x_embedder`: maps latent patches into DiT hidden states
- `dit.all_final_layer`: maps DiT hidden states back into latent patches

The goal of this experiment is to test whether Flux2 VAE can be made usable with Z-Image by training only the projection layers first, then optionally adding LoRA or full DiT finetuning if projection-only training is not enough.

## Which latent format to try

Start with `unpacked` if you want the smallest change from original Z-Image geometry. It keeps the `8x` spatial compression and `patch_size=2`, but changes latent channels from `16` to `32`.

Use `packed` if you want the native Flux2 VAE packed latent path. It changes both latent channels and spatial geometry: `128` channels at `16x` compression, with `patch_size=1`.

In practice, the recommended order is:

1. `unpacked` projection-only
2. `packed` projection-only
3. `unpacked` projection + LoRA
4. `packed` projection + LoRA
5. full DiT finetune only if the smaller runs are clearly underfitting

## Training types

### Projection-only

Projection-only trains only:

```bash
--trainable_models "dit.all_x_embedder,dit.all_final_layer"
```

Use this first. It is the cleanest test of whether the existing DiT can operate with Flux2 VAE latents after only changing the input and output projection layers.

Scripts:

- [`Z-Image-Flux2VAE.sh`](./Z-Image-Flux2VAE.sh): Z-Image base, packed `128, H/16, W/16`
- [`Z-Image-Flux2VAE-Unpacked.sh`](./Z-Image-Flux2VAE-Unpacked.sh): Z-Image base, unpacked `32, H/8, W/8`
- [`Z-Image-Turbo-Flux2VAE.sh`](./Z-Image-Turbo-Flux2VAE.sh): Z-Image Turbo, packed `128, H/16, W/16`
- [`Z-Image-Turbo-Flux2VAE-Unpacked.sh`](./Z-Image-Turbo-Flux2VAE-Unpacked.sh): Z-Image Turbo, unpacked `32, H/8, W/8`

Default learning rate: `1e-4`.

### Projection layers plus LoRA

These scripts train the projection layers as full parameters and also add LoRA adapters to the rest of the DiT.

Use this when projection-only training runs without errors but the model cannot recover enough image quality.

Scripts:

- [`Z-Image-Flux2VAE-Linear-Lora.sh`](./Z-Image-Flux2VAE-Linear-Lora.sh): Z-Image base, packed
- [`Z-Image-Flux2VAE-Unpacked-Linear-Lora.sh`](./Z-Image-Flux2VAE-Unpacked-Linear-Lora.sh): Z-Image base, unpacked
- [`Z-Image-Turbo-Flux2VAE-Linear-Lora.sh`](./Z-Image-Turbo-Flux2VAE-Linear-Lora.sh): Z-Image Turbo, packed
- [`Z-Image-Turbo-Flux2VAE-Unpacked-Linear-Lora.sh`](./Z-Image-Turbo-Flux2VAE-Unpacked-Linear-Lora.sh): Z-Image Turbo, unpacked

The LoRA target modules are:

```bash
--lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3"
```

Default LoRA rank: `32`.

Default learning rate: `1e-4`.

### Full DiT finetune

These scripts train the full DiT:

```bash
--trainable_models "dit"
```

Use this only after the projection-only and projection-plus-LoRA paths show that the experiment is worth scaling. Full DiT finetuning has the highest chance of adapting to the VAE change, but it is also the easiest to overfit or destabilize.

Scripts:

- [`Z-Image-Flux2VAE-Full.sh`](./Z-Image-Flux2VAE-Full.sh): Z-Image base, packed
- [`Z-Image-Flux2VAE-Unpacked-Full.sh`](./Z-Image-Flux2VAE-Unpacked-Full.sh): Z-Image base, unpacked
- [`Z-Image-Turbo-Flux2VAE-Full.sh`](./Z-Image-Turbo-Flux2VAE-Full.sh): Z-Image Turbo, packed
- [`Z-Image-Turbo-Flux2VAE-Unpacked-Full.sh`](./Z-Image-Turbo-Flux2VAE-Unpacked-Full.sh): Z-Image Turbo, unpacked

Default learning rate: `1e-5`.

## Base vs Turbo

Use the `Z-Image-*` scripts when you want to train the base DiT:

```bash
Tongyi-MAI/Z-Image:transformer/*.safetensors
```

These still use the Turbo text encoder because that is the available Z-Image text encoder path used by the local examples:

```bash
Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors
```

Use the `Z-Image-Turbo-*` scripts when you want to train the Turbo DiT:

```bash
Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors
```

Turbo is faster to test, but if distillation behavior is the source of poor quality, the base scripts are the better experiment.

## Inference scripts

Flux2 VAE inference examples:

- [`../../../model_inference/Z-Image-Turbo-Flux2VAE.py`](../../../model_inference/Z-Image-Turbo-Flux2VAE.py): packed Flux2 VAE inference
- [`../../../model_inference/Z-Image-Turbo-Flux2VAE-Unpacked.py`](../../../model_inference/Z-Image-Turbo-Flux2VAE-Unpacked.py): unpacked Flux2 VAE inference

Validation examples for training checkpoints:

- [`../../validate_full/Z-Image-Turbo-Flux2VAE.py`](../../validate_full/Z-Image-Turbo-Flux2VAE.py): packed Flux2 VAE validation
- [`../../validate_full/Z-Image-Turbo-Flux2VAE-Unpacked.py`](../../validate_full/Z-Image-Turbo-Flux2VAE-Unpacked.py): unpacked Flux2 VAE validation

The provided inference and validation examples currently target Z-Image Turbo. For base-model inference, use the same Flux2 VAE flags and swap the transformer model config from `Tongyi-MAI/Z-Image-Turbo` to `Tongyi-MAI/Z-Image`.

## Resume and follow-up training

The training entrypoint supports the normal DiffSynth checkpoint flow. To continue a longer run, resume from the saved checkpoint/output path using the same script settings and keep the same `--flux2_vae_latent_format`.

Do not switch between `packed` and `unpacked` when resuming. The projection layer shapes are different:

- packed uses `128` input/output latent channels
- unpacked uses `32` input/output latent channels

Switching formats requires starting a new run or loading only compatible weights manually.

## Expected behavior

Before training, inference is expected to run but produce poor images because the projection layers are newly initialized for the Flux2 VAE latent layout.

A useful result from projection-only training is not necessarily a perfect image. The first target is to verify that:

- training runs without shape errors
- loss decreases
- sampled images become less broken than zero-shot Flux2 VAE inference
- packed and unpacked can be compared under the same dataset and prompt settings

If projection-only training improves structure but not quality, try projection plus LoRA. If that still underfits, use the full DiT scripts.
