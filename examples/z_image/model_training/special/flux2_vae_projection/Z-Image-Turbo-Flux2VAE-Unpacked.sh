modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "z_image/Z-Image-Turbo/*" --local_dir ./data/diffsynth_example_dataset

# Projection-only compatibility training for running Z-Image-Turbo with Flux.2 VAE internal 32-channel latents.
# This is full-parameter training of dit.all_x_embedder and dit.all_final_layer only, not LoRA.
accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,black-forest-labs/FLUX.2-dev:vae/diffusion_pytorch_model.safetensors" \
  --use_flux2_vae \
  --flux2_vae_latent_format unpacked \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Turbo-Flux2VAE_unpacked_projection" \
  --trainable_models "dit.all_x_embedder,dit.all_final_layer" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8
