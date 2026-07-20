# DRaFT-LV aesthetic LoRA smoke test for Stable Diffusion v1.5.
# Only a CSV with a `prompt` column is needed; no training images are used.
# Set PROMPTS_CSV to a larger prompt set for real training.
: "${PROMPTS_CSV:=./data/draft_prompts.csv}"
mkdir -p "$(dirname "$PROMPTS_CSV")"
if [ ! -f "$PROMPTS_CSV" ]; then
  cat > "$PROMPTS_CSV" <<'CSV'
prompt
a detailed portrait photograph of a red fox in a snowy forest
a cinematic landscape of mountains at sunrise
a brass mechanical hummingbird on a flower
a watercolor painting of a small coastal village
CSV
fi

accelerate launch examples/stable_diffusion/model_training/train_draft.py \
  --dataset_metadata_path "$PROMPTS_CSV" \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "AI-ModelScope/stable-diffusion-v1-5:text_encoder/model.safetensors,AI-ModelScope/stable-diffusion-v1-5:unet/diffusion_pytorch_model.safetensors,AI-ModelScope/stable-diffusion-v1-5:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 4e-4 \
  --weight_decay 0.1 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.unet." \
  --output_path "./models/train/stable-diffusion-v1-5_draft_aesthetic_lora" \
  --lora_base_model "unet" \
  --lora_target_modules "" \
  --lora_rank 8 \
  --draft_num_inference_steps 50 \
  --draft_truncated_backprop_steps 1 \
  --draft_low_variance_samples 2 \
  --draft_low_variance_timestep 20 \
  --draft_cfg_scale 7.5 \
  --use_gradient_checkpointing
