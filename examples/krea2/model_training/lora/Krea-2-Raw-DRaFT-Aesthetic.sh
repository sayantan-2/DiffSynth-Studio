# Krea-2 Raw DRaFT-LV aesthetic LoRA smoke test. Requires only a prompt CSV.
# Krea-2 Turbo is intentionally not recommended for this fine-tuning path.
: "${PROMPTS_CSV:=./data/krea2_draft_prompts.csv}"
mkdir -p "$(dirname "$PROMPTS_CSV")"
if [ ! -f "$PROMPTS_CSV" ]; then
  cat > "$PROMPTS_CSV" <<'CSV'
prompt
a high-fashion editorial photograph of a red fox in a snowy forest
a cinematic wide-angle landscape of mountains at sunrise
a detailed brass mechanical hummingbird perched on a flower
a luminous watercolor painting of a small coastal village
CSV
fi

accelerate launch examples/krea2/model_training/train_draft.py \
  --dataset_metadata_path "$PROMPTS_CSV" \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "krea/Krea-2-Raw:raw.safetensors,Qwen/Qwen3-VL-4B-Instruct:*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "Qwen/Qwen3-VL-4B-Instruct:" \
  --learning_rate 4e-4 \
  --weight_decay 0.1 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Krea-2-Raw_draft_aesthetic_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "wq,wk,wv,gate,wo,up,down,first,tmlp.0,tmlp.2,projector,txtmlp.1,txtmlp.3,last.linear,tproj.1" \
  --lora_rank 8 \
  --draft_num_inference_steps 28 \
  --draft_truncated_backprop_steps 1 \
  --draft_low_variance_samples 2 \
  --draft_low_variance_timestep 12 \
  --draft_cfg_scale 3.5 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --align_to_opensource_format
