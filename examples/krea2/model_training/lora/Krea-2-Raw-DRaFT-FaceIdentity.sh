# Krea-2 Raw DRaFT face-identity training from image,prompt pairs.
# CSV columns: image,prompt. `image` is relative to DATASET_BASE_PATH.
: "${DATASET_BASE_PATH:?Set DATASET_BASE_PATH to the paired-image dataset directory}"
: "${DATASET_METADATA_PATH:?Set DATASET_METADATA_PATH to a CSV/JSON/JSONL with image,prompt fields}"

accelerate launch examples/krea2/model_training/train_draft.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "krea/Krea-2-Raw:raw.safetensors,Qwen/Qwen3-VL-4B-Instruct:*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "Qwen/Qwen3-VL-4B-Instruct:" \
  --learning_rate 4e-4 \
  --weight_decay 0.1 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Krea-2-Raw_draft_face_identity_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "wq,wk,wv,gate,wo,up,down,first,tmlp.0,tmlp.2,projector,txtmlp.1,txtmlp.3,last.linear,tproj.1" \
  --lora_rank 8 \
  --draft_num_inference_steps 30 \
  --draft_truncated_backprop_steps 1 \
  --draft_low_variance_samples 1 \
  --draft_cfg_scale 3.5 \
  --draft_reward face_identity \
  --draft_insightface_root "models/insightface" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --align_to_opensource_format