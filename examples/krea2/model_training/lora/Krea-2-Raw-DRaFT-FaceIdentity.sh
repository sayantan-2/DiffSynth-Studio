# Krea-2 Raw DRaFT face-identity LoRA smoke test.
# Provide one clear, front-facing reference image with a dominant face.
: "${REFERENCE_FACE:?Set REFERENCE_FACE to your reference face image path}"
: "${PROMPTS_CSV:=./data/krea2_draft_prompts.csv}"

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
  --output_path "./models/train/Krea-2-Raw_draft_face_identity_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "wq,wk,wv,gate,wo,up,down,first,tmlp.0,tmlp.2,projector,txtmlp.1,txtmlp.3,last.linear,tproj.1" \
  --lora_rank 8 \
  --draft_num_inference_steps 28 \
  --draft_truncated_backprop_steps 1 \
  --draft_low_variance_samples 1 \
  --draft_cfg_scale 3.5 \
  --draft_reward face_identity \
  --draft_face_reference_image "$REFERENCE_FACE" \
  --draft_insightface_root "models/insightface" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --align_to_opensource_format