from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    use_flux2_vae=True,
    flux2_vae_latent_format="packed",
)

prompt = "A cinematic portrait of a woman in a red dress standing under soft studio lighting, detailed fabric, realistic skin, shallow depth of field."
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("image_Z-Image-Turbo-Flux2VAE.jpg")
