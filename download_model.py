import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoPipelineForText2Image
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)
pipe_ip = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe_ip.load_ip_adapter("h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")