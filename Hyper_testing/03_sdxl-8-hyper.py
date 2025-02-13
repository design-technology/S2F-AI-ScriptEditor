import os
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
from huggingface_hub import login
# Authenticate with Hugging Face token
login(token="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")

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
base_model_id = "SG161222/RealVisXL_V4.0"
repo_name = "ByteDance/Hyper-SD"
generator = torch.Generator("cuda").manual_seed(5555)
# Take 8-steps lora as an example
ckpt_name = "Hyper-SDXL-8steps-lora.safetensors"
# Load model, please fill in your access tokens since SD3 repo is a gated model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, token="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")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)
pipe.enable_model_cpu_offload()

image=pipe(
    prompt="a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization", 
    num_inference_steps=8, 
    guidance_scale=5.0,
    generator = generator,
    ).images[0]
image.save("sdxl-2-2-hyper.png")
image.show()
