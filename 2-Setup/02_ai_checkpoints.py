# https://github.com/conda/conda/issues/8273 openssl issue

import os
from huggingface_hub import snapshot_download, hf_hub_download

# Set this to your desired cache location with at least 60GB free space
cache_path = 'C:\hg_models'  # e.g., 'D:\\huggingface_cache'

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

# Model IDs
base_model_id = "SG161222/RealVisXL_V4.0"  
controlnet_model_id = "xinsir/controlnet-canny-sdxl-1.0"
repo_name = "ByteDance/Hyper-SD"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
ckpt_name = "Hyper-SDXL-8steps-lora.safetensors"

print("Downloading models to cache (this may take a while)...")

# Download models to cache without loading them
snapshot_download(repo_id=base_model_id, cache_dir=cache_path)
snapshot_download(repo_id=controlnet_model_id, cache_dir=cache_path)
hf_hub_download(repo_id=repo_name, filename=ckpt_name, cache_dir=cache_path)
snapshot_download(repo_id=lcm_lora_id, cache_dir=cache_path)

print("All models downloaded successfully!")
