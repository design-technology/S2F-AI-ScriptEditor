import os

cache_path = 'C:\hg_models'  

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from diffusers import StableDiffusionXLPipeline
import torch

# Load the model
model_id = "SG161222/RealVisXL_V4.0"
# dtype, Model Instantiation dtype, for memory optimization.Normally gets instantiated with torch.float32 format
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# Move to GPU
pipe = pipe.to("cuda")
# speed up inference
pipe.enable_model_cpu_offload()
    
# Define the prompt
prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground,creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"


image = pipe(prompt).images[0]

# Save the image
image.save(f"test_2.png")
image.show()
