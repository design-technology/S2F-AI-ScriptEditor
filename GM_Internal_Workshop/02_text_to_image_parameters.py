import os
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch

# stable diffusion pipeline parameters
# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img

# Load the model
CompVis = "CompVis/stable-diffusion-v1-4"
sdlegacy = "sd-legacy/stable-diffusion-v1-5"
crynuxai = "crynux-ai/stable-diffusion-v1-5"
RealVisXL = "SG161222/RealVisXL_V4.0"

model_id = RealVisXL
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move to GPU

# Define the prompt
prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"

negative_prompt = "low res, bad quality"

# Generate the image
image = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    #Alignment with the input text prompt
    guidance_scale = 7.5,
    height = 1024,
    width = 1024,
    num_inference_steps = 25,
    num_images_per_prompt = 1,
    output_type = "pil",

    ).images[0]

# Save the image
image.save(f"GM_Internal_Workshop/images/test.png")
image.show()
