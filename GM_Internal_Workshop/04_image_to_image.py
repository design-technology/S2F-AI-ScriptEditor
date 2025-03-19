import os
cache_path = 'C:\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
import torch

# stable diffusion pipeline parameters
# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img

# Load the model

RealVisXL = "CompVis/stable-diffusion-v1-4"


model_id = RealVisXL
pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move to GPU

# Define the prompt
prompt =     "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"
negative_prompt = "low res, bad quality"
image_path = "GM_Internal_Workshop\images\pavilions.png"
init_image = load_image(image_path)

# Generate the image
image = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    image = init_image,
    #Alignment with the input text prompt
    guidance_scale = 2.5,
    height = 1024,
    width = 1024,
    num_inference_steps = 25,
    num_images_per_prompt = 1,
    strength=1.0,
    output_type = "pil",

    ).images[0]

# Save the image
image.save(f"GM_Internal_Workshop/images/test_image2image_5.png")
image_grid = make_image_grid([init_image, image], rows=1, cols=2)
image.show()
image_grid.show()
