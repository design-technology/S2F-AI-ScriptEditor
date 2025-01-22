import os

# Set cache path for models
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import numpy as np

# Create an output folder for images
output_folder = Path("GM_Internal_Workshop/images/steps_02")
output_folder.mkdir(parents=True, exist_ok=True)

# Load the model
model_id = "SG161222/RealVisXL_V4.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move to GPU

# Define the prompt
prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"
negative_prompt = "low res, bad quality"

#A callback is a function executed at specific points during a process. In this case, the callback runs after every step of the inference process to save intermediate images.

# Normalize the decoded latents to image range
def latents_to_image(decoded_latents):
    images = (decoded_latents / 2 + 0.5).clip(0, 1)  # Rescale to [0, 1]
    images = (images * 255).astype("uint8")          # Scale to [0, 255] as uint8
    return images

# Callback function to save images at each step
def save_step_images(step, timestep, latents):
    decoded_output = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)
    decoded_images = decoded_output.sample.cpu().detach().numpy()
    images = latents_to_image(decoded_images)
    pil_images = [Image.fromarray(image.transpose(1, 2, 0)) for image in images]  # Convert channel-first to channel-last
    pil_images[0].save(output_folder / f"step_{step:03d}.png")

# Generate the image with callback
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    num_inference_steps=25,
    num_images_per_prompt=1,
    callback=save_step_images,  # Attach the callback here
    callback_steps=1,  # Call the callback at every step
).images[0]

# Save the final image
image.save(output_folder / "final_image.png")
