import os
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from diffusers import StableDiffusionPipeline
import torch


# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move to GPU

# Define the prompt
prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"

# Generate the image
# output = pipe(prompt)
# output_image = pipe(prompt).images
# print(output)
# print(output_image)
# print(type(output))
# print(type(output_image))
image = pipe(prompt).images[0]

# Save the image
# image.save(f"test.png")
image.show()
