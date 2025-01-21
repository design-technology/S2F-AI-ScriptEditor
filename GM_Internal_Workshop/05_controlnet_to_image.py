import os
# Set cache path for Hugging Face models
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import make_image_grid, load_image
from transformers import AutoImageProcessor
import torch
import cv2  # for Canny edge detection


# Define paths and settings
controlnet_model_id = "lllyasviel/sd-controlnet-canny"
model_id = "SG161222/RealVisXL_V4.0"
output_dir = "GM_Internal_Workshop/images"
os.makedirs(output_dir, exist_ok=True)

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Prepare Canny edge detection input
input_image_path = "GM_Internal_Workshop\images\pavilions.png"  # Replace with your image path
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
low_threshold = 100
high_threshold = 200
edges = cv2.Canny(image, low_threshold, high_threshold)
edges_pil = Image.fromarray(edges)

# Save edges image for visualization (optional)
edges_pil.save(f"{output_dir}/edges_input.png")

# Define prompt and settings
prompt = (
    "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"
)
negative_prompt = "low res, bad quality"

# Generate image with ControlNet
generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=edges_pil,
    guidance_scale=5,
    num_inference_steps=25,
    height=1024,
    width=1024,
    controlnet_conditioning_scale = 0.8,
).images[0]

# Save the generated image
output_image_path = f"{output_dir}/generated_image_6.png"
generated_image.save(output_image_path)
generated_image.show()
image_grid = make_image_grid([edges_pil, generated_image], rows=1, cols=2)
image_grid.show()
