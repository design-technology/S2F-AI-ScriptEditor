import os

cache_path = 'C:\hg_models'  

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import make_image_grid, load_image
import torch

# Load the model
model_id = "SG161222/RealVisXL_V4.0"
# dtype, Model Instantiation dtype, for memory optimization.Normally gets instantiated with torch.float32 format
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# define device
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")

elif torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS ")

else:
    device = "cpu"
    print("Using CPU - this can take hours xD")

pipe = pipe.to(device)
# speed up inference
pipe.enable_model_cpu_offload()
    
# Define the prompt
prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"
negative_prompt = "low res, bad quality"
image_path = "0-Assets/pavilions.png"
init_image = load_image(image_path)
seed = 58964
generator = torch.Generator(device).manual_seed(seed)
# Generate the image
image = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    image = init_image,
    guidance_scale = 2.5,
    height = 1024,
    width = 1024,
    num_inference_steps = 25,
    num_images_per_prompt = 1,
    strength=1.0,
    output_type = "pil",
    generator = generator

    ).images[0]

# Save the image
# image.save(f"GM_Internal_Workshop/images/test_image2image_5.png")
image_grid = make_image_grid([init_image, image], rows=1, cols=2)
image.show()
image_grid.show()
