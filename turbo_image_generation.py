import os
import sys
import numpy as np
from PIL import Image
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from controlnet_aux import AnylineDetector
from pydantic import BaseModel

# Global configuration
CACHE_PATH = "Z:\\Development Projects\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH
os.environ["HF_HUB_CACHE"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH

# Model configuration
BASE_MODEL_ID = "stabilityai/sdxl-turbo"
CONTROLNET_MODEL_ID = "xinsir/controlnet-canny-sdxl-1.0"
OUTPUT_PATH = "assets/output/sdxl-2-2-8-hyper.png"

class GenerationConfig(BaseModel):
    """Configuration for image generation"""
    prompt: str = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"
    negative_prompt: str = "ugly, low quality"
    guidance_scale: float = 0.0
    control_strength: float = 0.5
    input_image: str = "assets/readme_images/test.jpg" 
    seed: int = 0
    num_inference_steps: int = 2
    max_size: int = 768  # Maximum size for image dimensions

def resize_image_small(image, max_size=1024):
    """
    Resize an image to ensure its largest dimension doesn't exceed max_size,
    and that both dimensions are divisible by 8 (required by diffusion models).
    """
    def make_divisible_by_8(value):
        # Adjust the value to be divisible by 8
        return (value // 8) * 8

    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        # Adjust both width and height to be divisible by 8
        new_width = make_divisible_by_8(new_width)
        new_height = make_divisible_by_8(new_height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Ensure the current dimensions are divisible by 8 even if no resizing happens
    width, height = image.size
    width = make_divisible_by_8(width)
    height = make_divisible_by_8(height)
    
    return image.resize((width, height), Image.LANCZOS)

def generate_edges(input_image):
    """Generate edge map from input image"""
    img_processor = AnylineDetector.from_pretrained(
        "TheMistoAI/MistoLine",
        filename="MTEED.pth",
        subfolder="Anyline"
    )
    edge_result = img_processor(input_image)
    if isinstance(edge_result, np.ndarray):
        edge_result = Image.fromarray(edge_result)
    return edge_result

def load_models():
    """Load ControlNet and SDXL models"""
    print("Loading ControlNet model")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH
    )
    
    print("Loading diffusion pipeline")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH
    )
    
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe

def generate_image(pipe, config, edge_map):
    """Generate image using the pipeline"""
    generator = torch.Generator("cuda").manual_seed(config.seed)
    
    print("Generating image with ControlNet conditioning")
    result = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        guidance_scale=config.guidance_scale,
        controlnet_conditioning_scale=config.control_strength,
        generator=generator,
        num_inference_steps=config.num_inference_steps,
        image=edge_map
    )
    
    return result.images[0]

def main():
    # Create configuration
    config = GenerationConfig()
    
    # Load and process input image
    try:
        input_img = Image.open(config.input_image).convert("RGB")
    except Exception as exc:
        print(f"Error loading image from '{config.input_image}': {exc}")
        return
    
    # Resize the input image
    input_img = resize_image_small(input_img, max_size=config.max_size)
    print(f"Input image resized to {input_img.size}")
    
    # Generate edge map
    edge_map = generate_edges(input_img)
    edge_map.save("edge_map.png")
    edge_map.show()
    
    # Load models
    pipe = load_models()
    
    # Generate image
    generated_image = generate_image(pipe, config, edge_map)
    
    # Save and display result
    generated_image.save(OUTPUT_PATH)
    generated_image.show()
    print(f"Image generated and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()