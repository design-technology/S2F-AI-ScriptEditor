import os
cache_path = 'Z:\\Development Projects\\huggingface'
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

from huggingface_hub import login
# Authenticate with Hugging Face token
login(token="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")

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    text_encoder_3=t5_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle, the roof is built using woven bamboo elements, surrounded by majestic mountains rising in the background and a serene river flowing in the foreground, the trees are way taller than the pavilions, earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle, volumetric sunlight goes across the jungle, creating fascinating light rays, 4k, high resolution, realistic render, architectural visualization"

image = pipeline(
    prompt=prompt,
    num_inference_steps=8,
    guidance_scale=0.0,
    max_sequence_length=512,
).images[0]
image.save("D3_test_6.png")
