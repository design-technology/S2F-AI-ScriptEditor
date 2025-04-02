import io
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline, OnnxStableDiffusionPipeline

from PIL import Image

from System.IO import MemoryStream
import System.Drawing.Imaging as sdi
import System.Drawing as sd
from System import Array, Byte

import Eto.Drawing as ed

from Rhino import RhinoApp

# Set the Python executable
import torch.multiprocessing as mp
import rhinocode
python = rhinocode.get_python_executable()
mp.set_executable(python)

class RenderPipe:
    def __init__(self, use_onnx: bool = False, model_id: str = "stabilityai/sd-turbo"):
        self.use_onnx = use_onnx
        
        self.model_id = model_id
        self.pipe = self.get_pipe()

    def get_pipe(self):
        print(f"Using model : {self.model_id}")

        if torch.backends.mps.is_available():
            device = "mps"  # Apple's Metal Performance Shaders for Mac GPU acceleration
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU acceleration using CUDA
        else:
            device = "cpu"  # Fallback to CPU if no GPU is available

        print(f"Using device: {device}")

        if self.use_onnx:
            print("Using ONNX")
            try:
                # ONNX (Open Neural Network Exchange) allows optimized inference across platforms
                # More info: https://onnx.ai/
                execution_provider = "CPUExecutionProvider" if device == "cpu" else "DmlExecutionProvider"
                pipe = OnnxStableDiffusionPipeline.from_pretrained(self.model_id, provider=execution_provider)
                return pipe
            except Exception as e:
                print(f"ONNX loading failed: {e}")
        
        # Standard PyTorch pipeline for Stable Diffusion with img2img support
        # More info: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion

        # Use lower precision for speed on GPUs (CPUs don't support 16 bit floats)
        if device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16

        print(f"Using precision {dtype}")

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        pipe.to(device)
        return pipe

    def bitmap_to_pil(self, bitmap: sd.Bitmap) -> Image.Image:
        net_stream = MemoryStream()
        
        bitmap.Save(net_stream, sdi.ImageFormat.Png)
        bitmap.Save("/Users/sykes/Desktop/sd.png", sdi.ImageFormat.Png)
        
        byte_array = net_stream.ToArray()
        pil_image = Image.open(io.BytesIO(byte_array))

        return pil_image.convert("RGB")
        
    def set_model(self, model_id: str):
        if self.model_id == model_id:
            return
        
        self.model_id = model_id
        self.pipe = self.get_pipe()

    def generate_image(self, init_bitmap: sd.Bitmap, seed: int, prompt: str, negative_prompt: str, steps: int = 25, strength: float = 0.75):
        try:
            width, height = int(512), int(512)  # Fixed dimensions

            init_image = self.bitmap_to_pil(init_bitmap)
            init_image = init_image.resize((width, height), Image.LANCZOS)

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    guidance_scale=int(5),
                    num_inference_steps=int(steps),
                    num_images_per_prompt=int(1),
                    strength=float(strength),
                    height=int(height),
                    width=int(width),
                    generator=generator,  # Pass the seed generator
                )

                return result.images[0]
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def pil_to_bitmap(self, pil_image: Image.Image) -> ed.Bitmap:
        if pil_image is None:
            return None
        
        stream = io.BytesIO()
        pil_image.save(stream, format="PNG")
        byte_array = stream.getvalue()
        byte_array_net = Array[Byte](byte_array)  # Convert to .NET Byte[]

        return ed.Bitmap(byte_array_net)
