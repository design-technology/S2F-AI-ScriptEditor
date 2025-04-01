import io
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline, OnnxStableDiffusionPipeline

from PIL import Image

from System.IO import MemoryStream, SeekOrigin

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

    def __init__(self):
        pass
    
    def get_pipe(self, USE_ONNX:bool=False, MODEL_ID:str="stabilityai/sd-turbo"):

        print(f"Using model : {MODEL_ID}")

        if torch.backends.mps.is_available():
            device = "mps"  # Apple's Metal Performance Shaders for Mac GPU acceleration
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU acceleration using CUDA
        else:
            device = "cpu"  # Fallback to CPU if no GPU is available

        print(f"Using device: {device}")

        if USE_ONNX:
            print(f"Using ONNX")

            try:
                # ONNX (Open Neural Network Exchange) allows optimized inference across platforms
                # More info: https://onnx.ai/
                execution_provider = "CPUExecutionProvider" if device == "cpu" else "DmlExecutionProvider"
                pipe = OnnxStableDiffusionPipeline.from_pretrained(MODEL_ID, provider=execution_provider)

                return pipe
            except:
                print("ONNX not installed locally, include onnxruntime as a requirement or install with pip")
        
        # Standard PyTorch pipeline for Stable Diffusion with img2img support
        # More info: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion

        # Use lower precision for speed on GPUs (CPUs don't support 16 bit floats)
        if device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16

        print(f"Using precision {dtype}")

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe.to(device)
        
        return pipe

    def bitmap_to_pil(self, bitmap: sd.Bitmap) -> Image.Image:
        net_stream = MemoryStream()
        bitmap.Save(net_stream, sdi.ImageFormat.Png)
        bitmap.Save("/Users/sykes/Desktop/sd.png", sdi.ImageFormat.Png)
        byte_array = net_stream.ToArray()

        pil_image = Image.open(io.BytesIO(byte_array))
        return pil_image.convert("RGB")

    def generate_image(self,
                        init_bitmap: sd.Bitmap,
                        seed: int,
                        prompt: str,
                        negative_prompt: str,
                        model_id: str,
                        steps: int = 25,
                        strength: float = 0.75) -> Image.Image:

        width = 512
        height = 512
        
        try:
            init_image = self.bitmap_to_pil(init_bitmap)

            # Resize to avoid memory issues
            init_image = init_image.resize((width, height), Image.LANCZOS)
            init_image.save("/Users/sykes/Desktop/pil_smol.png")

            with torch.inference_mode():
                pipe = self.get_pipe(False, model_id)
                result = pipe(
                    prompt = prompt,
                    # negative_prompt = negative_prompt,
                    image = init_image,
                    guidance_scale = 5,
                    num_inference_steps = steps,
                    num_images_per_prompt = 1,
                    strength=strength,
                    height = width,
                    width = height,
                    # generator: torch.Generator = None, // Seed
                    controlnet_conditioning_scale = 0.6,
                )

                return result.images[0]
        except:
            return None

    def pil_to_bitmap(self, pil_image: Image.Image) -> ed.Bitmap:
        if pil_image is None:
            return
        
        stream = io.BytesIO()
        pil_image.save(stream, format="PNG")
        pil_image.save("/Users/sykes/Desktop/pil.png")
        
        byte_array = stream.getvalue()
        byte_array_net = Array[Byte](byte_array)  # Convert to .NET Byte[]

        eto_bitmap = ed.Bitmap(byte_array_net)
        eto_bitmap.Save("/Users/sykes/Desktop/eto.png", sdi.ImageFormat.Png)
        return eto_bitmap
