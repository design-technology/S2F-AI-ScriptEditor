![[assets/readme_images/STF-AI-03.png]]

Documentation file for basic AI image generation tasks using the auto [stable diffusion pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview)

# Step-by-step guide to run AI models locally:
## **01-Prepare a conda environment for the app:**

- **create an env**: 
	- conda create --name generative_ai python==3.9
- **check if it's installed:** 
	- conda env list 
- **activate the environment:** 
	- conda activate generative_ai
- **check existing cuda version:**
	- nvcc --version
- Install the required libraries:
	- pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
	- pip install diffusers
	- pip install accelerate
	- pip install transformers 
	- pip install opencv-python


## **02-Huggingface caching:**

add this part at the beginning of your code to change the directory of all the downloaded models while working with huggingface

- cache_path = 'Z:\\Development Projects\\huggingface' 
- os.environ["TRANSFORMERS_CACHE"] = cache_path
- os.environ["HF_HUB_CACHE"] = cache_path
- os.environ["HF_HOME"] = cache_path

## **03-Choosing a model:**

![[assets/readme_images/AI-models.png]]
https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads

When selecting a model from Hugging Face, consider the following factors to ensure it meets your requirements:
#### **1. Task Compatibility**
- Identify the task you want to solve (e.g., text-to-image, image-to-image, text generation, etc.).
- Search for models trained specifically for that task (e.g., Stable Diffusion for image generation, GPT for text generation).
#### **2. Model Variants**
- **Base Models**: Use base models for general tasks (e.g., `runwayml/stable-diffusion-v1-5` for image generation).
- **Specialized Models**: Explore fine-tuned models for domain-specific needs (e.g., anime, architectural designs, etc.).
- **Control Models**: Consider models like ControlNet for additional control over the output (e.g., edge, depth maps).
#### **3. Hardware Requirements**
- Check the **VRAM** needed to run the model:
    - **8 GB or less**: Use lightweight models (e.g., Stable Diffusion v1.4/v1.5).
    - **12+ GB**: Opt for high-performance models like SDXL for better quality.
- If running locally, ensure your system meets these requirements.
#### **4. Resolution and Quality**
- For high-resolution outputs or fine details, choose models like **SDXL**.
- For quick or low-resource tasks, prefer smaller, efficient models.
#### **5. Hugging Face Model Page**
- Check the model's description on Hugging Face:
    - **Performance Metrics**: Look for example outputs and benchmarks.
    - **Dependencies**: Ensure you install all required libraries.
    - **Guidance**: Follow the usage notes for the best results.
### **6. Licensing**
- Verify the model's license to ensure compliance with your project requirements.


## **04-Choosing a pipeline:**
[Pipelines overview](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview)

- **StableDiffusionPipeline / StableDiffusionXLPipeline:**  
	Generates images from text prompts with Stable Diffusion models (SD for smaller, SDXL for high-res and detailed outputs).
    
- **StableDiffusionImg2ImgPipeline / StableDiffusionXLImg2ImgPipeline**:  
    Transforms an input image based on a text prompt to enhance or modify its content (SD and SDXL versions available).
    
- **StableDiffusionControlNetPipeline / StableDiffusionXLControlNetPipeline**:  
    Allows precise control of image generation using additional conditions like edges, depth maps, or segmentation

## **05-Calling a model:**

You can use a model for simple tasks easily using StabelDiffusionPipelines. For more [info](https://huggingface.co/docs/diffusers/main/en/tutorials/autopipeline)

Loading a model code snippet:

`model_id = "sd-legacy/stable-diffusion-v1-5"`
`pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)`
`pipe = pipe.to("cuda")  # Move to GPU`
`prompt = "scene description"`


## **06-Reading the output:**

`image = pipe(prompt).images[0]`
`output = pipe(prompt)` = diffusers.pipelines.stable_diffusion.pipeline_output

StableDiffusionPipelineOutput(images=<PIL.Image.Image image mode=RGB size=512x512 at 0x24226B0A880>, nsfw_content_detected=False)

 `output_image = pipe(prompt).images` = List = [PIL object](https://pillow.readthedocs.io/en/stable/)
<PIL.Image.Image image mode=RGB size=512x512 at 0x24224FF9E20>

![[assets/readme_images/AI-render.png]]
## **07-Adjusting model parameters:**

Text-to-image calling parameters [reference](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__)

`image = pipe(`
    `prompt = prompt,`
    `negative_prompt = negative_prompt,`
    `#Alignment with the input text prompt`
    `guidance_scale = 7.5,`
    `height = 1024,`
    `width = 1024,`
    `num_inference_steps = 25,`
    `num_images_per_prompt = 1,`
    `output_type = "pil",`
    `).images[0]`
   
- **prompt** (`str` or `List[str]`) — The prompt or prompts to guide image generation.
  
- **negative_prompt** (`str` or `List[str]`) — The prompt or prompts to guide what to not include in image generation.
  
- **height** (`int`) — The height in pixels of the generated image.
  
- **width** (`int`) — The width in pixels of the generated image..
  
- **num_inference_steps** (`int`, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
  
- **guidance_scale** (`float`, defaults to 7.5) — A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality.
  
- **num_images_per_prompt** (`int`, defaults to 1) — The number of images to generate per prompt.
  
- **generator** (`torch.Generator` or `List[torch.Generator]`) — A torch generator to make generation deterministic.
## **08-Other model setup parameters:**

#### **A-Assigning the Right Scheduler to a Pipeline**

![[assets/readme_images/scheduler-cats.png]]
Schedulers define how latent noise is reduced during the image generation process, directly impacting the quality, speed, and style of the output. 
https://www.kaggle.com/code/aisuko/schedulers-performance
https://github.com/huggingface/diffusers/pull/5541
##### **Commonly Used Schedulers**

- **EulerDiscreteScheduler (`K_EULER`)**:  
    A simple and efficient scheduler providing fast, balanced results.
    
- **EulerAncestralDiscreteScheduler (`K_EULER_ANCESTRAL`)**:  
    Produces more artistic and detailed outputs with a touch of randomness; good for creative styles.
    
- **DPMSolverMultistepScheduler (`DPMPP_2M`)**:  
    A high-performance scheduler for faster generation while maintaining good quality.
    
- **DDIMScheduler (`DDIM`)**:  
    Focused on deterministic outputs; suitable for iterative workflows like inpainting or img2img.
    
- **KDPM2DiscreteScheduler**:  
    Balances performance and image fidelity, useful for high-resolution tasks.
    
- **PNDMScheduler**:  
    Ideal for tasks needing high precision, such as structured or sharp outputs.

default scheduler
`pipeline.scheduler`
`EulerDiscreteScheduler {`
  `"_class_name": "EulerDiscreteScheduler",`
  `"_diffusers_version": "0.32.2",`
  `"beta_end": 0.012,`
  `"beta_schedule": "scaled_linear",`       
  `"beta_start": 0.00085,`
  `"clip_sample": false,`
  `"final_sigmas_type": "zero",`
  `"interpolation_type": "linear",`
  `"num_train_timesteps": 1000,`
  `"prediction_type": "epsilon",`
  `"rescale_betas_zero_snr": false,`
  `"sample_max_value": 1.0,`
  `"set_alpha_to_one": false,`
  `"sigma_max": null,`
  `"sigma_min": null,`
  `"skip_prk_steps": true,`
  `"steps_offset": 1,`
  `"timestep_spacing": "leading",`
  `"timestep_type": "discrete",`
  `"trained_betas": null,`
  `"use_beta_sigmas": false,`
  `"use_exponential_sigmas": false,`
  `"use_karras_sigmas": false`
`}`

change the default scheduler:
`pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)`