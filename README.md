![alt text](assets/readme_images/STF-AI-03.png)

Documentation file for basic AI image generation tasks using the auto [stable diffusion pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview)

# Step-by-step guide to run AI models locally:

# Prerequisites:
## Install cuda 12.6 (only for Windows uers):**

- **check cuda version:**
nvidia-smi
- **Install cuda 12.6:**
    - windows 11: 
    https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
    - windows 10: 
    https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

## Install miniconda for virtual environment management for Windows or Mac:**
https://www.anaconda.com/download/success

Make sure you enable this option during the installation:

![alt text](assets/readme_images/anaconda.png)

## **01-Prepare a conda environment for the app:**

- **create an env**: 
	- conda create --name generative_ai python==3.9
- **check if it's installed:** 
	- conda env list 
- **activate the environment:** 
	- conda activate generative_ai
- **Install the required libraries:**
	- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	- pip install diffusers
	- pip install accelerate
	- pip install transformers 
	- pip install opencv-python
    - pip install pyOpenSSL
    - pip install controlnet_aux
    - pip install pillow
    - pip install mediapipe
    - pip install numpy
    - pip install timm
    - pip install SSL
	One line to download all the libraries: 
    - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && pip install diffusers accelerate transformers opencv-python pyOpenSSL controlnet_aux pillow mediapipe numpy timm


## **02-Huggingface caching:**

add this part at the beginning of your code to change the directory of all the downloaded models while working with huggingface

- cache_path = 'choose a path'
- cache_path = 'Z:\\Development Projects\\huggingface' (example)
- os.environ["TRANSFORMERS_CACHE"] = cache_path
- os.environ["HF_HUB_CACHE"] = cache_path
- os.environ["HF_HOME"] = cache_path
