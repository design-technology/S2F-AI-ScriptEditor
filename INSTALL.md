# Install & Setup

## 1. Python
### 1.1 Environment Handler 
Environments in Python have many different solutions, we're going to use Conda, specially the MiniConda version as it's lighter.

#### [Anaconda Downloads Page](https://www.anaconda.com/download/success)

- [Windows 3.12](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
- [Mac 3.13 Silicon](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)
- [Mac 3.12 Intel (old)](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)

### 1.2 Environment Creation

Create the Environment using the env.yml file
``` bash
conda create -n genai python=3.9 --yes
```

Activate the environment and make it the current
``` bash
conda activate genai
```

### 1.3 Installing Packages

Install the packages
``` bash
pip install -r requirements.txt
```

### 1.4 Installing pyTorch for Windows

All the information below is obtained from [this guide](https://pytorch.org/get-started/locally/#windows-anaconda) simply made a little easier to follow.

#### Windows AMD
``` bash
pip install torch torchvision torchaudio
```

#### Windows NVidia

You need your CUDA version first, run this command in Command Prompt (cmd) or Windows Terminal.

``` bash
nvidia-smi
```

The following information will appear

``` bash
Thu Apr  3 14:35:59 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.07                 Driver Version: 566.07         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   47C    P8              4W /   35W |       0MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     12664    C+G   C:\Windows\System32\ShellHost.exe           N/A      |
|    0   N/A  N/A     22240    C+G   C:\Windows\explorer.exe                     N/A      |
+-----------------------------------------------------------------------------------------+
```

If nothing appears, review these [Nvidia Docs](https://www.supportyourtech.com/articles/how-to-check-my-cuda-version-windows-10-a-step-by-step-guide/) to find your version
You also may need to install CUDA [Link](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

If your CUDA version is 12.6 or higher run this

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If your CUDA version is 12.4 or higher run this

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If your CUDA version is 11.8 or higher run this

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. Models

Downloading AI models from [HuggingFace](https://huggingface.co) is easy, the best way is to clone through `git` using the command line.

100Gb - Premium Quality
[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

``` bash
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```


55Gb - Average Quality
[SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0)

``` bash
git clone https://huggingface.co/SG161222/RealVisXL_V5.0
```


25Gb - Works
[stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)

``` bash
git clone https://huggingface.co/stabilityai/sd-turbo
```
