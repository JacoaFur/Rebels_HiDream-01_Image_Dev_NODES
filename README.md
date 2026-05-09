# Rebels_HiDream-01_Image_Dev_NODES
node set to run the HiDream-01 Image Dev GGUF from smthem



Rebel HiDream-O1 GGUF Nodes for ComfyUI
Created by Rebel AI

This repository provides custom ComfyUI nodes to run HiDream-O1-Image-Dev GGUF models locally.

HiDream-O1 is a VAE-less, Pixel-Level Unified Transformer. Because it generates raw pixels token-by-token at massive resolutions, running it locally requires careful memory management. These nodes feature upfront dequantization (converting GGUF weights to native PyTorch tensors in system RAM during the load phase) to completely bypass NumPy single-threaded CPU bottlenecks during generation, allowing your GPU to run at maximum efficiency.

📦 Prerequisites
Before installing the custom nodes, you need the upstream model code and the weights.

Clone the Upstream HiDream-O1 Repo:
The nodes rely on the official pipeline logic. Clone this anywhere on your local system:


git clone https://github.com/HiDream-ai/HiDream-O1-Image.git

Download the GGUF Model:
Download HiDream-O1-Image-Dev GGUF (Q6_K) from Hugging Face: https://huggingface.co/smthem/HiDream-O1-Image-Dev/blob/main/HiDream-O1-Image-Dev-Q6_K.gguf
Place the .gguf file in: ComfyUI/models/diffusion_models/

🛠️ Installation
Option 1: ComfyUI Windows Portable
Note: Ensure your ComfyUI portable installation is located on a strict local system drive (e.g., C:\ or D:\). Do not install or run these nodes from a OneDrive-synced folder, as it will cause virtual environment and pathing errors.

Open a command prompt and navigate to your portable custom nodes directory:


cd \ComfyUI_windows_portable\ComfyUI\custom_nodes

Clone this repository:

git clone https://github.com/YourUsername/Rebels_HiDream_01_Image_Dev_NODES.git

Install the requirements using the embedded Python environment:


cd Rebels_HiDream_01_Image_Dev_NODES
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

Option 2: Desktop / Standard Python Environment

Navigate to your ComfyUI custom nodes directory:

cd ComfyUI/custom_nodes

Clone this repository:


git clone https://github.com/YourUsername/Rebels_HiDream_01_Image_Dev_NODES.git



Activate your ComfyUI virtual environment and install the requirements:

cd Rebels_HiDream_01_Image_Dev_NODES
pip install -r requirements.txt

🧩 Node Documentation
Rebel HiDream-O1 Loader (GGUF)
Loads the GGUF model and performs upfront dequantization.

gguf_name: Select your .gguf model from the diffusion_models folder.

tokenizer_path: Default is HiDream-ai/HiDream-O1-Image-Dev. It will automatically fetch the tokenizer config from Hugging Face.

upstream_repo_path: The absolute local path to where you cloned the HiDream-O1-Image repository in the prerequisites (e.g., C:\Users\name\HiDream-O1-Image).

device: Set to cuda for GPU acceleration.

offload:

aggressive: Heavily utilizes system RAM offloading (Recommended for 8GB VRAM cards like the RTX 3070).

balanced: Standard memory splitting.

minimal: Keeps most of the model in VRAM.

Note: The loader will hang for a moment at 100% while it unpacks the uint8 GGUF bytes into native PyTorch tensors in your system RAM. This is normal and prevents the CPU from bottlenecking your GPU during the actual generation steps.

Rebel HiDream-O1 Sampler
Connect the model output from the Loader here.

steps: 20-30 is the recommended sweet spot.

cfg: Keep between 2.5 - 4.0. Higher CFG combined with heavy styling tags can cause "deep-fried" or crushed-shadow artifacts due to the literal pixel-rendering nature of the model.

shift: keep at 3.0. Controls the timestep scheduling curve. 

scheduler_name:

default: Standard sampling.

flash: Injects specific noise profiles. Note: The noise_scale_start, noise_scale_end, and noise_clip_std parameters only apply if the scheduler is set to flash.

⚠️ Known Limitations & Behaviors
Resolution Snapping: HiDream-O1 enforces strict token sequence lengths based on its pre-trained position embeddings. If you input a lower resolution (like 512x512 or 1024x1024), the upstream pipeline will automatically "snap" and force the generation to 2048x2048.

Compute Times: Because the model renders a raw 2048x2048 image pixel-by-pixel without a VAE, generation times will be significantly longer than standard latent models (like SDXL or Flux).

Prompting Style: Avoid stacking heavy texture tags (e.g., "8k resolution, ultra-detailed, gritty textures") unless you want a heavily illustrated look. For photorealism, use clean, simple photographic terms.
