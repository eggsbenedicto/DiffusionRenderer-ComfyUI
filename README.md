Experimental wrapper for diffusion-renderer in ComfyUI. Still under testing and feature expansion

This node requires system-level graphics development libraries to compile a core dependency (`nvdiffrast`). nvdiffrast is only necessary for the environmental lighting in the "forward" pass, so if you only plan on using the "inverse" pass, you can simply remove nvdiffrast, you can remove references to it from the codebase. 

Currently I have only tested on Ubuntu. Information online indicates that installing nvdiffrast on windows can be a little more complicated. On Ubuntu/Debian, before installing the nodepack, open your terminal and run:

    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        curl \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev


You'll want to download the official Diffusion Renderer checkpoints into your diffusion_models directory in ComfyUI. They can be found here: https://huggingface.co/collections/zianw/cosmos-diffusionrenderer-6849f2a4da267e55409b8125 

You'll also want to download Nvidia's Video Tokenizers to your vae directory in ComfyUI. Make sure this entire repo is in your vae subfolder with the name "Cosmos-1.0-Tokenizer-CV8x8x8". This is the natural expected result if you download the repo using 'hf download nvidia/Cosmos-1.0-Tokenizer-CV8x8x8 --local-dir Cosmos-1.0-Tokenizer-CV8x8x8' from your ComfyUI/models/vae directory. [EDIT: You can delete the JIT tokenizers. They are currently unused. Keep the remaining file structure the same.]

