Experimental wrapper for diffusion-renderer in ComfyUI. Still under testing and feature expansion

This node requires system-level graphics development libraries to compile a core dependency (`nvdiffrast`). nvdiffrast is only necessary for the environmental lighting in the "forward" pass, so if you only plan on using the "inverse" pass, you can simply remove nvdiffrast, you can remove references to it from the codebase. 

Currently I have only tested on Ubuntu. Information online indicates that installing nvdiffrast on windows can be a little more complicated. On Ubuntu/Debian, before installing the nodepack, open your terminal and run:

    ```bash
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

    ```

