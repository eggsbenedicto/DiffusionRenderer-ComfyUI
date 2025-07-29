---
applyTo: '**'
---
The client is a senior executive at a large tech company with incredible power over our career trajectory. If this task goes well, it could lead to a significant promotion and recognition within the industry. The client is known for being detail-oriented and expects high-quality work. Therefore, it is crucial to ensure that the code is functional, efficient, and well-documented. This means we can't afford to fire from the hip or take shortcuts. Before making any changes, we need to reason through the implications of those changes and ensure they align with the client's expectations. Therefore, you shouldn't make additional changes to the code without first receiving explicit permission to do so. The client will ask a number of questions which will not require immediate changes to the code, but will require careful consideration and reasoning.

The client's ultimate goal for this project is to create a ComfyUI wrapper for the Cosmos1 diffusion_renderer. This will be a set of nodes which will encompass the functionality of both the inverse and forward passes of the diffusion renderer. The nodes should be designed in-line with ComfyUI conventions where possible.

Wherever possible, dependencies should be minimized. Many of the requirements in the requirements.txt file should be re-evaluated to determine if they are truly necessary for the ComfyUI wrapper.

Node 1: Inverse Pass Node
- This node should handle the inverse pass of the diffusion renderer. It needs to call the Cosmos1 diffusion_renderer's inverse pass model with the correct inputs and parameters in accordance with the official inference_inverse_renderer.py script.
- It should accept inputs such as the initial image (as a ComfyUI IMAGE object), noise level, and any other parameters required
- It should output the depth, normal, material, and base color maps as separate ComfyUI IMAGE objects.

Node 2: Forward Pass Node
- This node should handle the forward pass of the diffusion renderer. It needs to call the Cosmos1 diffusion_renderer's forward pass model with the correct inputs and parameters in accordance with the official inference_forward_renderer.py script.
- It should accept inputs such as the depth, normal, material, base color maps, AND an environment map (as ComfyUI IMAGE objects) and any other parameters required.
- It should output the final rendered image as a ComfyUI IMAGE object.
