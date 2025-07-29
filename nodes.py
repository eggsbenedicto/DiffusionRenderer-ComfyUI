import os
import sys

# Add the directory of this file to the sys.path
# This is necessary so that the cosmos_predict1 module can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import os
import imageio

from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING, envmap_vec
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

# Helper function to convert tensors to PIL images
def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

# Helper function to convert PIL images to tensors
def pil_to_tensor(image):
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    if tensor.ndim == 4 and tensor.shape[3] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor

class LoadDiffusionRendererModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "Models are loaded from 'ComfyUI/models/diffusion_models'"}),
            }
        }

    RETURN_TYPES = ("DIFFUSION_RENDERER_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Cosmos1"

    def load_pipeline(self, model):
        pipeline = DiffusionRendererPipeline(
            checkpoint_dir=os.path.join(folder_paths.models_dir, "diffusion_models"),
            checkpoint_name=model,
            # These are default values from the script, can be exposed as inputs later
            guidance=2.0,
            num_steps=20,
            height=1024,
            width=1024,
            num_video_frames=1,
            seed=42,
        )
        return (pipeline,)

class Cosmos1InverseRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("DIFFUSION_RENDERER_PIPELINE",),
                "image": ("IMAGE",),
                "noise_level": ("INT", {"default": 100, "min": 0, "max": 1000}),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "normal", "roughness", "metallic", "base_color")
    FUNCTION = "run_inverse_pass"
    CATEGORY = "Cosmos1"

    def run_inverse_pass(self, pipeline, image, noise_level, guidance=2.0, seed=42):
        # Configure pipeline with runtime parameters
        pipeline.guidance = guidance
        pipeline.seed = seed

        # Standardize image input to a 5D tensor (B, T, H, W, C)
        if isinstance(image, list):
            image = torch.stack(image)
        if image.ndim == 4:
            image = image.unsqueeze(0)

        B, T, H, W, C = image.shape
        pipeline.num_video_frames = T
        pipeline.height = H
        pipeline.width = W

        pbar = ProgressBar(B)
        inference_passes = ["depth", "normal", "basecolor", "roughness", "metallic"]
        batch_outputs = {pass_name: [] for pass_name in inference_passes}

        for i in range(B):
            image_slice = image[i] # Shape: (T, H, W, C)
            
            # Pipeline expects image shape (1, T, C, H, W)
            image_tensor = image_slice.permute(0, 3, 1, 2).unsqueeze(0)

            data_batch = {
                "image": image_tensor,
                "context_index": torch.zeros(1, 1, dtype=torch.long),
                "noise_level": torch.tensor([noise_level], dtype=torch.long),
            }

            for gbuffer_pass in inference_passes:
                context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
                data_batch["context_index"].fill_(context_index)

                output_tensor = pipeline.generate_video(
                    data_batch=data_batch,
                    normalize_normal=(gbuffer_pass == 'normal'),
                    seed=seed + i, # Use a different seed for each image in the batch
                )
                batch_outputs[gbuffer_pass].append(output_tensor)
            pbar.update(1)

        # Stack results from all videos in the batch
        outputs = {pass_name: torch.cat(batch_outputs[pass_name], dim=0) for pass_name in inference_passes}
        
        return (outputs["depth"], outputs["normal"], outputs["basecolor"], outputs["roughness"], outputs["metallic"])


class Cosmos1ForwardRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("DIFFUSION_RENDERER_PIPELINE",),
                "depth": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "base_color": ("IMAGE",),
                "env_map": ("IMAGE",),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_forward_pass"
    CATEGORY = "Cosmos1"

    def run_forward_pass(self, pipeline, depth, normal, roughness, metallic, base_color, env_map, guidance=2.0, seed=42):
        # Standardize all inputs to 5D tensors (B, T, H, W, C)
        gbuffer_tensors = {
            "depth": depth,
            "normal": normal,
            "roughness": roughness,
            "metallic": metallic,
            "base_color": base_color,
            "env_map": env_map,
        }

        # Standardize tensors and handle lists
        for name, tensor in gbuffer_tensors.items():
            if isinstance(tensor, list):
                tensor = torch.stack(tensor)
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0) # Add batch dimension
            gbuffer_tensors[name] = tensor

        # Get dimensions from the primary input
        B, T, H, W, C = gbuffer_tensors["depth"].shape

        # Configure pipeline with runtime parameters
        pipeline.guidance = guidance
        pipeline.seed = seed
        pipeline.num_video_frames = T
        pipeline.height = H
        pipeline.width = W

        device = mm.get_torch_device()
        pbar = ProgressBar(B)
        output_videos = []

        # Process each video in the batch
        for i in range(B):
            # Prepare G-buffer inputs for the current video
            data_batch = {}
            for name in ["depth", "normal", "roughness", "metallic", "base_color"]:
                tensor_slice = gbuffer_tensors[name][i] # (T, H, W, C)
                # Pipeline expects (1, T, C, H, W) and range [-1, 1]
                processed_tensor = tensor_slice.permute(0, 3, 1, 2).unsqueeze(0)
                data_batch[name.replace('_', '')] = processed_tensor * 2.0 - 1.0

            # Process the environment map for the current video
            import imageio
            temp_env_map_path = os.path.join(folder_paths.get_temp_directory(), f"temp_env_map_{i}.hdr")
            
            env_map_slice = gbuffer_tensors["env_map"][i] # (T, H, W, C)
            env_map_np = env_map_slice.cpu().numpy()
            # If env_map is a single frame for the whole video, take the first frame
            if env_map_np.shape[0] == 1 and T > 1:
                env_map_np = env_map_np[0]

            imageio.imwrite(temp_env_map_path, env_map_np)

            envlight_dict = process_environment_map(
                temp_env_map_path,
                resolution=(H, W),
                num_frames=T,
                fixed_pose=True,
                rotate_envlight=False,
                env_format=['proj', ],
                device=device,
            )
            
            env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
            env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
            env_nrm = envmap_vec([H, W], device=device)
            env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
            
            data_batch['env_ldr'] = env_ldr
            data_batch['env_log'] = env_log
            data_batch['env_nrm'] = env_nrm

            # Generate the final image for the current video
            output_tensor = pipeline.generate_video(
                data_batch=data_batch,
                seed=seed + i, # Use a different seed for each image in the batch
            )
            output_videos.append(output_tensor)
            pbar.update(1)

        # Stack results into a single batch tensor
        final_output = torch.cat(output_videos, dim=0)

        # The output is in range [-1, 1], so we scale it back to [0, 1]
        final_output = (final_output + 1.0) / 2.0

        return (final_output,)


NODE_CLASS_MAPPINGS = {
    "LoadDiffusionRendererModel": LoadDiffusionRendererModel,
    "Cosmos1InverseRenderer": Cosmos1InverseRenderer,
    "Cosmos1ForwardRenderer": Cosmos1ForwardRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiffusionRendererModel": "Load Diffusion Renderer Model",
    "Cosmos1InverseRenderer": "Cosmos1 Inverse Renderer",
    "Cosmos1ForwardRenderer": "Cosmos1 Forward Renderer",
}
