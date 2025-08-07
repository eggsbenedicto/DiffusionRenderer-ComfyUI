import torch
import numpy as np
from PIL import Image
import os
import sys
import imageio
import logging
import json

import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.utils import ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

# --- Corrected/New Imports ---
from diffusers import AutoencoderKLCosmos
from .CleanVAE import CleanVAE # Used as a simple wrapper for the Image VAE
from .pretrained_vae import VideoJITTokenizer, JointImageVideoTokenizer

from .diffusion_renderer_pipeline import CleanDiffusionRendererPipeline
from .model_diffusion_renderer import CleanDiffusionRendererModel
from .diffusion_renderer_config import get_inverse_renderer_config

from .preprocess_envmap import (
    render_projection_from_panorama,
    tonemap_image_direct,
    latlong_vec,
    clear_environment_cache,
    get_cache_stats
)

# Extracted utilities from the original codebase without dependencies
# Official mapping from cosmos_predict1/diffusion/inference/diffusion_renderer_utils/rendering_utils.py
GBUFFER_INDEX_MAPPING = {
    "basecolor": 0,
    "metallic": 1,
    "roughness": 2,
    "normal": 3,
    "depth": 4,
}

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

    # FINAL FIX: Removed the unused 'vae_model' parameter from the function signature.
    def load_pipeline(self, model):
        device = mm.get_torch_device()
        dtype = torch.bfloat16
        print(f"Targeting device: {device}, dtype: {dtype}")

        # --- VAE LOADING (Corrected Composite VAE Logic) ---
        
        # FINAL FIX: Removed the obsolete 'if vae_model...' check to ensure this block always runs.
        
        # This path should point to the root of the tokenizer repo
        # e.g., ComfyUI/models/vae/Cosmos-1.0-Tokenizer-CV8x8x8/
        vae_main_dir = folder_paths.get_full_path("vae", "Cosmos-1.0-Tokenizer-CV8x8x8")
        print(f"Loading composite VAE from main directory: {vae_main_dir}")

        # --- PART A: Load the IMAGE VAE (from the '/vae' subdirectory) ---
        image_vae_subfolder_path = os.path.join(vae_main_dir, "vae")

        if not os.path.isdir(image_vae_subfolder_path):
            raise FileNotFoundError(f"Image VAE subfolder not found at: {image_vae_subfolder_path}")

        # Instantiate the newly refactored CleanVAE, which handles its own loading.
        image_vae_instance = CleanVAE(model_path=image_vae_subfolder_path)
        image_vae_instance.to(device)
        image_vae_instance.reset_dtype(dtype) # Use the new method to set dtype
        print("✅ Image VAE loaded successfully via from_pretrained.")

        # --- PART B: Load the VIDEO VAE (from 'autoencoder.jit') ---
        video_vae_jit_path = os.path.join(vae_main_dir, "autoencoder.jit")
        if not os.path.exists(video_vae_jit_path):
            raise FileNotFoundError(f"Video VAE not found. Searched for 'autoencoder.jit' in: {vae_main_dir}")

        # Instantiate the official tokenizer class
        video_vae_instance = VideoJITTokenizer(
            name="video_vae",
            latent_ch=16,
            is_bf16=(dtype == torch.bfloat16),
            spatial_compression_factor=8,
            temporal_compression_factor=8,
            pixel_chunk_duration=17,
        )
        print(f"Loading VIDEO VAE from: {video_vae_jit_path}")

        # Load the JIT module and assign its components to the tokenizer instance
        video_jit_module = torch.load(video_vae_jit_path, map_location=device, weights_only=False)
        video_jit_module.eval()
        video_jit_module.to(dtype=dtype)
        
        video_vae_instance.encoder = video_jit_module.encoder
        video_vae_instance.decoder = video_jit_module.decoder
        video_vae_instance.register_mean_std(vae_main_dir)
        print("✅ Video VAE loaded successfully.")

        # --- PART C: Create the Dispatcher ---
        # This class holds both VAEs and is the final VAE object passed to the pipeline.
        vae_instance = JointImageVideoTokenizer(
            image_vae=image_vae_instance,
            video_vae=video_vae_instance,
            name="joint_vae",
            latent_ch=16,
            squeeze_for_image=True
        )
        vae_instance.to(device)
        vae_instance.eval()
        print("✅ Joint VAE Dispatcher created.")

        # --- MEMORY-EFFICIENT MODEL LOADING using META DEVICE ---
        checkpoint_path = folder_paths.get_full_path("diffusion_models", model)
        
        print(f"Loading checkpoint weights to CPU from: {checkpoint_path}")
        state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        print("Instantiating model skeleton on 'meta' device...")
        basic_config = get_inverse_renderer_config()
        with torch.device("meta"):
             model_instance = CleanDiffusionRendererModel(basic_config)

        print(f"Materializing model on {device} (step 1/2)...")
        model_instance.to_empty(device=device)
        print(f"Casting materialized model to {dtype} (step 2/2)...")
        model_instance.to(dtype=dtype)
        print("Loading weights into the full materialized GPU model...")
        model_instance.load_state_dict(state_dict, strict=True)
        
        del state_dict
        mm.soft_empty_cache()
        model_instance.eval()
        print(f"✅ Pre-loaded diffusion model successfully to {device}")

        pipeline = CleanDiffusionRendererPipeline(
            checkpoint_dir=os.path.dirname(checkpoint_path),
            checkpoint_name=os.path.basename(checkpoint_path),
            model_type=None,
            vae_instance=vae_instance, # Pass the joint tokenizer dispatcher
            model_instance=model_instance,
            guidance=0.0,
            num_steps=15,
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
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("base_color", "metallic", "roughness", "normal", "depth")
    FUNCTION = "run_inverse_pass"
    CATEGORY = "Cosmos1"

    def run_inverse_pass(self, pipeline, image, guidance=0.0, seed=42):
        pipeline.set_model_type("inverse")
        pipeline.guidance = guidance
        pipeline.seed = seed

        # === ROBUST INPUT HANDLING START ===
        print(f"[Nodes] Received input of type: {type(image)}")
        if isinstance(image, list):
            print(f"[Nodes] Input is a list. Stacking {len(image)} tensors.")
            try:
                image_5d = torch.stack(image, dim=0)
            except Exception as e:
                print(f"Warning: Could not stack tensors in list due to varying shapes: {e}. Processing first item only.")
                image_5d = image[0].unsqueeze(0)
        
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                print("[Nodes] Input is a 3D tensor (H,W,C). Adding Batch and Time dimensions.")
                image_5d = image.unsqueeze(0).unsqueeze(0)
            elif image.ndim == 4:
                print("[Nodes] Input is a 4D tensor. Assuming (B,H,W,C) and adding Time dimension.")
                image_5d = image.unsqueeze(1)
            elif image.ndim == 5:
                print("[Nodes] Input is a 5D tensor (B,T,H,W,C). Using as is.")
                image_5d = image
            else:
                raise ValueError(f"Unsupported tensor dimension: {image.ndim}. Expected 3D, 4D, or 5D.")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}. Expected torch.Tensor or list of Tensors.")

        print(f"[Nodes] Standardized input to 5D tensor with shape: {image_5d.shape}")

        # === PRE-PROCESSING FOR MODEL ===
        image_tensor = image_5d.permute(0, 4, 1, 2, 3)
        image_tensor = image_tensor * 2.0 - 1.0
        print(f"[Nodes] Pre-processed input for model with shape: {image_tensor.shape}")
        
        # === INFERENCE LOGIC (NOW BATCH-EFFICIENT) ===
        inference_passes = ["basecolor", "metallic", "roughness", "normal", "depth"]
        outputs = {}
        pbar = ProgressBar(len(inference_passes))

        for gbuffer_pass in inference_passes:
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            
            data_batch = {
                "rgb": image_tensor,
                "video": image_tensor,
                "context_index": torch.full((image_tensor.shape[0], 1), context_index, dtype=torch.long),
            }
            print(f"[Nodes] Running {gbuffer_pass} pass with context_index={context_index}")

            output_array = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(gbuffer_pass == 'normal'),
                seed=seed,
            )
            
            output_tensor = torch.from_numpy(output_array).float() / 255.0
            outputs[gbuffer_pass] = output_tensor
            pbar.update(1)

        return (outputs["basecolor"], outputs["metallic"], outputs["roughness"], outputs["normal"], outputs["depth"])


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
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "env_format": (["proj", "ball"], {"default": "proj"}),
                "env_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "env_flip_horizontal": ("BOOLEAN", {"default": False}),
                "env_rotation": ("FLOAT", {"default": 180.0, "min": 0, "max": 360, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_forward_pass"
    CATEGORY = "Cosmos1"

    def run_forward_pass(self, pipeline, depth, normal, roughness, metallic, base_color, env_map, 
                        guidance=0.0, seed=42, env_format="proj", env_brightness=1.0, 
                        env_flip_horizontal=False, env_rotation=0.0):
        
        pipeline.set_model_type("forward")
        pipeline.guidance = guidance
        pipeline.seed = seed

        gbuffer_tensors_in = {
            "depth": depth, "normal": normal, "roughness": roughness,
            "metallic": metallic, "base_color": base_color,
        }
        
        gbuffer_tensors_5d = {}
        for name, tensor_in in gbuffer_tensors_in.items():
            if isinstance(tensor_in, list):
                tensor_5d = torch.stack(tensor_in, dim=0)
            elif isinstance(tensor_in, torch.Tensor):
                if tensor_in.ndim == 3: tensor_5d = tensor_in.unsqueeze(0).unsqueeze(0)
                elif tensor_in.ndim == 4: tensor_5d = tensor_in.unsqueeze(1)
                elif tensor_in.ndim == 5: tensor_5d = tensor_in
                else: raise ValueError(f"Unsupported tensor dimension for '{name}': {tensor_in.ndim}")
            else: raise TypeError(f"Unsupported input type for '{name}': {type(tensor_in)}")
            gbuffer_tensors_5d[name] = tensor_5d
        
        B, T, H, W, C = gbuffer_tensors_5d["depth"].shape
        device = mm.get_torch_device()
        
        data_batch = {}
        key_mapping = {"base_color": "basecolor", "depth": "depth", "normal": "normal", 
                       "roughness": "roughness", "metallic": "metallic"}

        for name, tensor_5d in gbuffer_tensors_5d.items():
            processed_tensor = tensor_5d.permute(0, 4, 1, 2, 3) * 2.0 - 1.0
            data_batch[key_mapping[name]] = processed_tensor
        
        data_batch['video'] = data_batch['depth']

        envlight_dict = None
        if env_format == 'proj':
            print("[Nodes] Processing env_map as panoramic projection ('proj' mode).")
            envlight_dict = render_projection_from_panorama(
                env_input=env_map, resolution=(H, W), num_frames=T, device=device,
                env_brightness=env_brightness, env_flip=env_flip_horizontal, env_rot=env_rotation
            )
        elif env_format == 'ball':
            print("[Nodes] Processing env_map as a direct tonemap of a pre-rendered ball ('ball' mode).")
            if H != W:
                logging.warning(f"Ball mode expects a square input, but G-buffers are {W}x{H}. Results may be distorted.")
            envlight_dict = tonemap_image_direct(
                env_input=env_map, resolution=(H, W), num_frames=T, device=device
            )

        env_ldr = envlight_dict['env_ldr'].permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
        env_log = envlight_dict['env_log'].permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
        env_nrm = latlong_vec(resolution=(H, W), device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        data_batch['env_ldr'] = env_ldr.expand(B, -1, -1, -1, -1)
        data_batch['env_log'] = env_log.expand(B, -1, -1, -1, -1)
        data_batch['env_nrm'] = env_nrm.expand(B, -1, T, -1, -1)

        print("[Nodes] Data batch prepared. Calling diffusion pipeline...")
        output_array = pipeline.generate_video(data_batch=data_batch, seed=seed)
        final_output = torch.from_numpy(output_array).float() / 255.0

        return (final_output,)

class LoadHDRImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"tooltip": "Path to HDR image (.hdr, .exr)"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_hdr"
    CATEGORY = "Cosmos1"

    def load_hdr(self, path):
        img = imageio.imread(path, format='HDR-FI')
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        tensor = torch.from_numpy(img).float().unsqueeze(0)
        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "LoadDiffusionRendererModel": LoadDiffusionRendererModel,
    "Cosmos1InverseRenderer": Cosmos1InverseRenderer,
    "Cosmos1ForwardRenderer": Cosmos1ForwardRenderer,
    "LoadHDRImage": LoadHDRImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiffusionRendererModel": "Load Diffusion Renderer Model",
    "Cosmos1InverseRenderer": "Cosmos1 Inverse Renderer",
    "Cosmos1ForwardRenderer": "Cosmos1 Forward Renderer",
    "LoadHDRImage": "Load HDR Image",
}