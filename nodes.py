"""
ComfyUI Cosmos1 Diffusion Renderer Nodes

This nodepack provides ComfyUI nodes for the Cosmos1 diffusion renderer with the following features:

Core Nodes:
- LoadDiffusionRendererModel: Load inverse/forward diffusion renderer models
- Cosmos1InverseRenderer: RGB -> G-buffer maps (basecolor, metallic, roughness, normal, depth)  
- Cosmos1ForwardRenderer: G-buffer maps + environment -> rendered RGB

Data Range Compatibility:
- ComfyUI IMAGE tensors: [0, 1] range (standard)
- Cosmos1 models expect: [-1, 1] range (official implementation)
- Automatic conversion: ComfyUI [0,1] -> Cosmos1 [-1,1] in forward renderer
- Inverse renderer outputs: [0, 1] range (ComfyUI standard)
- Forward renderer outputs: [0, 1] range (converted from model's [-1, 1])

Enhanced Environment Map Processing:
- Official-quality HDR pipeline when preprocess_envmap.py is available
- Support for multiple formats: proj (equirectangular), ball (chrome ball), fixed
- Advanced controls: strength, flipping, rotation
- Intelligent caching system for performance
- Robust fallback to legacy processing when advanced system unavailable

Utility Nodes:
- PreprocessEnvironmentMap: Standalone environment map preprocessing
- EnvironmentMapCacheManager: Cache statistics and management

Dependencies:
- Requires diffusion_renderer_pipeline.py for core functionality
- Optional: preprocess_envmap.py for advanced environment map processing
- Falls back gracefully when advanced features are unavailable
"""

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
VAE_CONFIG_PATH = os.path.join(script_directory, "VAE_config.json")  # Placeholder, as requested

# Import our clean VAE implementation
from .CleanVAE import CleanVAE
from diffusers import AutoencoderKLCosmos

# Use our clean pipeline instead of the original kludgy one
from .diffusion_renderer_pipeline import CleanDiffusionRendererPipeline
from .model_diffusion_renderer import CleanDiffusionRendererModel
from .diffusion_renderer_config import get_inverse_renderer_config

# Import new environment map processing system
try:
    from preprocess_envmap import (
        process_environment_map_robust,
        latlong_vec,
        clear_environment_cache,
        get_cache_stats
    )
    ENVMAP_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced environment map processing not available: {e}")
    ENVMAP_SYSTEM_AVAILABLE = False

# Extracted utilities from the original codebase without dependencies
# Official mapping from cosmos_predict1/diffusion/inference/diffusion_renderer_utils/rendering_utils.py
GBUFFER_INDEX_MAPPING = {
    "basecolor": 0,
    "metallic": 1,
    "roughness": 2,
    "normal": 3,
    "depth": 4,
}

def envmap_vec_fallback(resolution, device):
    """Environment map direction vectors using official algorithm (fallback when preprocess_envmap is not available)"""
    H, W = resolution
    
    # Exact replication of official latlong_vec algorithm
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device), 
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device),
        indexing='ij'
    )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    dir_vec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
    ), dim=-1)
    
    # Apply official envmap_vec transformation: -latlong_vec().flip(0).flip(1)
    return -dir_vec.flip(0).flip(1)

def official_hdr_processing(env_tensor, log_scale=10000):
    """Implement official HDR processing pipeline without dependencies"""
    
    # Official NaN/Inf cleanup
    env_tensor = torch.nan_to_num(env_tensor, nan=0.0, posinf=65504.0, neginf=0.0)
    env_tensor = env_tensor.clamp(0.0, 65504.0)
    
    # Official Reinhard tone mapping (exact algorithm)
    def reinhard_official(x, max_point=16):
        return x * (1 + x / (max_point ** 2)) / (1 + x)
    
    # Official RGB to sRGB conversion
    def rgb2srgb_official(rgb):
        return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)
    
    # Apply official tone mapping
    env_ldr = rgb2srgb_official(reinhard_official(env_tensor, max_point=16).clamp(0, 1))
    env_log = rgb2srgb_official(torch.log1p(env_tensor) / np.log1p(log_scale)).clamp(0, 1)
    
    return env_ldr, env_log

def process_environment_map_fallback(env_map_path, resolution, num_frames, fixed_pose, rotate_envlight, env_format, device):
    """Enhanced environment map processing using official algorithms (fallback when preprocess_envmap is not available)"""
    import imageio
    
    H, W = resolution
    
    # Load the HDR environment map
    env_map = imageio.imread(env_map_path)
    
    # Convert to tensor
    env_tensor = torch.from_numpy(env_map).float().to(device)
    if env_tensor.ndim == 2:
        env_tensor = env_tensor.unsqueeze(-1).repeat(1, 1, 3)
    
    # Resize to target resolution if needed
    if env_tensor.shape[:2] != (H, W):
        env_tensor = torch.nn.functional.interpolate(
            env_tensor.permute(2, 0, 1).unsqueeze(0), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    
    # Use official HDR processing instead of simple normalization
    env_ldr, env_log = official_hdr_processing(env_tensor, log_scale=10000)
    
    # Expand for video frames if needed
    if num_frames > 1:
        env_ldr = env_ldr.unsqueeze(0).expand(num_frames, -1, -1, -1)
        env_log = env_log.unsqueeze(0).expand(num_frames, -1, -1, -1)
    else:
        env_ldr = env_ldr.unsqueeze(0)
        env_log = env_log.unsqueeze(0)
    
    return {
        'env_ldr': env_ldr,
        'env_log': env_log,
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
                "vae_model": (folder_paths.get_filename_list("vae"), {"default": "None", "tooltip": "VAE model for encoding/decoding. Loaded from 'ComfyUI/models/vae'. If None, uses mock VAE."}),
            }
        }

    RETURN_TYPES = ("DIFFUSION_RENDERER_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Cosmos1"

    def load_pipeline(self, model, vae_model="None"):
        device = mm.get_torch_device()
        dtype = torch.bfloat16
        print(f"Targeting device: {device}, dtype: {dtype}")

        # --- VAE LOADING (Robust Offline Method) ---
        vae_instance = None
        if vae_model and vae_model != "None":
            vae_path = folder_paths.get_full_path("vae", vae_model)
            print(f"Loading VAE from: {vae_path}")

            if not os.path.exists(VAE_CONFIG_PATH):
                raise FileNotFoundError(f"VAE config.json not found at {VAE_CONFIG_PATH}. Please create a 'vae_config' subfolder and place the config.json from the stabilityai/cosmo-vae Hugging Face repository inside it.")

            print(f"Loading VAE config from local file: {VAE_CONFIG_PATH}")
            with open(VAE_CONFIG_PATH, 'r') as f:
                vae_config_data = json.load(f)

            vae_model_diffusers = AutoencoderKLCosmos.from_config(vae_config_data)
            sd = comfy.utils.load_torch_file(vae_path)
            vae_model_diffusers.load_state_dict(sd)
            
            vae_instance = CleanVAE(vae_model_diffusers)
            vae_instance.to(device)
            vae_instance.model.to(dtype)
            print("VAE loaded successfully.")
        else:
            print("Using mock VAE (no VAE model specified).")

        # --- MEMORY-EFFICIENT MODEL LOADING using META DEVICE ---
        checkpoint_path = folder_paths.get_full_path("diffusion_models", model)
        
        # 1. Load the state_dict to CPU RAM. This is safe.
        print(f"Loading checkpoint weights to CPU from: {checkpoint_path}")
        state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)

        # Handle potential nesting and key prefixes
        if "model" in state_dict: state_dict = state_dict["model"]

        # FINAL FIX: The checkpoint keys are ALL prefixed with `net.`. We must remove it.
        if any(key.startswith('net.') for key in state_dict.keys()):
            print("Stripping 'net.' prefix from state_dict keys...")
            state_dict = {k.replace('net.', '', 1): v for k, v in state_dict.items()}

        print("Instantiating model skeleton on 'meta' device...")
        basic_config = get_inverse_renderer_config()
        with torch.device("meta"):
             model_instance = CleanDiffusionRendererModel(basic_config)

        # 3. Materialize the model on the GPU with the correct dtype. This allocates
        #    the ~28GB of VRAM for the empty bfloat16 tensors.
        print(f"Materializing model on {device} (step 1/2)...")
        model_instance.to_empty(device=device)

        print(f"Casting materialized model to {dtype} (step 2/2)...")
        model_instance.to(dtype=dtype)

        # 4. Load the CPU state_dict directly into the materialized GPU model.
        #    This is an in-place operation and avoids any intermediate copies.
        print("Loading weights into materialized GPU model...")
        model_instance.net.load_state_dict(state_dict, strict=True)
        
        # Clean up the large state_dict from CPU memory
        del state_dict
        mm.soft_empty_cache()

        model_instance.eval()
        print(f"✅ Pre-loaded diffusion model successfully to {device}")

        pipeline = CleanDiffusionRendererPipeline(
            checkpoint_dir=os.path.dirname(checkpoint_path),
            checkpoint_name=os.path.basename(checkpoint_path),
            model_type=None,
            vae_instance=vae_instance,
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
        # Set model type based on node being used (inverse renderer)
        pipeline.set_model_type("inverse")
        
        # Configure pipeline with runtime parameters
        pipeline.guidance = guidance
        pipeline.seed = seed

        # Standardize image input to a 5D tensor (B, T, H, W, C)
        print(f"[Nodes] Inverse renderer input image type: {type(image)}")
        if isinstance(image, list):
            image = torch.stack(image)
            print(f"[Nodes] Stacked list to tensor: {image.shape}")
        if image.ndim == 4:
            image = image.unsqueeze(0)
            print(f"[Nodes] Added batch dimension: {image.shape}")

        B, T, H, W, C = image.shape
        print(f"[Nodes] Final input shape: {image.shape} (B={B}, T={T}, H={H}, W={W}, C={C})")
        pipeline.num_video_frames = T
        pipeline.height = H
        pipeline.width = W

        pbar = ProgressBar(B)
        # Use official order: basecolor, metallic, roughness, normal, depth (indices 0, 1, 2, 3, 4)
        inference_passes = ["basecolor", "metallic", "roughness", "normal", "depth"]
        batch_outputs = {pass_name: [] for pass_name in inference_passes}

        for i in range(B):
            image_slice = image[i] # Shape: (T, H, W, C)
            print(f"[Nodes] Processing batch {i}, image_slice shape: {image_slice.shape}")
            
            # Pipeline expects image shape (1, T, C, H, W)
            image_tensor = image_slice.permute(0, 3, 1, 2).unsqueeze(0)
            print(f"[Nodes] Prepared image_tensor for pipeline: {image_tensor.shape} (expected: 1,T,C,H,W)")

            data_batch = {
                "image": image_tensor,
                "context_index": torch.zeros(1, 1, dtype=torch.long),  # Controls which G-buffer to generate
            }

            for gbuffer_pass in inference_passes:
                # Set context_index to specify which G-buffer pass to generate
                # This is required for inverse renderer (matches official implementation)
                context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
                data_batch["context_index"].fill_(context_index)
                print(f"[Nodes] Running {gbuffer_pass} pass with context_index={context_index}")

                output_tensor = pipeline.generate_video(
                    data_batch=data_batch,
                    normalize_normal=(gbuffer_pass == 'normal'),
                    seed=seed + i, # Use a different seed for each image in the batch
                )
                print(f"[Nodes] {gbuffer_pass} output shape: {output_tensor.shape}")
                batch_outputs[gbuffer_pass].append(output_tensor)
            pbar.update(1)

        # Stack results from all videos in the batch
        outputs = {pass_name: torch.cat(batch_outputs[pass_name], dim=0) for pass_name in inference_passes}
        print(f"[Nodes] Final outputs shapes: {[(k, v.shape) for k, v in outputs.items()]}")
        
        # Return in official order: basecolor, metallic, roughness, normal, depth
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
                # Enhanced environment map controls (only available with new system)
                "env_format": (["proj", "ball", "fixed"], {"default": "proj", "tooltip": "proj: equirectangular, ball: chrome ball, fixed: fixed format"}),
                "env_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Environment map brightness multiplier"}),
                "env_flip_horizontal": ("BOOLEAN", {"default": False, "tooltip": "Flip environment map horizontally"}),
                "env_flip_vertical": ("BOOLEAN", {"default": False, "tooltip": "Flip environment map vertically"}),
                "env_rotation": ("FLOAT", {"default": 180.0, "min": 0, "max": 360, "step": 1.0, "tooltip": "Rotate environment map (degrees)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_forward_pass"
    CATEGORY = "Cosmos1"

    def run_forward_pass(self, pipeline, depth, normal, roughness, metallic, base_color, env_map, 
                        guidance=0.0, seed=42, env_format="ball", env_brightness=1.0, 
                        env_flip_horizontal=False, env_flip_vertical=False, env_rotation=0.0):
        # Set model type based on node being used (forward renderer)
        pipeline.set_model_type("forward")
        
        # Standardize all inputs to 5D tensors (B, T, H, W, C)
        gbuffer_tensors = {
            "depth": depth,
            "normal": normal,
            "roughness": roughness,
            "metallic": metallic,
            "base_color": base_color,
            "env_map": env_map,
        }

        print(f"[Nodes] Forward renderer input tensor types and shapes:")
        for name, tensor in gbuffer_tensors.items():
            print(f"[Nodes]   {name}: {type(tensor)}, shape: {tensor.shape if hasattr(tensor, 'shape') else 'no shape'}")

        # Standardize tensors and handle lists
        for name, tensor in gbuffer_tensors.items():
            original_shape = tensor.shape if hasattr(tensor, 'shape') else 'unknown'
            if isinstance(tensor, list):
                tensor = torch.stack(tensor)
                print(f"[Nodes] Stacked {name} list: {original_shape} -> {tensor.shape}")
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0) # Add batch dimension
                print(f"[Nodes] Added batch dim to {name}: {tensor.shape}")
            gbuffer_tensors[name] = tensor

        # Get dimensions from the primary input
        B, T, H, W, C = gbuffer_tensors["depth"].shape
        print(f"[Nodes] Primary dimensions from depth: B={B}, T={T}, H={H}, W={W}, C={C}")

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
            print(f"[Nodes] Processing forward batch {i}/{B}")
            # Prepare G-buffer inputs for the current video
            # Note: Forward renderer does NOT use context_index - it processes all G-buffers simultaneously
            # This matches the official implementation where all maps are fed together
            data_batch = {}
            # Map ComfyUI tensor names to official Cosmos1 keys
            key_mapping = {
                "base_color": "basecolor",  # ComfyUI uses "base_color", official uses "basecolor"
                "depth": "depth",
                "normal": "normal", 
                "roughness": "roughness",
                "metallic": "metallic"
            }
            
            for name in ["depth", "normal", "roughness", "metallic", "base_color"]:
                tensor_slice = gbuffer_tensors[name][i] # (T, H, W, C)
                print(f"[Nodes]   {name} slice shape: {tensor_slice.shape}")
                # Pipeline expects (1, T, C, H, W) and range [-1, 1]
                # ComfyUI provides data in [0, 1], official expects [-1, 1]
                processed_tensor = tensor_slice.permute(0, 3, 1, 2).unsqueeze(0)
                print(f"[Nodes]   {name} processed shape: {processed_tensor.shape} (expected: 1,T,C,H,W)")
                official_key = key_mapping[name]
                data_batch[official_key] = processed_tensor * 2.0 - 1.0

            # Process the environment map for the current video
            env_map_slice = gbuffer_tensors["env_map"][i] # (T, H, W, C)
            print(f"[Nodes] env_map slice shape: {env_map_slice.shape}")
            
            
            # Use the new robust environment map processing system if available
            if ENVMAP_SYSTEM_AVAILABLE:
                try:
                    # Convert ComfyUI IMAGE tensor to the format expected by the preprocessing system
                    env_map_np = env_map_slice.cpu().numpy()
                    # Take first frame for processing (environment maps are typically static)
                    if env_map_np.shape[0] >= 1:
                        env_map_np = env_map_np[0]  # Always take first frame for environment processing
                    
                    envlight_dict = process_environment_map_robust(
                        env_input=env_map_np,  # Pass numpy array directly
                        resolution=(H, W),
                        num_frames=T,
                        env_format=env_format,
                        env_brightness=env_brightness,
                        env_flip_horizontal=env_flip_horizontal,
                        env_flip_vertical=env_flip_vertical,
                        env_rotation=env_rotation,
                        device=device,
                    )
                    
                    env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                    env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                    env_nrm = latlong_vec(resolution=(H, W), device=device)
                    env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
                    
                    data_batch['env_ldr'] = env_ldr
                    data_batch['env_log'] = env_log
                    data_batch['env_nrm'] = env_nrm
                    
                except Exception as e:
                    logging.warning(f"Advanced environment processing failed, falling back to enhanced method: {e}")
                    # Enhanced fallback processing with official algorithms
                    envlight_dict = self._process_env_fallback(env_map_slice, H, W, T, device, i)
                    env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                    env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                    
                    # Use official envmap_vec algorithm
                    env_nrm = envmap_vec_fallback([H, W], device=device)
                    env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
                    
                    data_batch['env_ldr'] = env_ldr
                    data_batch['env_log'] = env_log
                    data_batch['env_nrm'] = env_nrm
            else:
                # Enhanced processing with official algorithms when new system is not available
                envlight_dict = self._process_env_fallback(env_map_slice, H, W, T, device, i)
                env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                
                # Use official envmap_vec algorithm
                env_nrm = envmap_vec_fallback([H, W], device=device)
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

    def _process_env_fallback(self, env_map_slice, H, W, T, device, batch_idx):
        """Enhanced environment map processing for fallback using official algorithms"""
        import imageio
        temp_env_map_path = os.path.join(folder_paths.get_temp_directory(), f"temp_env_map_{batch_idx}.hdr")
        
        env_map_np = env_map_slice.cpu().numpy()
        # Take first frame for processing (environment maps are typically static)
        if env_map_np.shape[0] >= 1:
            env_map_np = env_map_np[0]  # Always take first frame for environment processing

        imageio.imwrite(temp_env_map_path, env_map_np)

        envlight_dict = process_environment_map_fallback(
            temp_env_map_path,
            resolution=(H, W),
            num_frames=T,
            fixed_pose=True,
            rotate_envlight=False,
            env_format=['proj'],
            device=device,
        )
        
        # Clean up temporary file
        try:
            os.remove(temp_env_map_path)
        except:
            pass
            
        return envlight_dict
    
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
        import imageio
        import torch
        import numpy as np

        # Load HDR image as float32 numpy array
        img = imageio.imread(path, format='HDR-FI')  # For .hdr files
        # If .exr, use format='EXR-FI' or OpenEXR for more control

        # Ensure shape is (H, W, C)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        # Convert to torch tensor, shape (1, H, W, C), dtype float32
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
