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

from .CleanVAE import CleanVAE
from diffusers import AutoencoderKLCosmos

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

        # --- MEMORY-EFFICIENT MODEL LOADING using META DEVICE ---
        checkpoint_path = folder_paths.get_full_path("diffusion_models", model)
        
        print(f"Loading checkpoint weights to CPU from: {checkpoint_path}")
        state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)

        # Handle potential nesting (e.g., {"model": ..., "ema": ...})
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # ------------------- CORRECTED LOADING LOGIC -------------------
        # The state_dict keys are ALREADY in the correct format (e.g., 'net.block0...', 'logvar.0...').
        # We do NOT need to strip any prefixes. We load directly into the parent model instance.

        print("Instantiating model skeleton on 'meta' device...")
        basic_config = get_inverse_renderer_config() # A basic config to build the skeleton
        with torch.device("meta"):
             # The model_instance is the PARENT module that contains both `net` and `logvar`
             model_instance = CleanDiffusionRendererModel(basic_config)

        print(f"Materializing model on {device} (step 1/2)...")
        model_instance.to_empty(device=device)

        print(f"Casting materialized model to {dtype} (step 2/2)...")
        model_instance.to(dtype=dtype)

        print("Loading weights into the full materialized GPU model...")
        # Load into the PARENT model instance. It knows about both `net` and `logvar`.
        model_instance.load_state_dict(state_dict, strict=True)
        # ----------------------------------------------------------------

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
        pipeline.set_model_type("inverse")
        pipeline.guidance = guidance
        pipeline.seed = seed

        # === ROBUST INPUT HANDLING START ===
        
        print(f"[Nodes] Received input of type: {type(image)}")
        
        # 1. Handle list of tensors (common for batched inputs from other nodes)
        if isinstance(image, list):
            # Each item in the list is typically (T, H, W, C)
            # Stacking on a new dimension 0 creates (B, T, H, W, C)
            print(f"[Nodes] Input is a list. Stacking {len(image)} tensors.")
            try:
                image_5d = torch.stack(image, dim=0)
            except Exception as e:
                # Fallback for lists of varying shapes, process the first one.
                print(f"Warning: Could not stack tensors in list due to varying shapes: {e}. Processing first item only.")
                image_5d = image[0].unsqueeze(0) # Add a batch dim
        
        # 2. Handle single tensor input
        elif isinstance(image, torch.Tensor):
            # Check dimensions and add batch/temporal dims if missing
            if image.ndim == 3: # Shape (H, W, C)
                print("[Nodes] Input is a 3D tensor (H,W,C). Adding Batch and Time dimensions.")
                image_5d = image.unsqueeze(0).unsqueeze(0) # -> (1, 1, H, W, C)
            elif image.ndim == 4: # Shape (B, H, W, C) or (T, H, W, C) - ComfyUI standard is ambiguous
                # We'll assume it's (B, H, W, C) and add a temporal dimension of 1
                print("[Nodes] Input is a 4D tensor. Assuming (B,H,W,C) and adding Time dimension.")
                image_5d = image.unsqueeze(1) # -> (B, 1, H, W, C)
            elif image.ndim == 5: # Shape (B, T, H, W, C)
                print("[Nodes] Input is a 5D tensor (B,T,H,W,C). Using as is.")
                image_5d = image
            else:
                raise ValueError(f"Unsupported tensor dimension: {image.ndim}. Expected 3D, 4D, or 5D.")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}. Expected torch.Tensor or list of Tensors.")

        print(f"[Nodes] Standardized input to 5D tensor with shape: {image_5d.shape}")

        # === PRE-PROCESSING FOR MODEL ===
        
        # 3. Permute from (B, T, H, W, C) to model's expected (B, C, T, H, W)
        image_tensor = image_5d.permute(0, 4, 1, 2, 3)
        
        # 4. Normalize range from [0, 1] to [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0
        
        print(f"[Nodes] Pre-processed input for model with shape: {image_tensor.shape}")
        
        # === INFERENCE LOGIC (NOW BATCH-EFFICIENT) ===

        # Use official order
        inference_passes = ["basecolor", "metallic", "roughness", "normal", "depth"]
        outputs = {}
        pbar = ProgressBar(len(inference_passes))

        for gbuffer_pass in inference_passes:
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            
            # Create data_batch for the entire batch
            data_batch = {
                "rgb": image_tensor,
                "video": image_tensor, # For shape inference in the pipeline
                "context_index": torch.full((image_tensor.shape[0], 1), context_index, dtype=torch.long),
            }
            print(f"[Nodes] Running {gbuffer_pass} pass with context_index={context_index}")

            # Single pipeline call for the whole batch
            output_array = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(gbuffer_pass == 'normal'),
                seed=seed,
            )
            
            # Post-process numpy array (B, T, H, W, C) back to ComfyUI IMAGE tensor
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
                
                # --- Refactored Environment Map Controls ---
                "env_format": (["proj", "ball"], {
                    "default": "proj", 
                    "tooltip": "'proj': Input is a panoramic HDR. 'ball': Input is a square, pre-rendered chrome ball HDR."
                }),
                "env_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, 
                    "tooltip": "Adjust brightness of the environment map ('proj' mode only)."
                }),
                "env_flip_horizontal": ("BOOLEAN", {"default": False, 
                    "tooltip": "Flip the panoramic environment map horizontally ('proj' mode only)."
                }),
                "env_rotation": ("FLOAT", {"default": 180.0, "min": 0, "max": 360, "step": 1.0, 
                    "tooltip": "Rotate the panoramic environment map ('proj' mode only)."
                }),
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

        # === 1. G-BUFFER INPUT HANDLING ===
        # Standardize all G-buffer inputs into 5D tensors: (B, T, H, W, C)
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
        
        # === 2. G-BUFFER PRE-PROCESSING ===
        # Get unified dimensions and device from a primary input
        B, T, H, W, C = gbuffer_tensors_5d["depth"].shape
        device = mm.get_torch_device()
        
        # Create the data_batch for the model and populate it with G-buffers
        data_batch = {}
        key_mapping = {"base_color": "basecolor", "depth": "depth", "normal": "normal", 
                       "roughness": "roughness", "metallic": "metallic"}

        for name, tensor_5d in gbuffer_tensors_5d.items():
            # Permute from (B,T,H,W,C) to model's (B,C,T,H,W) and normalize [0,1] to [-1,1]
            processed_tensor = tensor_5d.permute(0, 4, 1, 2, 3) * 2.0 - 1.0
            data_batch[key_mapping[name]] = processed_tensor
        
        # Use one G-buffer map for shape inference inside the pipeline
        data_batch['video'] = data_batch['depth']

        # === 3. ENVIRONMENT MAP PROCESSING (Refactored Logic) ===
        # Dispatch to the correct processing function based on the chosen format.
        envlight_dict = None
        if env_format == 'proj':
            print("[Nodes] Processing env_map as panoramic projection ('proj' mode).")
            envlight_dict = render_projection_from_panorama(
                env_input=env_map,
                resolution=(H, W),
                num_frames=T,
                device=device,
                env_brightness=env_brightness,
                env_flip=env_flip_horizontal,
                env_rot=env_rotation
            )
        elif env_format == 'ball':
            print("[Nodes] Processing env_map as a direct tonemap of a pre-rendered ball ('ball' mode).")
            if H != W:
                logging.warning(f"Ball mode expects a square input, but G-buffers are {W}x{H}. Results may be distorted.")
            envlight_dict = tonemap_image_direct(
                env_input=env_map,
                resolution=(H, W),
                num_frames=T,
                device=device
            )

        # === 4. PREPARE FINAL CONDITIONING TENSORS ===
        # The output of both envmap functions is a dict with (T, H, W, C) tensors in [0,1] range.
        # We must prepare them for the model: (B, C, T, H, W) in [-1,1] range.
        
        # Reshape and normalize LDR map
        env_ldr = envlight_dict['env_ldr'].permute(3, 0, 1, 2).unsqueeze(0) # (1, C, T, H, W)
        env_ldr = env_ldr * 2.0 - 1.0

        # Reshape and normalize LOG map
        env_log = envlight_dict['env_log'].permute(3, 0, 1, 2).unsqueeze(0) # (1, C, T, H, W)
        env_log = env_log * 2.0 - 1.0

        # Generate direction vectors for the environment normal map
        env_nrm = latlong_vec(resolution=(H, W), device=device)       # -> (H, W, C)
        env_nrm = env_nrm.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # -> (1, C, 1, H, W)

        # Add to data_batch and expand to match the input batch size (B) and time (T)
        data_batch['env_ldr'] = env_ldr.expand(B, -1, -1, -1, -1)
        data_batch['env_log'] = env_log.expand(B, -1, -1, -1, -1)
        data_batch['env_nrm'] = env_nrm.expand(B, -1, T, -1, -1)

        # === 5. INFERENCE AND POST-PROCESSING ===
        print("[Nodes] Data batch prepared. Calling diffusion pipeline...")
        output_array = pipeline.generate_video(
            data_batch=data_batch,
            seed=seed,
        )
        
        # Convert final numpy array back to a ComfyUI IMAGE tensor
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
