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

from diffusers import AutoencoderKLCosmos
from .CleanVAE import CleanVAE
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

    def load_pipeline(self, model):
        device = mm.get_torch_device()
        dtype = torch.bfloat16
        print(f"Targeting device: {device}, dtype: {dtype}")

        # --- VAE LOADING (Simplified and Corrected) ---
        vae_main_dir = os.path.join(folder_paths.models_dir, "vae", "Cosmos-1.0-Tokenizer-CV8x8x8")
        if not vae_main_dir:
             raise FileNotFoundError("Could not find 'Cosmos-1.0-Tokenizer-CV8x8x8' in any VAE model directory.")
        
        image_vae_subfolder_path = os.path.join(vae_main_dir, "vae")
        if not os.path.isdir(image_vae_subfolder_path):
            raise FileNotFoundError(f"Image VAE subfolder not found at: {image_vae_subfolder_path}")

        # We load one VAE and wrap it. This VAE handles both images (T=1) and videos.
        vae_instance = CleanVAE(model_path=image_vae_subfolder_path)
        vae_instance.to(device)
        vae_instance.reset_dtype(dtype)
        print("✅ Universal VAE loaded successfully via from_pretrained.")

        # --- MEMORY-EFFICIENT MODEL LOADING ---
        checkpoint_path = folder_paths.get_full_path("diffusion_models", model)
        
        print(f"Loading checkpoint weights to CPU from: {checkpoint_path}")
        state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        print("\n🔍 Context Embedding Debug:")
    
        # Look for context embedding in checkpoint
        context_emb_key = None
        for key in state_dict.keys():
            if 'context_embedding' in key:
                context_emb_key = key
                print(f"  Found in checkpoint: {key} shape={state_dict[key].shape}")
                break

        print("Instantiating model skeleton on 'meta' device...")
        basic_config = get_inverse_renderer_config()
        with torch.device("meta"):
             model_instance = CleanDiffusionRendererModel(basic_config)

        model_instance.to_empty(device=device)
        model_instance.to(dtype=dtype)
        model_instance.load_state_dict(state_dict, strict=True)

        print("\n🔍 Checkpoint key analysis:")
        sample_keys = list(state_dict.keys())[:20]
        for key in sample_keys:
            print(f"  {key}: shape={state_dict[key].shape}")

        if hasattr(model_instance, 'net') and hasattr(model_instance.net, 'context_embedding'):
            ctx_emb = model_instance.net.context_embedding.weight
            print(f"  Loaded context embedding: mean={ctx_emb.mean():.6f}, std={ctx_emb.std():.6f}")
            print(f"  Shape: {ctx_emb.shape}")

        # Check for net. prefix
        has_net_prefix = any(k.startswith('net.') for k in state_dict.keys())
        print(f"\nCheckpoint has 'net.' prefix: {has_net_prefix}")

        def check_if_weights_loaded(model):
            """Check if model has non-zero weights (not just initialized)"""
            total_params = 0
            zero_params = 0
    
            for name, param in model.named_parameters():
                total_params += param.numel()
                if torch.allclose(param, torch.zeros_like(param), atol=1e-8):
                    zero_params += param.numel()
    
            zero_percent = (zero_params / total_params) * 100
            print(f"\n📊 Weight Analysis:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Zero parameters: {zero_params:,} ({zero_percent:.2f}%)")
    
            if zero_percent > 50:
                print("  ❌ WARNING: Most weights are zero! Checkpoint may not be loaded properly!")
            else:
                print("  ✅ Weights appear to be loaded")
    
            # Check specific critical weights
            if hasattr(model, 'net'):
                if hasattr(model.net, 'context_embedding'):
                    ctx_emb = model.net.context_embedding.weight
                    print(f"  Context embedding: mean={ctx_emb.mean():.6f}, std={ctx_emb.std():.6f}")

        check_if_weights_loaded(model_instance)

        ca_block = model_instance.net.blocks['block0'].blocks[1]  # First CA block
        if hasattr(ca_block.block, 'attn'):
            k_weight = ca_block.block.attn.to_k[0].weight
            v_weight = ca_block.block.attn.to_v[0].weight
    
            print(f"\n🔍 Cross-Attention K/V Projection Analysis:")
            print(f"  K weight shape: {k_weight.shape}")  # Should be [4096, 1024]
            print(f"  K weight stats: mean={k_weight.mean():.6f}, std={k_weight.std():.6f}")
            print(f"  K weight norm: {k_weight.norm():.4f}")
    
            # Check if weights are near zero
            near_zero_k = (k_weight.abs() < 1e-6).sum().item()
            print(f"  K near-zero weights: {near_zero_k}/{k_weight.numel()} ({100*near_zero_k/k_weight.numel():.2f}%)")
    
            print(f"  V weight stats: mean={v_weight.mean():.6f}, std={v_weight.std():.6f}")
            print(f"  V weight norm: {v_weight.norm():.4f}")
        
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
            dtype=dtype,
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
                "context_index": torch.full((image_tensor.shape[0], 1), context_index, dtype=torch.long, device=image_tensor.device),
            }
            print(f"[Nodes] Running {gbuffer_pass} pass with context_index={context_index}")

            output_array = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(gbuffer_pass == 'normal'),
                seed=seed,
            )
            
            output_tensor = torch.from_numpy(output_array).float() / 255.0

            b, t, h, w, c = output_tensor.shape
            output_tensor_4d = output_tensor.reshape(b * t, h, w, c)

            outputs[gbuffer_pass] = output_tensor_4d
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
    
class VAEPassthroughTest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae_path": ("STRING", {"default": "Cosmos-1.0-Tokenizer-CV8x8x8/vae"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("output", "diff_image", "stats")
    FUNCTION = "test_vae"
    CATEGORY = "Cosmos1/Debug"

    def test_vae(self, image, vae_path):
        import torch
        import numpy as np
        import os
        import folder_paths
        from .CleanVAE import CleanVAE
        
        # Initialize VAE
        vae_full_path = os.path.join(folder_paths.models_dir, "vae", vae_path)
        vae = CleanVAE(model_path=vae_full_path)
        vae.to(torch.device('cuda'))
        
        # Use float32 for best quality in testing
        vae.reset_dtype(torch.float32)
        
        # Prepare input - convert to 5D tensor in [-1, 1]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        
        # Handle dimensions
        if image.ndim == 3:  # (H, W, C)
            image = image.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W, C)
        elif image.ndim == 4:  # (B, H, W, C)
            image = image.unsqueeze(1)  # -> (B, 1, H, W, C)
        elif image.ndim == 5:  # Already (B, T, H, W, C)
            pass
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
        
        B, T, H, W, C = image.shape
        
        # Convert to model format: (B, C, T, H, W) in range [-1, 1]
        image_tensor = image.permute(0, 4, 1, 2, 3)
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Move to GPU with float32 for best precision
        image_tensor = image_tensor.to(device='cuda', dtype=torch.float32)
        
        print(f"\n{'='*60}")
        print(f"VAE PASSTHROUGH TEST (float32 precision)")
        print(f"{'='*60}")
        print(f"Input shape: {image_tensor.shape}")
        print(f"Input range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        print(f"Input mean: {image_tensor.mean():.3f}, std: {image_tensor.std():.3f}")
        
        # Encode
        with torch.no_grad():
            latent = vae.encode(image_tensor)
            print(f"\nLatent shape: {latent.shape}")
            print(f"Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"Latent mean: {latent.mean():.3f}, std: {latent.std():.3f}")
            
            # Decode
            reconstructed = vae.decode(latent)
            print(f"\nReconstructed shape: {reconstructed.shape}")
            print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            print(f"Reconstructed mean: {reconstructed.mean():.3f}, std: {reconstructed.std():.3f}")
        
        # Calculate difference
        diff = (reconstructed - image_tensor).abs()
        mse = ((reconstructed - image_tensor) ** 2).mean()
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # PSNR for [-1,1] range
        
        print(f"\nReconstruction Metrics:")
        print(f"  L1 Error: {diff.mean():.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Max Error: {diff.max():.6f}")
        
        # Try different output normalizations
        print(f"\n{'='*40}")
        print("Testing different output normalizations:")
        print(f"{'='*40}")
        
        # Method 1: Standard [-1,1] to [0,1]
        output1 = (reconstructed + 1.0) / 2.0
        print(f"Method 1 (standard): [{output1.min():.3f}, {output1.max():.3f}]")
        
        # Method 2: Clamp first then normalize (for outputs slightly outside [-1,1])
        output2 = (reconstructed.clamp(-1, 1) + 1.0) / 2.0
        print(f"Method 2 (clamp then normalize): [{output2.min():.3f}, {output2.max():.3f}]")
        
        # Method 3: Min-max normalization
        if reconstructed.max() > reconstructed.min():
            output3 = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
            print(f"Method 3 (min-max): [{output3.min():.3f}, {output3.max():.3f}]")
        else:
            output3 = output2  # Fallback if constant
        
        # Use method 2 for output (clamp then normalize - safest)
        output_tensor = output2.clamp(0, 1)
        
        # Convert back to ComfyUI format (B, T, H, W, C)
        output_tensor = output_tensor.permute(0, 2, 3, 4, 1)
        output_tensor = output_tensor.reshape(B * T, H, W, C)
        
        # Create difference visualization (amplify for visibility)
        diff_vis = diff * 5.0  # Amplify difference for visualization
        diff_vis = diff_vis.clamp(0, 1)
        diff_tensor = diff_vis.permute(0, 2, 3, 4, 1)
        diff_tensor = diff_tensor.reshape(B * T, H, W, C)
        
        # Create stats string
        stats = f"""VAE Test Results (float32):
Input: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]
Latent: [{latent.min():.3f}, {latent.max():.3f}]
Output: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]

Quality Metrics:
  L1 Error: {diff.mean():.6f}
  MSE: {mse:.6f}
  PSNR: {psnr:.2f} dB
  Max Error: {diff.max():.6f}

Statistics:
  Output Mean: {reconstructed.mean():.3f} (ideal: ~0)
  Output Std: {reconstructed.std():.3f} (ideal: ~0.5)
"""
        
        print(f"\n{stats}")
        print(f"{'='*60}\n")
        
        return (output_tensor.cpu(), diff_tensor.cpu(), stats)


NODE_CLASS_MAPPINGS = {
    "LoadDiffusionRendererModel": LoadDiffusionRendererModel,
    "Cosmos1InverseRenderer": Cosmos1InverseRenderer,
    "Cosmos1ForwardRenderer": Cosmos1ForwardRenderer,
    "LoadHDRImage": LoadHDRImage,
    "VAEPassthroughTest" : VAEPassthroughTest
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiffusionRendererModel": "Load Diffusion Renderer Model",
    "Cosmos1InverseRenderer": "Cosmos1 Inverse Renderer",
    "Cosmos1ForwardRenderer": "Cosmos1 Forward Renderer",
    "LoadHDRImage": "Load HDR Image",
    "VAEPassthroughTest": "VAE Passthrough Test"
}