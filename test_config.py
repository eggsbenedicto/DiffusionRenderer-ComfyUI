#!/usr/bin/env python3
"""
Test script for the clean diffusion renderer configuration system.
This validates that our config system works correctly.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_renderer_config import (
    get_inverse_renderer_config,
    get_forward_renderer_config,
    get_config_by_model_type,
    validate_config,
    get_preset_config,
    PRESET_CONFIGS
)

def test_configs():
    """Test all configuration functions"""
    print("=" * 60)
    print("Testing Diffusion Renderer Configuration System")
    print("=" * 60)
    
    # Test inverse renderer config
    print("\n1. Testing Inverse Renderer Config:")
    inverse_config = get_inverse_renderer_config(1024, 1024, 1)
    print(f"  Condition keys: {inverse_config['condition_keys']}")
    print(f"  Latent shape: {inverse_config['latent_shape']}")
    print(f"  Additional concat channels: {inverse_config['net']['additional_concat_ch']}")
    print(f"  Use context embedding: {inverse_config['net']['use_context_embedding']}")
    
    # Test forward renderer config
    print("\n2. Testing Forward Renderer Config:")
    forward_config = get_forward_renderer_config(1024, 1024, 1)
    print(f"  Condition keys: {forward_config['condition_keys']}")
    print(f"  Latent shape: {forward_config['latent_shape']}")
    print(f"  Additional concat channels: {forward_config['net']['additional_concat_ch']}")
    print(f"  Use context embedding: {forward_config['net']['use_context_embedding']}")
    
    # Test configuration validation
    print("\n3. Testing Configuration Validation:")
    try:
        validate_config(inverse_config)
        print("  ✓ Inverse config validation passed")
    except Exception as e:
        print(f"  ✗ Inverse config validation failed: {e}")
        
    try:
        validate_config(forward_config)
        print("  ✓ Forward config validation passed")
    except Exception as e:
        print(f"  ✗ Forward config validation failed: {e}")
    
    # Test config by model type
    print("\n4. Testing Config by Model Type:")
    inverse_via_type = get_config_by_model_type("inverse", 1024, 1024, 1)
    forward_via_type = get_config_by_model_type("forward", 1024, 1024, 1)
    
    print(f"  Inverse via type == direct: {inverse_via_type == inverse_config}")
    print(f"  Forward via type == direct: {forward_via_type == forward_config}")
    
    # Test preset configs
    print("\n5. Testing Preset Configurations:")
    print(f"  Available presets: {list(PRESET_CONFIGS.keys())}")
    
    for preset_name in PRESET_CONFIGS.keys():
        try:
            preset_config = get_preset_config(preset_name)
            validate_config(preset_config)
            print(f"  ✓ {preset_name}: {preset_config['latent_shape']} - {preset_config['condition_keys']}")
        except Exception as e:
            print(f"  ✗ {preset_name}: {e}")
    
    # Test network configuration details
    print("\n6. Testing Network Configuration Details:")
    net_config = inverse_config['net']
    print(f"  Model channels: {net_config['model_channels']}")
    print(f"  Num blocks: {net_config['num_blocks']}")
    print(f"  Num heads: {net_config['num_heads']}")
    print(f"  Patch spatial: {net_config['patch_spatial']}")
    print(f"  RoPE extrapolation ratios: H={net_config['rope_h_extrapolation_ratio']}, "
          f"W={net_config['rope_w_extrapolation_ratio']}, T={net_config['rope_t_extrapolation_ratio']}")
    
    # Test scheduler configuration
    print("\n7. Testing Scheduler Configuration:")
    scheduler_config = inverse_config['scheduler']
    print(f"  Sigma max: {scheduler_config['sigma_max']}")
    print(f"  Sigma min: {scheduler_config['sigma_min']}")
    print(f"  Sigma data: {scheduler_config['sigma_data']}")
    print(f"  Prediction type: {scheduler_config['prediction_type']}")
    
    print("\n" + "=" * 60)
    print("✓ All configuration tests completed successfully!")
    print("=" * 60)


def test_clean_pipeline():
    """Test that our clean pipeline can be initialized with the config system"""
    print("\n" + "=" * 60)
    print("Testing Clean Pipeline Integration")
    print("=" * 60)
    
    try:
        from diffusion_renderer_pipeline import CleanDiffusionRendererPipeline
        
        # Test inverse renderer pipeline
        print("\n1. Testing Inverse Renderer Pipeline:")
        inverse_pipeline = CleanDiffusionRendererPipeline(
            checkpoint_dir="./checkpoints",  # Non-existent dir for testing
            checkpoint_name="test_model.pt",
            model_type="inverse",
            height=1024,
            width=1024,
            num_video_frames=1
        )
        print("  ✓ Inverse pipeline initialized successfully")
        print(f"  Config condition keys: {inverse_pipeline.config['condition_keys']}")
        print(f"  Config latent shape: {inverse_pipeline.config['latent_shape']}")
        
        # Test forward renderer pipeline
        print("\n2. Testing Forward Renderer Pipeline:")
        forward_pipeline = CleanDiffusionRendererPipeline(
            checkpoint_dir="./checkpoints",  # Non-existent dir for testing
            checkpoint_name="test_model.pt",
            model_type="forward",
            height=1024,
            width=1024,
            num_video_frames=1
        )
        print("  ✓ Forward pipeline initialized successfully")
        print(f"  Config condition keys: {forward_pipeline.config['condition_keys']}")
        print(f"  Config latent shape: {forward_pipeline.config['latent_shape']}")
        
        print("\n✓ Pipeline integration tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        test_configs()
        test_clean_pipeline()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
