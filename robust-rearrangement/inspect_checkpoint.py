#!/usr/bin/env python3
"""
Quick script to inspect the checkpoint structure and fix the config.
"""

import torch
import pprint
from omegaconf import OmegaConf

def inspect_checkpoint():
    policy_path = "models/rppo/one_leg/low/actor_chkpt.pt"
    
    print("Loading checkpoint...")
    checkpoint = torch.load(policy_path, map_location='cpu')
    
    print("Checkpoint keys:", list(checkpoint.keys()))
    
    if "config" in checkpoint:
        print("\nOriginal config:")
        config = checkpoint["config"]
        pprint.pprint(config)
        
        print("\nConfig keys:", list(config.keys()) if isinstance(config, dict) else "Not a dict")
        
        # Check for nested structures
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"\n{key} contains:")
                    pprint.pprint(value)
    
    # Try to determine action_dim from model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\nModel state dict keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'No shape'
            print(f"  {key}: {shape}")
        
        # Look for action-related layers
        action_keys = [k for k in state_dict.keys() if 'action' in k.lower() or 'output' in k.lower() or 'head' in k.lower()]
        print(f"\nAction-related keys:")
        for key in action_keys[:5]:  # Show first 5
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'No shape'
            print(f"  {key}: {shape}")

if __name__ == "__main__":
    inspect_checkpoint()