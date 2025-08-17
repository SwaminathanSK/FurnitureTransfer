#!/usr/bin/env python3
"""
Test script for the dense reward system in one-leg furniture assembly task.
This script runs a simple episode to verify the dense reward system is working.
"""

import os
import sys
import torch
import numpy as np

# Add the robust-rearrangement path to the Python path
sys.path.insert(0, '/home/swaminathan/git/FurnitureTransfer/robust-rearrangement')

from src.gym import get_rl_env

def test_dense_rewards():
    """Test the dense reward system with the one-leg task."""
    print("Testing dense reward system for one-leg task...")
    
    # Create environment with dense rewards
    env = get_rl_env(
        gpu_id=0,
        task='one_leg',
        num_envs=1,
        randomness='low', 
        observation_space='state',
        max_env_steps=100,
        resize_img=False,
        act_rot_repr='rot_6d',
        action_type='pos',
        april_tags=False,
        verbose=True,
        headless=True  # Run without visualization for testing
    )
    
    print(f"Environment created: {type(env)}")
    print(f"Has dense reward systems: {hasattr(env, 'dense_reward_systems')}")
    
    if hasattr(env, 'dense_reward_systems'):
        print(f"Number of dense reward systems: {len(env.dense_reward_systems)}")
    
    # Reset environment
    obs = env.reset()
    print(f"Environment reset. Observation keys: {list(obs.keys())}")
    
    # Run a few steps and check rewards
    total_reward = 0
    for step in range(20):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward.item() if hasattr(reward, 'item') else reward
        
        # Print reward info every 5 steps
        if step % 5 == 0:
            print(f"Step {step}: Reward = {reward}, Total = {total_reward:.4f}")
            
            # Check if we have reward breakdown information
            if hasattr(env, 'reward_info') and env.reward_info:
                reward_breakdown = env.reward_info.get(0, {})
                if reward_breakdown:
                    print(f"  Reward breakdown: {reward_breakdown}")
                    
            # Check current stage if available
            if (hasattr(env, 'dense_reward_systems') and 
                env.dense_reward_systems and 
                hasattr(env.dense_reward_systems[0], 'current_stage')):
                current_stage = env.dense_reward_systems[0].current_stage
                print(f"  Current stage: {current_stage}")
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"\nTest completed. Final total reward: {total_reward:.4f}")
    
    # Test reset functionality
    print("\nTesting reset functionality...")
    obs = env.reset()
    
    if (hasattr(env, 'dense_reward_systems') and 
        env.dense_reward_systems and 
        hasattr(env.dense_reward_systems[0], 'step_count')):
        step_count = env.dense_reward_systems[0].step_count
        print(f"Step count after reset: {step_count}")
    
    print("Dense reward system test completed successfully!")

def test_reward_components():
    """Test individual reward components with dummy data."""
    print("\nTesting individual reward components...")
    
    from furniture_bench.envs.dense_reward_one_leg import OneLegDenseReward
    
    # Create dense reward system
    reward_system = OneLegDenseReward(device='cpu')
    
    # Create dummy data
    gripper_pos = torch.tensor([0.0, 0.0, 0.5])  # Above table
    gripper_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])  # No rotation
    gripper_state = torch.tensor([0.04, 0.04])  # Open gripper
    
    # Dummy parts poses (7 values per part: pos + quat)
    parts_poses = torch.zeros(35)  # 5 parts * 7 values
    # Table at origin
    parts_poses[0:3] = torch.tensor([0.0, 0.0, 0.3])  # table position
    parts_poses[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # table orientation
    # Leg4 at some offset
    parts_poses[28:31] = torch.tensor([0.1, 0.1, 0.35])  # leg position
    parts_poses[31:35] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # leg orientation
    
    # Test reward computation
    total_reward, reward_breakdown = reward_system.compute_dense_reward(
        gripper_pos=gripper_pos,
        gripper_quat=gripper_quat,
        parts_poses=parts_poses,
        gripper_state=gripper_state,
        assembled=False
    )
    
    print(f"Total reward: {total_reward:.4f}")
    print("Reward breakdown:")
    for component, value in reward_breakdown.items():
        print(f"  {component}: {value:.4f}")
    
    print(f"Current stage: {reward_system.current_stage}")
    print("Individual reward component test completed!")

if __name__ == "__main__":
    try:
        # Test individual components first
        test_reward_components()
        
        # Then test full environment
        test_dense_rewards()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()