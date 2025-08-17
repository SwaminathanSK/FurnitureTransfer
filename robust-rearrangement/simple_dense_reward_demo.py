#!/usr/bin/env python3
"""
Simple demo to show dense rewards in action with your trained policy.
This focuses on showing the dense reward system working.
"""

import sys
import os
sys.path.insert(0, '/home/swaminathan/git/FurnitureTransfer/robust-rearrangement')

# Import furniture_bench first to avoid PyTorch/Isaac Gym import issues
import furniture_bench
import torch
import numpy as np
from omegaconf import OmegaConf
from src.gym import get_rl_env
from src.behavior import get_actor
from src.common.tasks import task2idx

def load_policy():
    """Load the trained policy."""
    policy_path = "models/rppo/one_leg/low/actor_chkpt.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading policy from: {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device)
    config = OmegaConf.create(checkpoint["config"])
    
    # Fix missing config values (hotfix from evaluate_model.py)
    if "base_policy" in config:
        config.action_dim = config.base_policy.action_dim
    if "critic" in config:
        config.actor.critic = config.critic
        config.actor.init_logstd = config.init_logstd
        config.discount = config.base_policy.discount
    
    # Create and load actor
    actor = get_actor(cfg=config, device=device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    actor.set_task(task2idx['one_leg'])
    
    print(f"Policy loaded successfully! Action dim: {config.action_dim}")
    return actor, device

def simple_dense_reward_demo():
    """Demo with trained policy to show dense rewards."""
    
    # Load the trained policy first
    actor, device = load_policy()
    
    print("Creating environment with dense rewards...")
    
    # Create environment - exactly like in evaluate_model.py
    env = get_rl_env(
        gpu_id=0,
        task='one_leg',
        num_envs=1,
        randomness='low',
        observation_space='state',
        max_env_steps=700,
        resize_img=False,
        act_rot_repr="rot_6d", 
        action_type='pos',
        april_tags=False,
        verbose=True,
        headless=False,  # Show visualization
    )
    
    print(f"Environment created: {type(env)}")
    print(f"Has dense reward systems: {hasattr(env, 'dense_reward_systems')}")
    if hasattr(env, 'dense_reward_systems'):
        print(f"Number of dense reward systems: {len(env.dense_reward_systems) if env.dense_reward_systems else 0}")
    
    # Run one episode with detailed logging
    print("\n" + "="*80)
    print("STARTING DENSE REWARD DEMO WITH TRAINED POLICY")
    print("="*80)
    
    obs = env.reset()
    total_reward = 0
    step = 0
    max_steps = 500  # Longer demo to see full assembly
    
    while step < max_steps:
        # Use trained policy to get action
        with torch.no_grad():
            # Extract observation for the policy
            # The observation is a dict, we need to get the right format
            if isinstance(obs, dict):
                # Build robot state from individual components
                robot_state_parts = []
                robot_keys = ['robot_state/ee_pos', 'robot_state/ee_quat', 'robot_state/ee_pos_vel', 
                             'robot_state/ee_ori_vel', 'robot_state/gripper_width', 'robot_state/joint_positions',
                             'robot_state/joint_velocities', 'robot_state/joint_torques']
                
                for key in robot_keys:
                    if key in obs:
                        value = obs[key]
                        if isinstance(value, torch.Tensor):
                            robot_state_parts.append(value.flatten())
                        else:
                            robot_state_parts.append(torch.tensor(value, dtype=torch.float32, device=device).flatten())
                
                # Get parts poses
                parts_poses = obs.get('parts_poses', None)
                if parts_poses is not None:
                    if not isinstance(parts_poses, torch.Tensor):
                        parts_poses = torch.tensor(parts_poses, dtype=torch.float32, device=device)
                
                # Create the observation format expected by the policy
                if robot_state_parts and parts_poses is not None:
                    robot_state = torch.cat(robot_state_parts)
                    
                    # Create observation dict with proper format
                    obs_dict = {
                        'robot_state': robot_state.unsqueeze(0),  # Add batch dimension
                        'parts_poses': parts_poses.unsqueeze(0)   # Add batch dimension
                    }
                    
                    # Use the action method for inference
                    action = actor.action(obs_dict)
                    
                    # Convert to numpy for environment
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy().squeeze()  # Remove batch dimension
                else:
                    print("Warning: Could not extract observation, using random action")
                    action = env.action_space.sample()
            else:
                print("Warning: Unexpected observation format, using random action")
                action = env.action_space.sample()
        
        # Step the environment
        next_obs, reward, done, info = env.step(action)
        
        # Convert reward to scalar
        reward_scalar = reward.item() if hasattr(reward, 'item') else float(reward)
        total_reward += reward_scalar
        
        # Print detailed reward information every 15 steps or if reward is significant
        if step % 15 == 0 or abs(reward_scalar) > 0.5 or step < 30:
            print(f"\nStep {step:3d}: Reward = {reward_scalar:7.3f}, Total = {total_reward:8.3f}")
            
            # Check if dense reward system is providing breakdown
            if hasattr(env, 'reward_info') and env.reward_info:
                reward_breakdown = env.reward_info.get(0, {})
                if reward_breakdown:
                    print("    Dense Reward Breakdown:")
                    for component, value in reward_breakdown.items():
                        if abs(value) > 1e-6:  # Only show non-zero components
                            weighted_value = value * env.dense_reward_systems[0].weights.get(component, 1.0)
                            print(f"      {component:12s}: raw={value:6.3f}, weighted={weighted_value:6.3f}")
            
            # Show any non-zero rewards even if breakdown isn't available
            elif abs(reward_scalar) > 0.01:
                print(f"    Non-zero reward detected: {reward_scalar:.3f}")
            
            # Print current stage
            if hasattr(env, 'dense_reward_systems') and env.dense_reward_systems:
                dense_system = env.dense_reward_systems[0]
                if hasattr(dense_system, 'current_stage'):
                    print(f"    Current Stage: {dense_system.current_stage}")
                    
                    # Show stage completion
                    completed = [k for k, v in dense_system.stage_completion.items() if v]
                    if completed:
                        print(f"    Completed Stages: {completed}")
        
        # Check for done condition
        if done:
            print(f"\nEpisode completed at step {step}!")
            break
            
        obs = next_obs
        step += 1
    
    print(f"\n" + "="*80)
    print(f"DEMO COMPLETED")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward/max(1, step):.3f}")
    
    # Final stage status
    if hasattr(env, 'dense_reward_systems') and env.dense_reward_systems:
        dense_system = env.dense_reward_systems[0]
        print(f"Final stage: {dense_system.current_stage}")
        completed_count = sum(dense_system.stage_completion.values())
        total_stages = len(dense_system.stage_completion)
        print(f"Stages completed: {completed_count}/{total_stages}")
        
        # Show all stage completions
        print("Stage completion status:")
        for stage, completed in dense_system.stage_completion.items():
            status = "✓" if completed else "✗"
            print(f"  {status} {stage}")
    
    print("="*80)

if __name__ == "__main__":
    try:
        simple_dense_reward_demo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()