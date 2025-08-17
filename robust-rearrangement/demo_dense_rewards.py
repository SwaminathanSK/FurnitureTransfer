#!/usr/bin/env python3
"""
Demo script to show dense rewards in action with your trained policy.
This will print detailed reward information step by step.
"""

import sys
import os
sys.path.insert(0, '/home/swaminathan/git/FurnitureTransfer/robust-rearrangement')

# Import furniture_bench first to avoid PyTorch/Isaac Gym import issues
import furniture_bench  # This must come before torch
import torch
import numpy as np
from omegaconf import OmegaConf
from src.behavior import get_actor
from src.common.tasks import task2idx
from src.gym import get_rl_env
from src.eval.rollout import calculate_success_rate

def demo_with_verbose_rewards():
    """Demo the dense reward system with step-by-step output."""
    
    # Load the policy
    policy_path = "models/rppo/one_leg/low/actor_chkpt.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading policy...")
    checkpoint = torch.load(policy_path, map_location=device)
    config = OmegaConf.create(checkpoint["config"])
    
    # Fix missing config values that are needed for policy loading (based on evaluate_model.py hotfixes)
    
    # Temporary fix for residual missing field
    if "base_policy" in config:
        print("Applying residual field hotfix")
        config.action_dim = config.base_policy.action_dim
    
    # Temporary fix for dagger missing field
    if "student_policy" in config:
        print("Applying dagger field hotfix")
        config.action_dim = config.student_policy.action_dim
        
    # Temporary fix for critic missing field in actor config
    if "critic" in config:
        print("Applying critic field hotfix")
        config.actor.critic = config.critic
        config.actor.init_logstd = config.init_logstd
        config.discount = config.base_policy.discount
    
    print(f"Policy type: {config.get('behavior', 'Unknown')}")
    print(f"Action dim: {config.get('action_dim', 'Missing')}")
    
    # Create actor
    actor = get_actor(cfg=config, device=device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    actor.to(device)
    actor.set_task(task2idx['one_leg'])
    
    print("Creating environment with dense rewards...")
    
    # Create environment - exactly like in evaluate_model.py
    env = get_rl_env(
        gpu_id=0,
        task='one_leg',
        num_envs=1,
        randomness='low',
        observation_space='state',
        max_env_steps=5_000,
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
    print("STARTING DEMO WITH DENSE REWARD VISUALIZATION")
    print("="*80)
    
    obs = env.reset()
    total_reward = 0
    step = 0
    max_steps = 700
    
    while step < max_steps:
        # Get action from policy
        with torch.no_grad():
            # The observation should be a dict, extract the state
            if isinstance(obs, dict):
                # Look for robot state or concatenated state
                state_obs = None
                for key in ['robot_state', 'state', 'observation']:
                    if key in obs:
                        state_obs = obs[key]
                        break
                
                # If not found, try to find any tensor/array
                if state_obs is None:
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() > 10:  # Use numel() for tensors
                                state_obs = value
                                break
                        elif isinstance(value, np.ndarray) and np.prod(value.shape) > 10:
                            state_obs = value
                            break
                
                if state_obs is not None:
                    if isinstance(state_obs, dict):
                        # If it's still a dict, try to extract the first tensor/array value
                        for k, v in state_obs.items():
                            if isinstance(v, torch.Tensor):
                                if v.numel() > 10:  # Use numel() instead of np.array for tensors
                                    state_obs = v
                                    break
                            elif isinstance(v, (np.ndarray, list)) and np.prod(np.array(v).shape) > 10:
                                state_obs = v
                                break
                    
                    if isinstance(state_obs, np.ndarray):
                        state_obs = torch.from_numpy(state_obs).float()
                    elif isinstance(state_obs, list):
                        state_obs = torch.tensor(state_obs, dtype=torch.float32)
                    elif isinstance(state_obs, dict):
                        print("Warning: Still dict after extraction, using random action")
                        action = env.action_space.sample()
                        state_obs = None
                    
                    if state_obs is not None:
                        state_obs = state_obs.to(device)
                        if state_obs.dim() == 1:
                            state_obs = state_obs.unsqueeze(0)
                        
                        action = actor(state_obs)
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        if action.ndim > 1:
                            action = action[0]
                else:
                    print("Warning: Could not extract state from observation, using random action")
                    action = env.action_space.sample()
            else:
                print("Warning: Unexpected observation format")
                action = env.action_space.sample()
            
            # Ensure action is a numpy array
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if not isinstance(action, np.ndarray):
                action = np.array(action)
        
        # Step the environment
        next_obs, reward, done, info = env.step(action)
        
        # Convert reward to scalar
        reward_scalar = reward.item() if hasattr(reward, 'item') else float(reward)
        total_reward += reward_scalar
        
        # Print detailed reward information every 10 steps or if reward is significant
        if step % 10 == 0 or abs(reward_scalar) > 0.5 or step < 20:
            print(f"\nStep {step:3d}: Reward = {reward_scalar:7.3f}, Total = {total_reward:8.3f}")
            
            # Check if dense reward system is providing breakdown
            if hasattr(env, 'reward_info') and env.reward_info:
                reward_breakdown = env.reward_info.get(0, {})
                if reward_breakdown:
                    print("    Dense Reward Breakdown:")
                    for component, value in reward_breakdown.items():
                        if abs(value) > 1e-6:  # Only show non-zero components
                            weighted_value = value * env.dense_reward_systems[0].weights.get(component, 1.0)
                            print(f"      {component:12s}: {value:6.3f} (weighted: {weighted_value:6.3f})")
            
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
        demo_with_verbose_rewards()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()