#!/usr/bin/env python3
"""
Test script for dense reward system with trained policy and visualization.
This loads your trained actor_chkpt.pt policy and runs it with visualization
while printing detailed reward information to the terminal.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the robust-rearrangement path to the Python path
sys.path.insert(0, '/home/swaminathan/git/FurnitureTransfer/robust-rearrangement')

from src.gym import get_rl_env
from src.behavior import get_actor
from src.common.tasks import task2idx
from omegaconf import OmegaConf

def load_policy(policy_path, device='cuda'):
    """Load the trained policy from checkpoint."""
    print(f"Loading policy from: {policy_path}")
    
    checkpoint = torch.load(policy_path, map_location=device)
    
    # Create config from checkpoint
    config = OmegaConf.create(checkpoint["config"])
    print(f"Policy config loaded: {config.get('behavior', 'Unknown behavior')}")
    
    # Create actor
    actor = get_actor(cfg=config, device=device)
    
    # Load state dict
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    actor.to(device)
    
    # Set task
    actor.set_task(task2idx['one_leg'])
    
    print("Policy loaded successfully!")
    return actor, config

def run_policy_with_dense_rewards(policy_path, num_episodes=3, max_steps=500):
    """Run the trained policy with dense reward visualization."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained policy
    actor, config = load_policy(policy_path, device)
    
    # Create environment with visualization enabled
    print("Creating environment with visualization...")
    env = get_rl_env(
        gpu_id=0,
        task='one_leg',
        num_envs=1,
        randomness='low',
        observation_space='state',  # Use state obs as per your policy
        max_env_steps=max_steps,
        resize_img=False,
        act_rot_repr='rot_6d',
        action_type='pos',
        april_tags=False,
        verbose=True,
        headless=False  # Enable visualization
    )
    
    print(f"Environment created successfully!")
    print(f"Dense reward system active: {hasattr(env, 'dense_reward_systems')}")
    
    if hasattr(env, 'dense_reward_systems') and env.dense_reward_systems:
        print(f"Number of dense reward systems: {len(env.dense_reward_systems)}")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # Reset environment
        obs = env.reset()
        total_reward = 0
        episode_rewards = []
        
        print("Starting episode...")
        
        for step in range(max_steps):
            # Get action from policy
            with torch.no_grad():
                # Convert observation to tensor if needed
                if isinstance(obs, dict):
                    # Extract state observation
                    state_obs = None
                    for key in obs.keys():
                        if 'state' in key.lower() or 'robot' in key.lower():
                            state_obs = obs[key]
                            break
                    
                    if state_obs is None:
                        # Try to find any tensor observation
                        for key, value in obs.items():
                            if isinstance(value, (torch.Tensor, np.ndarray)):
                                state_obs = value
                                break
                    
                    if state_obs is None:
                        print("Warning: Could not find state observation, using random action")
                        action = env.action_space.sample()
                    else:
                        if isinstance(state_obs, np.ndarray):
                            state_obs = torch.from_numpy(state_obs).float().to(device)
                        if state_obs.dim() == 1:
                            state_obs = state_obs.unsqueeze(0)  # Add batch dimension
                        
                        action = actor(state_obs)
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        if action.ndim > 1:
                            action = action[0]  # Remove batch dimension
                else:
                    print("Warning: Unexpected observation format, using random action")
                    action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Convert reward to scalar
            if hasattr(reward, 'item'):
                reward_scalar = reward.item()
            elif isinstance(reward, (list, np.ndarray)):
                reward_scalar = float(reward[0]) if len(reward) > 0 else 0.0
            else:
                reward_scalar = float(reward)
            
            total_reward += reward_scalar
            episode_rewards.append(reward_scalar)
            
            # Print detailed reward information every 10 steps
            if step % 10 == 0 or step < 5:
                print(f"\nStep {step:3d}: Reward = {reward_scalar:6.3f}, Total = {total_reward:8.3f}")
                
                # Print reward breakdown if available
                if hasattr(env, 'reward_info') and env.reward_info:
                    reward_breakdown = env.reward_info.get(0, {})
                    if reward_breakdown:
                        print("  Reward Breakdown:")
                        for component, value in reward_breakdown.items():
                            if abs(value) > 1e-6:  # Only show non-zero components
                                print(f"    {component:12s}: {value:6.3f}")
                
                # Print current stage if available
                if (hasattr(env, 'dense_reward_systems') and 
                    env.dense_reward_systems and 
                    hasattr(env.dense_reward_systems[0], 'current_stage')):
                    current_stage = env.dense_reward_systems[0].current_stage
                    stage_completion = env.dense_reward_systems[0].stage_completion
                    completed_stages = [k for k, v in stage_completion.items() if v]
                    print(f"    Current Stage: {current_stage}")
                    print(f"    Completed: {completed_stages}")
            
            # Check for assembly completion
            if hasattr(info, 'get') and info.get('assembly_complete', False):
                print(f"\n🎉 ASSEMBLY COMPLETED at step {step}!")
                break
            
            if done:
                print(f"\nEpisode finished at step {step}")
                break
        
        # Episode summary
        print(f"\n--- Episode {episode + 1} Summary ---")
        print(f"Total steps: {step + 1}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward per step: {total_reward / (step + 1):.3f}")
        print(f"Max single step reward: {max(episode_rewards):.3f}")
        print(f"Min single step reward: {min(episode_rewards):.3f}")
        
        # Final stage information
        if (hasattr(env, 'dense_reward_systems') and 
            env.dense_reward_systems):
            final_stage = env.dense_reward_systems[0].current_stage
            stage_completion = env.dense_reward_systems[0].stage_completion
            completed_count = sum(stage_completion.values())
            print(f"Final stage: {final_stage}")
            print(f"Stages completed: {completed_count}/{len(stage_completion)}")
        
        # Wait for user input before next episode (optional)
        if episode < num_episodes - 1:
            input("\nPress Enter to continue to next episode...")
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETED!")
    print(f"{'='*60}")

def main():
    """Main function to run the test."""
    
    # Path to your trained policy
    policy_path = "/home/swaminathan/git/FurnitureTransfer/robust-rearrangement/models/rppo/one_leg/low/actor_chkpt.pt"
    
    if not Path(policy_path).exists():
        print(f"Error: Policy file not found at {policy_path}")
        print("Please check the path and try again.")
        return
    
    print("Dense Reward System Test with Trained Policy")
    print("=" * 50)
    print(f"Policy: {policy_path}")
    print(f"Task: one_leg")
    print(f"Visualization: Enabled")
    print(f"Dense Rewards: Enabled")
    print("=" * 50)
    
    try:
        run_policy_with_dense_rewards(
            policy_path=policy_path,
            num_episodes=3,  # Run 3 episodes
            max_steps=700    # Max steps per episode
        )
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()