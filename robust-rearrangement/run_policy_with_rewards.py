#!/usr/bin/env python3
"""
Simple script to run your trained policy with dense reward visualization.
Usage: python run_policy_with_rewards.py
"""

import sys
sys.path.insert(0, '/home/swaminathan/git/FurnitureTransfer/robust-rearrangement')

import torch
from pathlib import Path
from omegaconf import OmegaConf
from src.gym import get_rl_env
from src.behavior import get_actor
from src.common.tasks import task2idx

# Configuration
POLICY_PATH = "models/rppo/one_leg/low/actor_chkpt.pt"
NUM_EPISODES = 5
MAX_STEPS = 700

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load policy
    print("Loading policy...")
    checkpoint = torch.load(POLICY_PATH, map_location=device)
    config = OmegaConf.create(checkpoint["config"])
    
    actor = get_actor(cfg=config, device=device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    actor.set_task(task2idx['one_leg'])
    
    # Create environment with visualization
    print("Creating environment...")
    env = get_rl_env(
        gpu_id=0,
        task='one_leg', 
        num_envs=1,
        randomness='low',
        observation_space='state',
        max_env_steps=MAX_STEPS,
        resize_img=False,
        act_rot_repr='rot_6d',
        action_type='pos',
        april_tags=False,
        verbose=False,
        headless=False  # Show visualization
    )
    
    print(f"Dense rewards enabled: {hasattr(env, 'dense_reward_systems')}")
    
    # Run episodes
    for episode in range(NUM_EPISODES):
        print(f"\n=== EPISODE {episode + 1} ===")
        obs = env.reset()
        total_reward = 0
        
        for step in range(MAX_STEPS):
            # Get action from policy
            with torch.no_grad():
                # Find state observation
                state_obs = None
                for key, value in obs.items():
                    if isinstance(value, (torch.Tensor, list)) and len(value) > 10:  # Likely state vector
                        state_obs = torch.tensor(value, dtype=torch.float32, device=device)
                        break
                
                if state_obs is not None:
                    if state_obs.dim() == 1:
                        state_obs = state_obs.unsqueeze(0)
                    action = actor(state_obs).cpu().numpy()[0]
                else:
                    action = env.action_space.sample()
            
            # Step
            obs, reward, done, info = env.step(action)
            reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
            total_reward += reward_val
            
            # Print reward info every 15 steps
            if step % 15 == 0:
                print(f"Step {step:3d}: R={reward_val:6.3f}, Total={total_reward:7.2f}", end="")
                
                # Print stage info
                if hasattr(env, 'dense_reward_systems') and env.dense_reward_systems:
                    stage = env.dense_reward_systems[0].current_stage
                    print(f", Stage: {stage}", end="")
                
                # Print reward breakdown for significant rewards
                if hasattr(env, 'reward_info') and env.reward_info.get(0):
                    breakdown = env.reward_info[0]
                    significant = {k: v for k, v in breakdown.items() if abs(v) > 0.1}
                    if significant:
                        print(f", Key rewards: {significant}", end="")
                
                print()  # New line
            
            if done:
                print(f"Episode done at step {step}")
                break
        
        print(f"Episode {episode + 1} total reward: {total_reward:.2f}")
    
    print("Testing completed!")

if __name__ == "__main__":
    main()