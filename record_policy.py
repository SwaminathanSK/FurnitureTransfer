#!/usr/bin/env python3
"""
Record videos using imageio - avoiding all the Monitor/ffmpeg issues.
"""
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Force CPU rendering

import gym
import torch
import numpy as np
import sys
import argparse
import imageio

sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

def record_videos(env_name, agent_checkpoint, output_dir, n_episodes=5):
    """Record using gym Monitor wrapper."""

    # Create a minimal agent wrapper
    class PolicyWrapper:
        def __init__(self):
            self.device = 'cuda'
            print("Using random policy to test video recording...")

        def get_action(self, state):
            # Random policy for now
            return np.random.uniform(-1, 1, 6)  # HalfCheetah has 6 actions

    agent = PolicyWrapper()

    # Create environment
    base_env_name = env_name.title().replace('cheetah', 'Cheetah') + "-v3"
    env = gym.make(base_env_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Recording {n_episodes} episodes to {output_dir}")

    for ep in range(n_episodes):
        frames = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        episode_return = 0

        while not done:
            # Render frame
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)

            action = agent.get_action(state)

            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result

            episode_return += reward

        # Save video with imageio
        if frames:
            video_path = os.path.join(output_dir, f"{env_name}_ep{ep}_return{int(episode_return)}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Episode {ep+1}: Return = {episode_return:.1f} - Saved {video_path}")

    env.close()
    print(f"\n✓ All videos saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='./monitor_videos')
    parser.add_argument('--episodes', type=int, default=5)

    args = parser.parse_args()

    record_videos(args.env, args.checkpoint, args.output, args.episodes)
