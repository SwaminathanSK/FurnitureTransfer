#!/usr/bin/env python3
"""
Render videos from saved episode data.
Run this SEPARATELY in a fresh Python session to avoid MuJoCo conflicts.
"""
import os
# Use EGL for GPU-accelerated headless rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import gym
import cv2
from tqdm import tqdm
import argparse
import glob

# Force MuJoCo to use software rendering AFTER gym import
try:
    import mujoco_py
    # Override GPU selection
    from mujoco_py.builder import cymj
    cymj.MjRenderContextOffscreen = lambda *args, **kwargs: None
except:
    pass

def render_episode(episode_file, env_name='halfcheetah', output_dir='./videos'):
    """Render a single saved episode to video."""

    # Load episode data
    data = np.load(episode_file)
    states = data['states']
    actions = data['actions']
    ep_return = float(data['episode_return'])

    print(f"Rendering {len(states)} frames from {episode_file}")

    # Create environment
    base_env_name = env_name.title().replace('cheetah', 'Cheetah') + "-v3"
    env = gym.make(base_env_name)

    frames = []
    state = env.reset()

    # Just replay actions instead of setting states (more reliable)
    for i in tqdm(range(len(actions)), desc="Rendering frames"):
        # Render current state
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Render failed at frame {i}: {e}")
            break

        # Execute action
        if i < len(actions):
            step_result = env.step(actions[i])
            if len(step_result) == 5:
                state, _, done, _, _ = step_result
            else:
                state, _, done, _ = step_result

            if done:
                break

    env.close()

    # Save video
    if frames:
        video_path = episode_file.replace('.npz', '.mp4')
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        for frame in tqdm(frames, desc="Writing video"):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"✓ Saved: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Episode .npz file or directory')
    parser.add_argument('--env', type=str, default='halfcheetah')

    args = parser.parse_args()

    # Find episode files
    if os.path.isdir(args.input):
        episode_files = glob.glob(os.path.join(args.input, '*.npz'))
    else:
        episode_files = [args.input]

    print(f"Found {len(episode_files)} episodes to render")

    for ep_file in episode_files:
        try:
            render_episode(ep_file, args.env)
        except Exception as e:
            print(f"Failed to render {ep_file}: {e}")
