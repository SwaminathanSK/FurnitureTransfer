#!/usr/bin/env python3
"""
Record comparison videos between standard BC and latent CVAE BC.
"""
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Force CPU rendering

import sys
import torch
import torch.nn as nn
import numpy as np
import gym
import d4rl
import argparse
import imageio
from pathlib import Path
from tqdm import tqdm

sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

# Model definitions (matching the training script)
class HigherDimLatentCVAE(nn.Module):
    """CVAE with latent_dim = action_dim + extra_dim"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extra_latent_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = action_dim + extra_latent_dim
        self.extra_latent_dim = extra_latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: (state, action) -> latent parameters
        encoder_layers = []
        encoder_input_dim = state_dim + action_dim

        encoder_layers.extend([
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(num_layers - 2):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.latent_mean = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, self.latent_dim)

        # Decoder: (state, latent) -> action
        decoder_layers = []
        decoder_input_dim = state_dim + self.latent_dim

        decoder_layers.extend([
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(num_layers - 2):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        decoder_layers.append(nn.Linear(hidden_dim, action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode (state, latent) to action."""
        x = torch.cat([state, latent], dim=-1)
        action = self.decoder(x)
        return action


class BCPolicy(nn.Module):
    """Behavioral Cloning Policy."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.output_dim = output_dim

        layers = []
        layers.extend([
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class LatentBCAgent:
    """Agent that uses BC policy + CVAE decoder."""

    def __init__(self, bc_policy, cvae, state_mean, state_std, action_mean, action_std, device):
        self.bc_policy = bc_policy
        self.cvae = cvae
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device

        self.bc_policy.eval()
        self.cvae.eval()

    def get_action(self, state):
        """Get action from state."""
        # Normalize state
        state_norm = (state - self.state_mean) / self.state_std
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # BC policy predicts latent action
            latent_action = self.bc_policy(state_tensor)
            # CVAE decoder maps to real action
            action_norm = self.cvae.decode(state_tensor, latent_action)
            action = action_norm.cpu().numpy()[0]

        # Denormalize action
        action = action * self.action_std + self.action_mean
        return action


class StandardBCAgent:
    """Agent that uses standard BC policy."""

    def __init__(self, bc_policy, state_mean, state_std, action_mean, action_std, device):
        self.bc_policy = bc_policy
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device

        self.bc_policy.eval()

    def get_action(self, state):
        """Get action from state."""
        # Normalize state
        state_norm = (state - self.state_mean) / self.state_std
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_norm = self.bc_policy(state_tensor)
            action = action_norm.cpu().numpy()[0]

        # Denormalize action
        action = action * self.action_std + self.action_mean
        return action


def load_normalization_stats(env_name, dataset_name):
    """Load dataset to get normalization statistics."""
    env_id = f"{env_name}-{dataset_name}-v2"
    env = gym.make(env_id)
    dataset = env.get_dataset()

    states = dataset['observations']
    actions = dataset['actions']

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6
    action_mean = np.mean(actions, axis=0)
    action_std = np.std(actions, axis=0) + 1e-6

    return state_mean, state_std, action_mean, action_std


def record_videos(agent, agent_name, env_name, output_dir, n_episodes=10):
    """Record videos for an agent."""

    # Create environment
    env_id = f"{env_name}-medium-replay-v2"
    env = gym.make(env_id)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRecording {n_episodes} episodes for {agent_name} to {output_dir}")

    returns = []

    for ep in tqdm(range(n_episodes), desc=f"Recording {agent_name}"):
        frames = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        episode_return = 0
        steps = 0

        while not done and steps < 1000:  # Max 1000 steps per episode
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
            steps += 1

        returns.append(episode_return)

        # Save video with imageio
        if frames:
            video_path = os.path.join(output_dir, f"{agent_name}_ep{ep:02d}_return{int(episode_return)}.mp4")
            imageio.mimsave(video_path, frames, fps=30)

    env.close()

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"✓ {agent_name}: Mean return = {mean_return:.2f} ± {std_return:.2f}")

    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Directory containing the trained models (e.g., outputs/latent_cvae_comparison/walker2d_full_20251022_022935)')
    parser.add_argument('--env', type=str, default='walker2d',
                        help='Environment name')
    parser.add_argument('--dataset', type=str, default='medium-replay',
                        help='Dataset name')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='./videos/bc_comparison',
                        help='Output directory for videos')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load normalization stats
    print("\nLoading normalization statistics...")
    state_mean, state_std, action_mean, action_std = load_normalization_stats(args.env, args.dataset)
    state_mean = state_mean
    state_std = state_std
    action_mean = action_mean
    action_std = action_std

    # Determine dimensions
    env_id = f"{args.env}-{args.dataset}-v2"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Load models
    run_dir = Path(args.run_dir)
    cvae_path = run_dir / f"cvae_{args.env}.pt"
    bc_latent_path = run_dir / f"bc_latent_{args.env}.pt"
    bc_standard_path = run_dir / f"bc_standard_{args.env}.pt"

    print(f"\nLoading models from {run_dir}")

    # Load CVAE
    cvae_checkpoint = torch.load(cvae_path, map_location=device)
    extra_latent_dim = cvae_checkpoint['extra_latent_dim']
    cvae = HigherDimLatentCVAE(state_dim, action_dim, extra_latent_dim).to(device)
    cvae.load_state_dict(cvae_checkpoint['model_state_dict'])
    cvae.eval()
    print(f"✓ Loaded CVAE (latent_dim = {action_dim + extra_latent_dim})")

    # Load latent BC policy
    bc_latent_checkpoint = torch.load(bc_latent_path, map_location=device)
    bc_latent_policy = BCPolicy(state_dim, action_dim + extra_latent_dim).to(device)
    bc_latent_policy.load_state_dict(bc_latent_checkpoint['bc_policy_state_dict'])
    bc_latent_policy.eval()
    print("✓ Loaded Latent BC policy")

    # Load standard BC policy
    bc_standard_checkpoint = torch.load(bc_standard_path, map_location=device)
    bc_standard_policy = BCPolicy(state_dim, action_dim).to(device)
    bc_standard_policy.load_state_dict(bc_standard_checkpoint['model_state_dict'])
    bc_standard_policy.eval()
    print("✓ Loaded Standard BC policy")

    # Create agents
    latent_agent = LatentBCAgent(bc_latent_policy, cvae, state_mean, state_std, action_mean, action_std, device)
    standard_agent = StandardBCAgent(bc_standard_policy, state_mean, state_std, action_mean, action_std, device)

    # Create output directories
    timestamp = run_dir.name
    latent_output = Path(args.output) / timestamp / "latent_bc"
    standard_output = Path(args.output) / timestamp / "standard_bc"

    # Record videos
    latent_returns = record_videos(latent_agent, "latent_bc", args.env, latent_output, args.episodes)
    standard_returns = record_videos(standard_agent, "standard_bc", args.env, standard_output, args.episodes)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Standard BC:  {np.mean(standard_returns):.2f} ± {np.std(standard_returns):.2f}")
    print(f"Latent BC:    {np.mean(latent_returns):.2f} ± {np.std(latent_returns):.2f}")
    improvement = ((np.mean(latent_returns) - np.mean(standard_returns)) / np.mean(standard_returns)) * 100
    print(f"Improvement:  {improvement:+.2f}%")
    print(f"\nVideos saved to: {args.output}/{timestamp}/")
    print("="*60)


if __name__ == "__main__":
    main()
