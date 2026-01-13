#!/usr/bin/env python3
"""
Visualize CVAE-constructed rewards vs actual environment rewards on sample episodes.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')
from simple_cvae_mujoco import SimpleMuJoCoDataset, train_cvae, CVAERewardComputer, load_cvae_from_checkpoint

def visualize_rewards_comparison(env_name='halfcheetah', num_episodes=5, cvae_checkpoint=None):
    """
    Compare CVAE-constructed rewards with actual environment rewards.

    Args:
        env_name: Environment name (halfcheetah, walker2d, ant, hopper)
        num_episodes: Number of episodes to visualize
        cvae_checkpoint: Optional path to pre-trained CVAE checkpoint
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {env_name} dataset...")
    dataset = SimpleMuJoCoDataset(env_name, "medium-expert")

    # Load or train CVAE
    if cvae_checkpoint and os.path.exists(cvae_checkpoint):
        print(f"Loading CVAE from {cvae_checkpoint}")
        cvae_model = load_cvae_from_checkpoint(cvae_checkpoint, device=device)
    else:
        print("Training CVAE (this may take a few minutes)...")
        cvae_model = train_cvae(
            dataset,
            hidden_dim=256,
            num_epochs=50,
            batch_size=256,
            device=device,
            use_wandb=False
        )
        # Save for future use
        os.makedirs(f"./cvae_checkpoints/{env_name}", exist_ok=True)
        save_path = f"./cvae_checkpoints/{env_name}/cvae_model.pt"
        save_dict = {
            'model_state_dict': cvae_model.state_dict(),
            'state_dim': dataset.state_dim,
            'action_dim': dataset.action_dim,
            'hidden_dim': 256
        }
        if hasattr(cvae_model, 'state_mean'):
            save_dict['state_mean'] = cvae_model.state_mean.cpu()
            save_dict['state_std'] = cvae_model.state_std.cpu()
            save_dict['action_mean'] = cvae_model.action_mean.cpu()
            save_dict['action_std'] = cvae_model.action_std.cpu()
        torch.save(save_dict, save_path)
        print(f"Saved CVAE to {save_path}")

    # Set up reward computer
    reward_computer = CVAERewardComputer(cvae_model, device)

    # Visualize episodes
    fig, axes = plt.subplots(num_episodes, 3, figsize=(15, 4*num_episodes))
    if num_episodes == 1:
        axes = axes.reshape(1, -1)

    # Statistics for summary
    all_correlations = []
    all_cvae_ranges = []
    all_env_ranges = []

    for ep_idx in range(min(num_episodes, dataset.n_episodes)):
        start = dataset.episode_starts[ep_idx]
        end = dataset.episode_ends[ep_idx]

        episode_states = dataset.states[start:end]
        env_rewards = dataset.rewards[start:end]
        true_progress = dataset.progress_labels[start:end]

        # Compute CVAE rewards
        cvae_rewards = []
        for state in episode_states:
            reward = reward_computer.compute_progress_reward(state)
            cvae_rewards.append(reward)
        cvae_rewards = np.array(cvae_rewards)

        # Scale CVAE rewards to match training (0-1 to 0-10)
        cvae_rewards_scaled = cvae_rewards * 10.0

        timesteps = np.arange(len(episode_states))

        # Compute statistics
        correlation = np.corrcoef(env_rewards, cvae_rewards_scaled)[0, 1]
        all_correlations.append(correlation)
        all_cvae_ranges.append((cvae_rewards_scaled.min(), cvae_rewards_scaled.max()))
        all_env_ranges.append((env_rewards.min(), env_rewards.max()))

        # Plot 1: Raw rewards comparison
        axes[ep_idx, 0].plot(timesteps, env_rewards, label='Environment Reward', alpha=0.7, linewidth=2)
        axes[ep_idx, 0].plot(timesteps, cvae_rewards_scaled, label='CVAE Reward (scaled)', alpha=0.7, linewidth=2)
        axes[ep_idx, 0].set_xlabel('Timestep')
        axes[ep_idx, 0].set_ylabel('Reward')
        axes[ep_idx, 0].set_title(f'Episode {ep_idx+1}: Reward Comparison (Corr={correlation:.3f})')
        axes[ep_idx, 0].legend()
        axes[ep_idx, 0].grid(True, alpha=0.3)

        # Plot 2: Cumulative rewards
        cum_env = np.cumsum(env_rewards)
        cum_cvae = np.cumsum(cvae_rewards_scaled)
        axes[ep_idx, 1].plot(timesteps, cum_env, label='Cumulative Env Reward', alpha=0.7, linewidth=2)
        axes[ep_idx, 1].plot(timesteps, cum_cvae, label='Cumulative CVAE Reward', alpha=0.7, linewidth=2)
        axes[ep_idx, 1].set_xlabel('Timestep')
        axes[ep_idx, 1].set_ylabel('Cumulative Reward')
        axes[ep_idx, 1].set_title(f'Episode {ep_idx+1}: Cumulative Returns')
        axes[ep_idx, 1].legend()
        axes[ep_idx, 1].grid(True, alpha=0.3)

        # Plot 3: CVAE progress vs true progress labels
        axes[ep_idx, 2].plot(timesteps, true_progress, label='True Progress (from cum. rewards)', alpha=0.7, linewidth=2)
        axes[ep_idx, 2].plot(timesteps, cvae_rewards, label='CVAE Progress Prediction', alpha=0.7, linewidth=2)
        progress_corr = np.corrcoef(true_progress, cvae_rewards)[0, 1]
        axes[ep_idx, 2].set_xlabel('Timestep')
        axes[ep_idx, 2].set_ylabel('Progress [0-1]')
        axes[ep_idx, 2].set_title(f'Episode {ep_idx+1}: Progress Signals (Corr={progress_corr:.3f})')
        axes[ep_idx, 2].legend()
        axes[ep_idx, 2].grid(True, alpha=0.3)

        # Print episode statistics
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx+1} Statistics (Length: {len(episode_states)} steps)")
        print(f"{'='*60}")
        print(f"Environment Rewards:")
        print(f"  Mean: {env_rewards.mean():.3f}, Std: {env_rewards.std():.3f}")
        print(f"  Range: [{env_rewards.min():.3f}, {env_rewards.max():.3f}]")
        print(f"  Total Return: {env_rewards.sum():.3f}")
        print(f"\nCVAE Rewards (scaled 0-10):")
        print(f"  Mean: {cvae_rewards_scaled.mean():.3f}, Std: {cvae_rewards_scaled.std():.3f}")
        print(f"  Range: [{cvae_rewards_scaled.min():.3f}, {cvae_rewards_scaled.max():.3f}]")
        print(f"  Total Return: {cvae_rewards_scaled.sum():.3f}")
        print(f"\nCorrelation (Env vs CVAE): {correlation:.3f}")
        print(f"Progress Correlation: {progress_corr:.3f}")

    plt.tight_layout()

    # Save figure
    os.makedirs('./reward_visualizations', exist_ok=True)
    save_path = f'./reward_visualizations/{env_name}_reward_comparison_no_recon_loss.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {save_path}")
    print(f"{'='*60}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS {len(all_correlations)} EPISODES")
    print(f"{'='*60}")
    print(f"Correlation (Env vs CVAE Reward):")
    print(f"  Mean: {np.mean(all_correlations):.3f}")
    print(f"  Std:  {np.std(all_correlations):.3f}")
    print(f"  Range: [{np.min(all_correlations):.3f}, {np.max(all_correlations):.3f}]")

    plt.show()

    return cvae_model, dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CVAE vs Environment Rewards')
    parser.add_argument('--env', type=str, default='halfcheetah',
                       choices=['halfcheetah', 'walker2d', 'ant', 'hopper'],
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize')
    parser.add_argument('--cvae_checkpoint', type=str, default=None,
                       help='Path to pre-trained CVAE checkpoint')

    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint = args.cvae_checkpoint
    if checkpoint is None:
        checkpoint = f"./cvae_checkpoints/{args.env}/cvae_model.pt"
        if not os.path.exists(checkpoint):
            checkpoint = None

    visualize_rewards_comparison(args.env, args.episodes, checkpoint)
