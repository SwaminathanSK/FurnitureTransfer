#!/usr/bin/env python3
"""
Simple Behavior Cloning (BC) baseline for MuJoCo D4RL tasks.
Pure supervised learning - learns to clone expert demonstrations.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import gym
import d4rl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm, trange
import argparse


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, act_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)

    def act(self, obs, deterministic=True):
        """Get action for evaluation."""
        with torch.no_grad():
            action = self.forward(obs)
        return action


class D4RLDataset(Dataset):
    """PyTorch dataset wrapper for D4RL datasets."""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def evaluate_policy(env, policy, max_episode_steps=1000, n_episodes=10):
    """Evaluate policy and return episode returns."""
    policy.eval()
    returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        episode_return = 0
        for _ in range(max_episode_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda()

            action = policy.act(obs_tensor, deterministic=True)
            action = action.cpu().numpy().squeeze()

            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            episode_return += reward

            if done:
                break

        returns.append(episode_return)

    policy.train()
    return np.array(returns)


def train_bc(
    env_name,
    log_dir,
    seed=0,
    hidden_dim=256,
    n_hidden=2,
    n_steps=1_000_000,
    batch_size=256,
    learning_rate=3e-4,
    eval_period=5000,
    n_eval_episodes=10,
    max_episode_steps=1000
):
    """Train BC policy on D4RL dataset."""

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = Path(log_dir) / env_name / f'bc_seed{seed}'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BC on {env_name}")
    print(f"Log dir: {log_dir}")
    print(f"Device: {device}")

    # Load environment and dataset
    env = gym.make(env_name)
    env.reset(seed=seed)
    dataset = d4rl.qlearning_dataset(env)

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    print(f"Dataset size: {len(dataset['observations'])}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    # Create policy
    policy = BCPolicy(obs_dim, act_dim, hidden_dim, n_hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    train_dataset = D4RLDataset(dataset['observations'], dataset['actions'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Training loop
    policy.train()
    step = 0
    epoch = 0
    losses = []

    # Results tracking
    results = {
        'step': [],
        'return_mean': [],
        'return_std': [],
        'normalized_return_mean': [],
        'normalized_return_std': [],
        'loss': []
    }

    print(f"\nTraining for {n_steps} steps...")
    pbar = tqdm(total=n_steps, desc="BC Training")

    while step < n_steps:
        epoch += 1

        for obs_batch, act_batch in train_loader:
            if step >= n_steps:
                break

            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Forward pass
            pred_actions = policy(obs_batch)
            loss = F.mse_loss(pred_actions, act_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1
            pbar.update(1)

            # Evaluate periodically
            if step % eval_period == 0:
                eval_returns = evaluate_policy(env, policy, max_episode_steps, n_eval_episodes)
                normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0

                avg_loss = np.mean(losses)
                losses = []

                results['step'].append(step)
                results['return_mean'].append(eval_returns.mean())
                results['return_std'].append(eval_returns.std())
                results['normalized_return_mean'].append(normalized_returns.mean())
                results['normalized_return_std'].append(normalized_returns.std())
                results['loss'].append(avg_loss)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'return': f'{eval_returns.mean():.1f}',
                    'norm_return': f'{normalized_returns.mean():.1f}'
                })

                print(f"\nStep {step}:")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Return: {eval_returns.mean():.2f} ± {eval_returns.std():.2f}")
                print(f"  Normalized Return: {normalized_returns.mean():.2f} ± {normalized_returns.std():.2f}")

    pbar.close()

    # Save final model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'args': {
            'env_name': env_name,
            'seed': seed,
            'hidden_dim': hidden_dim,
            'n_hidden': n_hidden,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, log_dir / 'final.pt')

    # Save results
    np.savez(log_dir / 'results.npz', **results)

    print(f"\nTraining complete!")
    print(f"Final normalized return: {results['normalized_return_mean'][-1]:.2f} ± {results['normalized_return_std'][-1]:.2f}")
    print(f"Model saved to: {log_dir / 'final.pt'}")

    return policy, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BC baseline on D4RL')
    parser.add_argument('--env-name', type=str, required=True,
                       help='D4RL environment name (e.g., halfcheetah-medium-expert-v2)')
    parser.add_argument('--log-dir', type=str, default='./baseline_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--n-hidden', type=int, default=2,
                       help='Number of hidden layers')
    parser.add_argument('--n-steps', type=int, default=1_000_000,
                       help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--eval-period', type=int, default=5000,
                       help='Evaluation period')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                       help='Max episode steps')

    args = parser.parse_args()

    train_bc(
        env_name=args.env_name,
        log_dir=args.log_dir,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_period=args.eval_period,
        n_eval_episodes=args.n_eval_episodes,
        max_episode_steps=args.max_episode_steps
    )
