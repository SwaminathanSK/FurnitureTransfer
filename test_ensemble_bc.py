#!/usr/bin/env python3
"""
Ensemble Behavioral Cloning with CVAE
Tests ensemble approaches to reduce variance in Latent BC predictions
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gym
import d4rl
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for ensemble BC experiments"""
    env_name: str
    dataset_name: str
    latent_extra_dim: int
    device: str
    bc_hidden_dim: int = 256
    bc_num_layers: int = 3
    bc_dropout: float = 0.1
    bc_learning_rate: float = 3e-4
    bc_batch_size: int = 256
    bc_epochs: int = 100
    eval_episodes: int = 100
    save_dir: str = "./outputs/ensemble_bc"


class HigherDimLatentCVAE(nn.Module):
    """CVAE with higher-dimensional latent space"""
    def __init__(self, state_dim, action_dim, extra_latent_dim=1,
                 hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extra_latent_dim = extra_latent_dim
        self.latent_dim = action_dim + extra_latent_dim

        # Encoder: (state, action) -> latent
        encoder_layers = []
        input_dim = state_dim + action_dim
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.latent_dim)

        # Decoder: (state, latent) -> action
        decoder_layers = []
        input_dim = state_dim + self.latent_dim
        for i in range(num_layers):
            decoder_layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        decoder_layers.append(nn.Linear(hidden_dim, action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def forward(self, state, action):
        mu, logvar = self.encode(state, action)
        z = self.reparameterize(mu, logvar)
        recon_action = self.decode(state, z)
        return recon_action, mu, logvar

    def get_latent_action(self, state, action):
        """Encode state-action pair to latent action (deterministic)"""
        mu, _ = self.encode(state, action)
        return mu


class BCPolicy(nn.Module):
    """Standard BC policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        input_dim = state_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class EnsembleBC:
    """Ensemble of BC policies for variance reduction"""
    def __init__(self, policies: List[BCPolicy], cvae: Optional[HigherDimLatentCVAE] = None):
        self.policies = policies
        self.cvae = cvae
        self.num_policies = len(policies)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Predict action by averaging over ensemble"""
        predictions = []
        for policy in self.policies:
            pred = policy(state)
            predictions.append(pred)

        # Average predictions
        mean_pred = torch.stack(predictions).mean(dim=0)

        # If using CVAE, decode from latent space
        if self.cvae is not None:
            mean_pred = self.cvae.decode(state, mean_pred)

        return mean_pred

    def predict_with_variance(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict action with uncertainty estimate"""
        predictions = []
        for policy in self.policies:
            pred = policy(state)
            if self.cvae is not None:
                pred = self.cvae.decode(state, pred)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


def load_cvae_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load CVAE from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cvae = HigherDimLatentCVAE(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        extra_latent_dim=checkpoint['extra_latent_dim'],
        hidden_dim=256,
        num_layers=3,
        dropout=0.1
    ).to(device)

    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()

    return cvae, checkpoint


def load_d4rl_dataset(env_name: str, dataset_name: str):
    """Load D4RL dataset"""
    env_id = f"{env_name}-{dataset_name}-v2"

    # Try different versions
    for version in ['v2', 'v1', 'v0']:
        try:
            env_id = f"{env_name}-{dataset_name}-{version}"
            env = gym.make(env_id)
            dataset = d4rl.qlearning_dataset(env)
            print(f"Loaded dataset: {env_id}")
            return env, dataset
        except:
            continue

    raise ValueError(f"Could not load dataset for {env_name}-{dataset_name}")


def train_bc_policy(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_dim: int,
    action_dim: int,
    config: ExperimentConfig,
    seed: int
) -> BCPolicy:
    """Train a single BC policy"""
    # Set seed for this policy
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    policy = BCPolicy(state_dim, action_dim, config.bc_hidden_dim,
                     config.bc_num_layers, config.bc_dropout).to(config.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.bc_learning_rate)

    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=config.bc_batch_size,
                           shuffle=True, drop_last=True)

    print(f"\nTraining BC policy (seed={seed})...")
    policy.train()

    for epoch in range(config.bc_epochs):
        total_loss = 0
        num_batches = 0

        for batch_states, batch_actions in dataloader:
            pred_actions = policy(batch_states)
            loss = F.mse_loss(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config.bc_epochs}, Loss: {avg_loss:.4f}")

    policy.eval()
    return policy


def evaluate_ensemble(
    env,
    ensemble: EnsembleBC,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: Optional[np.ndarray],
    action_std: Optional[np.ndarray],
    num_episodes: int,
    device: str,
    with_variance: bool = False
) -> Dict:
    """Evaluate ensemble policy"""
    returns = []
    episode_lengths = []

    if with_variance:
        action_uncertainties = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        if with_variance:
            episode_uncertainties = []

        while not done:
            # Normalize state
            state_normalized = (state - state_mean) / state_std
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(device)

            with torch.no_grad():
                if with_variance:
                    action_pred, action_std_pred = ensemble.predict_with_variance(state_tensor)
                    episode_uncertainties.append(action_std_pred.mean().cpu().numpy())
                else:
                    action_pred = ensemble.predict(state_tensor)

                action = action_pred.cpu().numpy()[0]

            # Denormalize action if needed
            if action_mean is not None and action_std is not None:
                action = action * action_std + action_mean

            state, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        episode_lengths.append(episode_length)

        if with_variance:
            action_uncertainties.append(np.mean(episode_uncertainties))

    results = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(episode_lengths),
    }

    if with_variance:
        results['mean_uncertainty'] = np.mean(action_uncertainties)
        results['std_uncertainty'] = np.std(action_uncertainties)

    return results


def main():
    parser = argparse.ArgumentParser(description='Ensemble BC with CVAE')
    parser.add_argument('--env', type=str, default='walker2d',
                       help='Environment name')
    parser.add_argument('--dataset', type=str, default='medium-replay',
                       help='Dataset name')
    parser.add_argument('--cvae-checkpoint', type=str, required=True,
                       help='Path to CVAE checkpoint')
    parser.add_argument('--num-policies', type=int, default=5,
                       help='Number of policies in ensemble')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 42, 123],
                       help='Seeds for ensemble policies')
    parser.add_argument('--bc-epochs', type=int, default=100,
                       help='BC training epochs per policy')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')

    args = parser.parse_args()

    # Load CVAE
    print("Loading CVAE checkpoint...")
    cvae, cvae_checkpoint = load_cvae_checkpoint(args.cvae_checkpoint, args.device)

    # Get normalization stats
    state_mean = cvae_checkpoint['state_mean'].cpu().numpy()
    state_std = cvae_checkpoint['state_std'].cpu().numpy()
    action_mean = cvae_checkpoint['action_mean'].cpu().numpy()
    action_std = cvae_checkpoint['action_std'].cpu().numpy()

    state_dim = cvae_checkpoint['state_dim']
    action_dim = cvae_checkpoint['action_dim']
    latent_dim = cvae.latent_dim

    print(f"CVAE: state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}")

    # Load dataset
    print("\nLoading dataset...")
    env, dataset = load_d4rl_dataset(args.env, args.dataset)

    # Normalize and convert to latent space
    states = (dataset['observations'] - state_mean) / state_std
    actions_normalized = (dataset['actions'] - action_mean) / action_std

    states_tensor = torch.FloatTensor(states).to(args.device)
    actions_tensor = torch.FloatTensor(actions_normalized).to(args.device)

    # Encode actions to latent space
    print("Encoding actions to latent space...")
    with torch.no_grad():
        latent_actions = cvae.get_latent_action(states_tensor, actions_tensor)

    # Create config
    config = ExperimentConfig(
        env_name=args.env,
        dataset_name=args.dataset,
        latent_extra_dim=cvae_checkpoint['extra_latent_dim'],
        device=args.device,
        bc_epochs=args.bc_epochs,
        eval_episodes=args.eval_episodes
    )

    # Train ensemble of policies
    print(f"\n{'='*60}")
    print(f"Training Ensemble of {args.num_policies} BC Policies")
    print(f"{'='*60}")

    policies = []
    ensemble_seeds = args.seeds[:args.num_policies]

    for i, seed in enumerate(ensemble_seeds):
        print(f"\n--- Policy {i+1}/{args.num_policies} (seed={seed}) ---")
        policy = train_bc_policy(
            states_tensor, latent_actions,
            state_dim, latent_dim,
            config, seed
        )
        policies.append(policy)

    # Create ensemble
    ensemble = EnsembleBC(policies, cvae)

    # Evaluate ensemble
    print(f"\n{'='*60}")
    print("Evaluating Ensemble")
    print(f"{'='*60}")

    results = evaluate_ensemble(
        env, ensemble,
        state_mean, state_std,
        action_mean, action_std,
        args.eval_episodes,
        args.device,
        with_variance=True
    )

    # Also evaluate individual policies for comparison
    print("\nEvaluating individual policies...")
    individual_results = []
    for i, policy in enumerate(policies):
        single_ensemble = EnsembleBC([policy], cvae)
        result = evaluate_ensemble(
            env, single_ensemble,
            state_mean, state_std,
            action_mean, action_std,
            args.eval_episodes,
            args.device,
            with_variance=False
        )
        individual_results.append(result['mean_return'])
        print(f"Policy {i+1}: {result['mean_return']:.2f} ± {result['std_return']:.2f}")

    # Print results
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Ensemble Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Individual Mean: {np.mean(individual_results):.2f} ± {np.std(individual_results):.2f}")
    print(f"Ensemble Improvement: {results['mean_return'] - np.mean(individual_results):.2f}")
    print(f"Mean Action Uncertainty: {results['mean_uncertainty']:.4f}")
    print(f"D4RL Score (Ensemble): {env.get_normalized_score(results['mean_return']) * 100:.2f}")
    print(f"D4RL Score (Individual Avg): {env.get_normalized_score(np.mean(individual_results)) * 100:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.save_dir, f"{args.env}_ensemble_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    results_dict = {
        'ensemble_mean': results['mean_return'],
        'ensemble_std': results['std_return'],
        'individual_returns': individual_results,
        'individual_mean': np.mean(individual_results),
        'individual_std': np.std(individual_results),
        'mean_uncertainty': results['mean_uncertainty'],
        'num_policies': args.num_policies,
        'seeds': ensemble_seeds,
        'cvae_checkpoint': args.cvae_checkpoint,
        'env': args.env,
        'dataset': args.dataset
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
