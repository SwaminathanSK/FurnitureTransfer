"""
Test Hypothesis: Higher-Dimensional Latent CVAE for Action Learning

This script tests whether using a CVAE with latent_dim = action_dim + 1
improves behavioral cloning performance compared to:
1. Standard BC on original state-action pairs
2. The current approach of using a scalar latent (latent_dim = 1)

Hypothesis:
- The extra dimension in the latent space can capture task progress
- The remaining dimensions encode action information in a compressed form
- This compressed representation might be easier for BC to learn
- The combined latent action (action_dim + 1) may have better properties than raw actions

Experiment Design:
1. Train CVAE with latent_dim = action_dim + 1
2. Train BC policy on (state -> latent_action) pairs from CVAE encoder
3. At test time: BC outputs latent_action -> CVAE decoder produces real action
4. Compare against standard BC trained on (state -> action) pairs
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import argparse
from datetime import datetime

# Add INAC to path for dataset utilities
sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    env_name: str = "hopper"  # Hopper is known to be challenging for BC due to instability
    dataset_name: str = "medium-expert"
    device: str = "cuda"

    # CVAE settings
    cvae_latent_dim_extra: int = 1  # action_dim + this value
    cvae_hidden_dim: int = 256
    cvae_num_layers: int = 3
    cvae_dropout: float = 0.1
    cvae_epochs: int = 50
    cvae_batch_size: int = 256
    cvae_lr: float = 1e-3
    cvae_beta_start: float = 0.0  # Starting KL weight (warmup from this)
    cvae_beta_end: float = 0.01  # Final KL weight (much lower to prevent collapse)
    cvae_beta_warmup_epochs: int = 30  # Number of epochs to warmup beta

    # BC settings
    bc_hidden_dim: int = 256
    bc_num_layers: int = 3
    bc_dropout: float = 0.1
    bc_epochs: int = 100
    bc_batch_size: int = 256
    bc_lr: float = 3e-4

    # Evaluation
    eval_episodes: int = 10

    # Logging
    use_wandb: bool = True
    experiment_name: str = "higher_dim_latent_cvae"
    save_dir: str = "./outputs/latent_cvae_comparison"


class HigherDimLatentCVAE(nn.Module):
    """
    CVAE with latent_dim = action_dim + 1

    The extra dimension can serve as a task progress indicator,
    while the other dimensions encode action information.
    """

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

        print(f"\nHigherDimLatentCVAE Architecture:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Latent dim: {self.latent_dim} (action_dim + {extra_latent_dim})")

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (state, action) to latent distribution parameters."""
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = self.latent_mean(h)
        logvar = self.latent_logvar(h)
        logvar = torch.clamp(logvar, -10, 10)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode (state, latent) to action."""
        x = torch.cat([state, latent], dim=-1)
        action = self.decoder(x)
        return action

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        latent_mean, latent_logvar = self.encode(state, action)
        latent = self.reparameterize(latent_mean, latent_logvar)
        recon_action = self.decode(state, latent)

        return {
            'recon_action': recon_action,
            'latent_mean': latent_mean,
            'latent_logvar': latent_logvar,
            'latent': latent
        }

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute CVAE loss."""
        outputs = self.forward(state, action)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['recon_action'], action)

        # KL divergence loss
        latent_mean = outputs['latent_mean']
        latent_logvar = outputs['latent_logvar']
        kl_loss = -0.5 * torch.mean(
            1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
        )
        kl_loss = torch.clamp(kl_loss, 0, 100)

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def get_latent_action(self, state: torch.Tensor, action: torch.Tensor, batch_size: int = 10000) -> torch.Tensor:
        """Get latent action representation (for BC training) in batches to avoid OOM."""
        self.eval()
        latent_means = []

        with torch.no_grad():
            n_samples = state.shape[0]
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_state = state[i:end_idx]
                batch_action = action[i:end_idx]
                latent_mean, _ = self.encode(batch_state, batch_action)
                latent_means.append(latent_mean)

        return torch.cat(latent_means, dim=0)


class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy.

    Can be used for:
    1. Standard BC: state -> action
    2. Latent BC: state -> latent_action (then use CVAE decoder)
    """

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class LatentBCPolicy(nn.Module):
    """
    BC Policy that uses CVAE latent actions.

    Forward pass:
    1. BC network: state -> latent_action
    2. CVAE decoder: (state, latent_action) -> action
    """

    def __init__(self, bc_policy: BCPolicy, cvae_decoder: HigherDimLatentCVAE):
        super().__init__()
        self.bc_policy = bc_policy
        self.cvae_decoder = cvae_decoder

        # Freeze CVAE decoder during BC training
        for param in self.cvae_decoder.parameters():
            param.requires_grad = False

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state -> latent_action -> real_action"""
        latent_action = self.bc_policy(state)
        real_action = self.cvae_decoder.decode(state, latent_action)
        return real_action


def load_d4rl_dataset(env_name: str, dataset_name: str):
    """Load D4RL dataset (MuJoCo or Adroit environments)."""
    # Adroit environments have different naming convention
    adroit_envs = ['pen', 'hammer', 'door', 'relocate']

    if env_name in adroit_envs:
        # Adroit: pen-human-v0, pen-cloned-v0, etc.
        env_id_options = [
            f"{env_name}-{dataset_name}-v0",
            f"{env_name}-{dataset_name}-v1"
        ]
    else:
        # MuJoCo: halfcheetah-medium-expert-v2, etc.
        env_id_options = [
            f"{env_name}-{dataset_name}-v2",
            f"{env_name}-{dataset_name}-v1",
            f"{env_name}-{dataset_name}-v0"
        ]

    env = None
    for env_id in env_id_options:
        try:
            env = gym.make(env_id)
            print(f"Successfully loaded environment: {env_id}")
            break
        except:
            continue

    if env is None:
        raise ValueError(f"Could not find environment for {env_name}-{dataset_name}")

    dataset = env.get_dataset()

    states = torch.FloatTensor(dataset['observations'])
    actions = torch.FloatTensor(dataset['actions'])

    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    print(f"\nDataset Statistics:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    # Normalize
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0) + 1e-6
    states = (states - state_mean) / state_std

    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0) + 1e-6
    actions = (actions - action_mean) / action_std

    return {
        'states': states,
        'actions': actions,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'env': env
    }


def analyze_latent_collapse(
    cvae: HigherDimLatentCVAE,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: ExperimentConfig
) -> Dict:
    """
    Analyze whether latent dimensions have collapsed.

    Returns diagnostics including:
    - Per-dimension variance
    - Per-dimension mean
    - Effective dimensionality
    - Correlation between latent dimensions
    """
    print("\n" + "="*60)
    print("LATENT COLLAPSE ANALYSIS")
    print("="*60)

    device = torch.device(config.device)
    cvae.eval()

    # Extract latent representations for entire dataset (or subsample if too large)
    n_samples = min(10000, len(states))
    indices = torch.randperm(len(states))[:n_samples]
    sample_states = states[indices].to(device)
    sample_actions = actions[indices].to(device)

    with torch.no_grad():
        latent_means, latent_logvars = cvae.encode(sample_states, sample_actions)

    latent_means = latent_means.cpu().numpy()
    latent_logvars = latent_logvars.cpu().numpy()
    latent_stds = np.exp(0.5 * latent_logvars)

    # Compute statistics
    dim_means = latent_means.mean(axis=0)
    dim_stds = latent_means.std(axis=0)
    dim_variances = dim_stds ** 2

    # Effective dimensionality (based on variance distribution)
    # Higher = more dimensions are being used
    normalized_vars = dim_variances / (dim_variances.sum() + 1e-8)
    entropy = -np.sum(normalized_vars * np.log(normalized_vars + 1e-8))
    effective_dim = np.exp(entropy)

    # Correlation matrix
    correlation_matrix = np.corrcoef(latent_means.T)

    # Average absolute correlation (excluding diagonal)
    mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
    avg_abs_corr = np.abs(correlation_matrix[mask]).mean()

    # Identify collapsed dimensions (very low variance)
    collapse_threshold = 0.01
    collapsed_dims = np.where(dim_stds < collapse_threshold)[0]

    print(f"\nLatent Dimension Statistics:")
    print(f"  Total latent dims: {cvae.latent_dim}")
    print(f"  Action dims: {cvae.action_dim}")
    print(f"  Extra dims: {cvae.extra_latent_dim}")
    print(f"\nPer-dimension analysis:")
    for i in range(cvae.latent_dim):
        status = "⚠️ COLLAPSED" if i in collapsed_dims else "✓"
        extra_marker = " [EXTRA]" if i >= cvae.action_dim else ""
        print(f"  Dim {i}{extra_marker}: mean={dim_means[i]:.3f}, std={dim_stds[i]:.3f}, "
              f"posterior_std={latent_stds[:, i].mean():.3f} {status}")

    print(f"\nCollapse Metrics:")
    print(f"  Collapsed dimensions: {len(collapsed_dims)}/{cvae.latent_dim}")
    print(f"  Effective dimensionality: {effective_dim:.2f}/{cvae.latent_dim}")
    print(f"  Avg absolute correlation: {avg_abs_corr:.3f}")

    # Check specifically for extra dimension collapse
    extra_dim_indices = list(range(cvae.action_dim, cvae.latent_dim))
    extra_dim_stds = dim_stds[extra_dim_indices]
    extra_dim_collapsed = np.sum(extra_dim_stds < collapse_threshold)

    print(f"\nExtra Dimension Analysis:")
    print(f"  Extra dims collapsed: {extra_dim_collapsed}/{len(extra_dim_indices)}")
    print(f"  Extra dims mean std: {extra_dim_stds.mean():.3f}")

    if extra_dim_collapsed == len(extra_dim_indices):
        print("  ⚠️  WARNING: All extra dimensions have collapsed!")
    elif extra_dim_collapsed > 0:
        print(f"  ⚠️  WARNING: {extra_dim_collapsed} extra dimension(s) collapsed")
    else:
        print("  ✓ Extra dimensions are active")

    print("="*60)

    diagnostics = {
        'dim_means': dim_means,
        'dim_stds': dim_stds,
        'dim_variances': dim_variances,
        'effective_dim': effective_dim,
        'correlation_matrix': correlation_matrix,
        'avg_abs_corr': avg_abs_corr,
        'collapsed_dims': collapsed_dims,
        'extra_dim_collapsed': extra_dim_collapsed,
        'extra_dim_stds': extra_dim_stds
    }

    return diagnostics


def train_cvae(
    cvae: HigherDimLatentCVAE,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: ExperimentConfig
) -> HigherDimLatentCVAE:
    """Train the higher-dimensional latent CVAE with beta warmup to prevent collapse."""
    print("\n" + "="*60)
    print("Training Higher-Dimensional Latent CVAE")
    print("="*60)
    print(f"Beta warmup schedule: {config.cvae_beta_start:.4f} -> {config.cvae_beta_end:.4f} over {config.cvae_beta_warmup_epochs} epochs")

    device = torch.device(config.device)
    cvae = cvae.to(device)
    states = states.to(device)
    actions = actions.to(device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=config.cvae_lr)
    n_batches = len(states) // config.cvae_batch_size

    best_loss = float('inf')

    for epoch in tqdm(range(config.cvae_epochs), desc="CVAE Training"):
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}

        # Beta warmup schedule
        if epoch < config.cvae_beta_warmup_epochs:
            # Linear warmup
            beta = config.cvae_beta_start + (config.cvae_beta_end - config.cvae_beta_start) * (epoch / config.cvae_beta_warmup_epochs)
        else:
            beta = config.cvae_beta_end

        # Shuffle data
        indices = torch.randperm(len(states))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.cvae_batch_size
            end_idx = start_idx + config.cvae_batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]

            optimizer.zero_grad()
            loss_dict = cvae.compute_loss(batch_states, batch_actions, beta=beta)

            if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                print(f"\nWARNING: NaN/Inf at epoch {epoch}, batch {batch_idx}")
                continue

            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['recon'] += loss_dict['recon_loss'].item()
            epoch_losses['kl'] += loss_dict['kl_loss'].item()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.cvae_epochs}: "
                  f"Loss={epoch_losses['total']:.4f}, "
                  f"Recon={epoch_losses['recon']:.4f}, "
                  f"KL={epoch_losses['kl']:.4f}, "
                  f"Beta={beta:.4f}")

        if config.use_wandb:
            wandb.log({
                'cvae/total_loss': epoch_losses['total'],
                'cvae/recon_loss': epoch_losses['recon'],
                'cvae/kl_loss': epoch_losses['kl'],
                'cvae/beta': beta,
                'cvae/epoch': epoch + 1
            })

    print(f"\nCVAE Training Complete. Best Loss: {best_loss:.4f}")
    return cvae


def train_bc_standard(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_dim: int,
    action_dim: int,
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None
) -> BCPolicy:
    """Train standard BC on state-action pairs.

    Args:
        checkpoint_path: If provided, will try to load from this path.
                        If file exists, loads the checkpoint.
                        After training, saves to this path.
    """
    # Try to load existing checkpoint
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print("\n" + "="*60)
        print("Loading Existing Standard BC Checkpoint")
        print("="*60)
        print(f"Path: {checkpoint_path}")

        policy = BCPolicy(state_dim, action_dim, config.bc_hidden_dim,
                         config.bc_num_layers, config.bc_dropout).to(config.device)
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded Standard BC (trained for {checkpoint['epoch']} epochs)")
        print(f"  Best loss: {checkpoint['best_loss']:.4f}")
        print("="*60)
        return policy

    print("\n" + "="*60)
    print("Training Standard BC (state -> action)")
    print("="*60)

    device = torch.device(config.device)
    states = states.to(device)
    actions = actions.to(device)

    policy = BCPolicy(
        state_dim=state_dim,
        output_dim=action_dim,
        hidden_dim=config.bc_hidden_dim,
        num_layers=config.bc_num_layers,
        dropout=config.bc_dropout
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.bc_lr)
    n_batches = len(states) // config.bc_batch_size

    best_loss = float('inf')

    for epoch in tqdm(range(config.bc_epochs), desc="Standard BC Training"):
        epoch_loss = 0

        # Shuffle data
        indices = torch.randperm(len(states))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.bc_batch_size
            end_idx = start_idx + config.bc_batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]

            optimizer.zero_grad()
            pred_actions = policy(batch_states)
            loss = F.mse_loss(pred_actions, batch_actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.bc_epochs}: Loss={epoch_loss:.4f}")

        if config.use_wandb:
            wandb.log({
                'bc_standard/loss': epoch_loss,
                'bc_standard/epoch': epoch + 1
            })

    print(f"\nStandard BC Training Complete. Best Loss: {best_loss:.4f}")

    # Save checkpoint if path provided
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': policy.state_dict(),
            'epoch': config.bc_epochs,
            'best_loss': best_loss,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }, checkpoint_path)
        print(f"✓ Saved Standard BC checkpoint to: {checkpoint_path}")

    return policy


def train_bc_latent(
    states: torch.Tensor,
    actions: torch.Tensor,
    cvae: HigherDimLatentCVAE,
    state_dim: int,
    config: ExperimentConfig
) -> LatentBCPolicy:
    """Train BC on latent actions from CVAE."""
    print("\n" + "="*60)
    print("Training Latent BC (state -> latent_action -> action)")
    print("="*60)

    device = torch.device(config.device)
    states = states.to(device)
    actions = actions.to(device)
    cvae = cvae.to(device)
    cvae.eval()

    # Extract latent actions from CVAE
    print("Extracting latent actions from CVAE...")
    latent_actions = cvae.get_latent_action(states, actions)
    print(f"Latent actions shape: {latent_actions.shape}")

    # Create BC policy for latent space
    bc_policy = BCPolicy(
        state_dim=state_dim,
        output_dim=cvae.latent_dim,
        hidden_dim=config.bc_hidden_dim,
        num_layers=config.bc_num_layers,
        dropout=config.bc_dropout
    ).to(device)

    optimizer = torch.optim.Adam(bc_policy.parameters(), lr=config.bc_lr)
    n_batches = len(states) // config.bc_batch_size

    best_loss = float('inf')

    for epoch in tqdm(range(config.bc_epochs), desc="Latent BC Training"):
        epoch_loss = 0

        # Shuffle data
        indices = torch.randperm(len(states))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.bc_batch_size
            end_idx = start_idx + config.bc_batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_latent_actions = latent_actions[batch_indices]

            optimizer.zero_grad()
            pred_latent_actions = bc_policy(batch_states)
            loss = F.mse_loss(pred_latent_actions, batch_latent_actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.bc_epochs}: Loss={epoch_loss:.4f}")

        if config.use_wandb:
            wandb.log({
                'bc_latent/loss': epoch_loss,
                'bc_latent/epoch': epoch + 1
            })

    print(f"\nLatent BC Training Complete. Best Loss: {best_loss:.4f}")

    # Create combined policy
    combined_policy = LatentBCPolicy(bc_policy, cvae)
    return combined_policy


def evaluate_policy(
    policy: nn.Module,
    env_name: str,
    n_episodes: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: str = 'cuda'
) -> Dict:
    """Evaluate a policy."""
    # Adroit environments have different naming
    adroit_envs = ['pen', 'hammer', 'door', 'relocate']

    if env_name in adroit_envs:
        # Adroit: pen-v0, hammer-v0, etc.
        base_env_name = f"{env_name}-v0"
    else:
        # MuJoCo: HalfCheetah-v3, Hopper-v3, etc.
        env_name_map = {
            'halfcheetah': 'HalfCheetah',
            'hopper': 'Hopper',
            'walker2d': 'Walker2d',
            'ant': 'Ant',
            'humanoid': 'Humanoid'
        }
        base_env_name = env_name_map.get(env_name, env_name.title()) + "-v3"

    env = gym.make(base_env_name)
    policy.eval()

    episode_returns = []
    episode_lengths = []

    for episode in tqdm(range(n_episodes), desc="Evaluating", leave=False):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        episode_return = 0
        episode_length = 0
        done = False

        while not done:
            # Normalize state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            state_tensor = (state_tensor - state_mean.to(device)) / state_std.to(device)

            # Get action
            with torch.no_grad():
                action = policy(state_tensor)

            # Denormalize action
            action = action * action_std.to(device) + action_mean.to(device)
            action = action.cpu().numpy().squeeze()

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result

            episode_return += reward
            episode_length += 1

            if episode_length >= 1000:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    env.close()

    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episodes': episode_returns
    }

    return results


def load_cvae_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[HigherDimLatentCVAE, Dict]:
    """
    Load a trained CVAE from checkpoint.

    Returns:
        cvae: Loaded CVAE model
        checkpoint_data: Dictionary containing normalization stats and other info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cvae = HigherDimLatentCVAE(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        extra_latent_dim=checkpoint['extra_latent_dim'],
        hidden_dim=256,  # Assuming default, could also save this in checkpoint
        num_layers=3,
        dropout=0.1
    ).to(device)

    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()

    print(f"✓ Loaded CVAE from {checkpoint_path}")
    print(f"  State dim: {checkpoint['state_dim']}, Action dim: {checkpoint['action_dim']}")
    print(f"  Latent dim: {checkpoint['action_dim'] + checkpoint['extra_latent_dim']}")

    return cvae, checkpoint


def main():
    """Main experiment function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Higher-Dimensional Latent CVAE for BC')
    parser.add_argument('--env', type=str, default='hopper',
                        choices=['halfcheetah', 'hopper', 'walker2d', 'ant', 'humanoid',
                                 'pen', 'hammer', 'door', 'relocate'],
                        help='Environment to test on (default: hopper)')
    parser.add_argument('--dataset', type=str, default='medium-expert',
                        choices=['medium', 'medium-expert', 'expert', 'medium-replay',
                                 'random', 'human', 'cloned'],
                        help='D4RL dataset type (default: medium-expert)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--latent-extra', type=int, default=1,
                        help='Extra latent dimensions beyond action_dim (default: 1)')
    parser.add_argument('--cvae-epochs', type=int, default=50,
                        help='Number of CVAE training epochs (default: 50)')
    parser.add_argument('--bc-epochs', type=int, default=100,
                        help='Number of BC training epochs (default: 100)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--beta-start', type=float, default=0.0,
                        help='Starting beta (KL weight) for warmup (default: 0.0)')
    parser.add_argument('--beta-end', type=float, default=0.01,
                        help='Final beta (KL weight) after warmup (default: 0.01)')
    parser.add_argument('--beta-warmup', type=int, default=30,
                        help='Number of epochs for beta warmup (default: 30)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility (default: 0)')

    # Mode selection arguments
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'collapse-only', 'bc-only', 'eval-only'],
                        help='Run mode: full (train CVAE+BC+eval), collapse-only (analyze existing CVAE), '
                             'bc-only (train BC with existing CVAE), eval-only (evaluate existing policies)')
    parser.add_argument('--cvae-checkpoint', type=str, default=None,
                        help='Path to CVAE checkpoint (required for collapse-only, bc-only, eval-only modes)')
    parser.add_argument('--bc-standard-checkpoint', type=str, default=None,
                        help='Path to standard BC checkpoint (for eval-only mode)')
    parser.add_argument('--bc-latent-checkpoint', type=str, default=None,
                        help='Path to latent BC checkpoint (for eval-only mode)')

    args = parser.parse_args()

    # Validate mode requirements
    if args.mode in ['collapse-only', 'bc-only', 'eval-only']:
        if args.cvae_checkpoint is None:
            parser.error(f"--cvae-checkpoint is required for --mode {args.mode}")

    if args.mode == 'eval-only':
        if args.bc_standard_checkpoint is None or args.bc_latent_checkpoint is None:
            parser.error("--bc-standard-checkpoint and --bc-latent-checkpoint are required for --mode eval-only")

    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {args.seed}")

    # Create config with command line args
    config = ExperimentConfig()
    config.env_name = args.env
    config.dataset_name = args.dataset
    config.use_wandb = not args.no_wandb
    config.cvae_latent_dim_extra = args.latent_extra
    config.cvae_epochs = args.cvae_epochs
    config.bc_epochs = args.bc_epochs
    config.eval_episodes = args.eval_episodes
    config.cvae_beta_start = args.beta_start
    config.cvae_beta_end = args.beta_end
    config.cvae_beta_warmup_epochs = args.beta_warmup

    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.save_dir, f"{config.env_name}_{args.mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Mode: {args.mode}")

    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(
            project="latent-cvae-bc-comparison",
            entity="swami2004",
            name=f"{config.experiment_name}_{config.env_name}",
            config=vars(config)
        )

    print("\n" + "="*60)
    print("HYPOTHESIS TEST: Higher-Dimensional Latent CVAE for BC")
    print("="*60)
    print(f"Environment: {config.env_name}-{config.dataset_name}")
    print(f"Device: {config.device}")
    print(f"Latent dim will be: action_dim + {config.cvae_latent_dim_extra}")

    # Load dataset
    print("\nLoading dataset...")
    data = load_d4rl_dataset(config.env_name, config.dataset_name)

    # =========================================================================
    # MODE: COLLAPSE-ONLY - Just analyze an existing CVAE
    # =========================================================================
    if args.mode == 'collapse-only':
        print("\n" + "="*60)
        print("MODE: COLLAPSE ANALYSIS ONLY")
        print("="*60)

        cvae, checkpoint = load_cvae_checkpoint(args.cvae_checkpoint, config.device)
        collapse_diagnostics = analyze_latent_collapse(cvae, data['states'], data['actions'], config)

        # Save analysis results
        analysis_path = os.path.join(run_dir, f"collapse_analysis_{config.env_name}.txt")
        with open(analysis_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LATENT COLLAPSE ANALYSIS\n")
            f.write("="*60 + "\n")
            f.write(f"CVAE Checkpoint: {args.cvae_checkpoint}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Environment: {config.env_name}-{config.dataset_name}\n")
            f.write(f"Latent Dim: {cvae.latent_dim} (action_dim={cvae.action_dim} + {cvae.extra_latent_dim})\n")
            f.write("\n")
            f.write(f"Effective dimensionality: {collapse_diagnostics['effective_dim']:.2f}/{cvae.latent_dim}\n")
            f.write(f"Collapsed dimensions: {len(collapse_diagnostics['collapsed_dims'])}/{cvae.latent_dim}\n")
            f.write(f"Extra dims collapsed: {collapse_diagnostics['extra_dim_collapsed']}/{cvae.extra_latent_dim}\n")
            f.write(f"Avg correlation: {collapse_diagnostics['avg_abs_corr']:.3f}\n")
            f.write("\n")
            f.write("Per-dimension statistics:\n")
            for i in range(cvae.latent_dim):
                extra_marker = " [EXTRA]" if i >= cvae.action_dim else ""
                f.write(f"  Dim {i}{extra_marker}: mean={collapse_diagnostics['dim_means'][i]:.3f}, "
                       f"std={collapse_diagnostics['dim_stds'][i]:.3f}\n")

        print(f"\nAnalysis saved to {analysis_path}")
        print("\n" + "="*60)
        print("COLLAPSE ANALYSIS COMPLETE")
        print("="*60)
        return

    # =========================================================================
    # CVAE Training (full or bc-only modes)
    # =========================================================================
    if args.mode == 'full':
        # Initialize and train CVAE
        cvae = HigherDimLatentCVAE(
            state_dim=data['state_dim'],
            action_dim=data['action_dim'],
            extra_latent_dim=config.cvae_latent_dim_extra,
            hidden_dim=config.cvae_hidden_dim,
            num_layers=config.cvae_num_layers,
            dropout=config.cvae_dropout
        )
        cvae = train_cvae(cvae, data['states'], data['actions'], config)
        collapse_diagnostics = analyze_latent_collapse(cvae, data['states'], data['actions'], config)
    elif args.mode == 'bc-only':
        # Load existing CVAE
        print("\n" + "="*60)
        print("MODE: BC TRAINING ONLY")
        print("="*60)
        cvae, checkpoint = load_cvae_checkpoint(args.cvae_checkpoint, config.device)

        # Use checkpoint's normalization stats
        data['state_mean'] = checkpoint['state_mean']
        data['state_std'] = checkpoint['state_std']
        data['action_mean'] = checkpoint['action_mean']
        data['action_std'] = checkpoint['action_std']

        # Optional: analyze collapse
        collapse_diagnostics = analyze_latent_collapse(cvae, data['states'], data['actions'], config)

    # Log collapse metrics to wandb
    if config.use_wandb and args.mode != 'eval-only':
        wandb.log({
            'collapse/effective_dim': collapse_diagnostics['effective_dim'],
            'collapse/num_collapsed': len(collapse_diagnostics['collapsed_dims']),
            'collapse/extra_dim_collapsed': collapse_diagnostics['extra_dim_collapsed'],
            'collapse/avg_abs_corr': collapse_diagnostics['avg_abs_corr'],
            'collapse/extra_dim_mean_std': collapse_diagnostics['extra_dim_stds'].mean()
        })

        # Log per-dimension stats
        for i in range(cvae.latent_dim):
            wandb.log({
                f'collapse/dim_{i}_std': collapse_diagnostics['dim_stds'][i],
                f'collapse/dim_{i}_mean': collapse_diagnostics['dim_means'][i]
            })

    # Save CVAE (only in full mode)
    if args.mode == 'full':
        cvae_path = os.path.join(run_dir, f"cvae_{config.env_name}.pt")
        torch.save({
            'model_state_dict': cvae.state_dict(),
            'state_dim': data['state_dim'],
            'action_dim': data['action_dim'],
            'extra_latent_dim': config.cvae_latent_dim_extra,
            'state_mean': data['state_mean'],
            'state_std': data['state_std'],
            'action_mean': data['action_mean'],
            'action_std': data['action_std'],
            'collapse_diagnostics': collapse_diagnostics,
            'timestamp': timestamp
        }, cvae_path)
        print(f"\nSaved CVAE to {cvae_path}")

    # =========================================================================
    # BC Training (full or bc-only modes)
    # =========================================================================
    if args.mode in ['full', 'bc-only']:
        # Create checkpoint path for Standard BC (seed-specific, env-specific)
        bc_checkpoint_dir = os.path.join(config.save_dir, "bc_checkpoints")
        bc_standard_checkpoint = os.path.join(
            bc_checkpoint_dir,
            f"standard_bc_{config.env_name}_{config.dataset_name}_seed{args.seed}.pt"
        )

        # Train Standard BC (or load if already trained with this seed)
        bc_standard = train_bc_standard(
            data['states'],
            data['actions'],
            data['state_dim'],
            data['action_dim'],
            config,
            checkpoint_path=bc_standard_checkpoint
        )

        # Train Latent BC
        bc_latent = train_bc_latent(
            data['states'],
            data['actions'],
            cvae,
            data['state_dim'],
            config
        )

        # Save BC policies
        bc_standard_path = os.path.join(run_dir, f"bc_standard_{config.env_name}.pt")
        torch.save({
            'model_state_dict': bc_standard.state_dict(),
            'state_dim': data['state_dim'],
            'action_dim': data['action_dim'],
            'timestamp': timestamp
        }, bc_standard_path)
        print(f"\nSaved Standard BC to {bc_standard_path}")

        bc_latent_path = os.path.join(run_dir, f"bc_latent_{config.env_name}.pt")
        torch.save({
            'bc_policy_state_dict': bc_latent.bc_policy.state_dict(),
            'cvae_checkpoint': args.cvae_checkpoint if args.mode == 'bc-only' else cvae_path,
            'state_dim': data['state_dim'],
            'latent_dim': cvae.latent_dim,
            'timestamp': timestamp
        }, bc_latent_path)
        print(f"\nSaved Latent BC to {bc_latent_path}")

    elif args.mode == 'eval-only':
        # Load BC policies from checkpoints
        print("\n" + "="*60)
        print("MODE: EVALUATION ONLY")
        print("="*60)

        cvae, checkpoint = load_cvae_checkpoint(args.cvae_checkpoint, config.device)

        # Load standard BC
        bc_checkpoint = torch.load(args.bc_standard_checkpoint, map_location=config.device)
        bc_standard = BCPolicy(
            state_dim=bc_checkpoint['state_dim'],
            output_dim=bc_checkpoint['action_dim'],
            hidden_dim=config.bc_hidden_dim,
            num_layers=config.bc_num_layers,
            dropout=config.bc_dropout
        ).to(config.device)
        bc_standard.load_state_dict(bc_checkpoint['model_state_dict'])
        print(f"✓ Loaded Standard BC from {args.bc_standard_checkpoint}")

        # Load latent BC
        bc_latent_checkpoint = torch.load(args.bc_latent_checkpoint, map_location=config.device)
        bc_policy = BCPolicy(
            state_dim=bc_latent_checkpoint['state_dim'],
            output_dim=bc_latent_checkpoint['latent_dim'],
            hidden_dim=config.bc_hidden_dim,
            num_layers=config.bc_num_layers,
            dropout=config.bc_dropout
        ).to(config.device)
        bc_policy.load_state_dict(bc_latent_checkpoint['bc_policy_state_dict'])
        bc_latent = LatentBCPolicy(bc_policy, cvae)
        print(f"✓ Loaded Latent BC from {args.bc_latent_checkpoint}")

        # Use checkpoint normalization stats
        data['state_mean'] = checkpoint['state_mean']
        data['state_std'] = checkpoint['state_std']
        data['action_mean'] = checkpoint['action_mean']
        data['action_std'] = checkpoint['action_std']

    # =========================================================================
    # Evaluation
    # =========================================================================
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    print("\nEvaluating Standard BC...")
    results_standard = evaluate_policy(
        bc_standard,
        config.env_name,
        config.eval_episodes,
        data['state_mean'],
        data['state_std'],
        data['action_mean'],
        data['action_std'],
        config.device
    )

    print("\nEvaluating Latent BC...")
    results_latent = evaluate_policy(
        bc_latent,
        config.env_name,
        config.eval_episodes,
        data['state_mean'],
        data['state_std'],
        data['action_mean'],
        data['action_std'],
        config.device
    )

    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"\nStandard BC (state -> action):")
    print(f"  Mean Return: {results_standard['mean_return']:.2f} ± {results_standard['std_return']:.2f}")
    print(f"  Mean Length: {results_standard['mean_length']:.1f}")

    print(f"\nLatent BC (state -> latent_action -> action):")
    print(f"  Mean Return: {results_latent['mean_return']:.2f} ± {results_latent['std_return']:.2f}")
    print(f"  Mean Length: {results_latent['mean_length']:.1f}")

    improvement = ((results_latent['mean_return'] - results_standard['mean_return'])
                   / abs(results_standard['mean_return']) * 100)
    print(f"\nImprovement: {improvement:+.2f}%")

    if config.use_wandb:
        wandb.log({
            'final/standard_bc_return': results_standard['mean_return'],
            'final/standard_bc_std': results_standard['std_return'],
            'final/latent_bc_return': results_latent['mean_return'],
            'final/latent_bc_std': results_latent['std_return'],
            'final/improvement_pct': improvement
        })

    # Save results
    results_path = os.path.join(run_dir, f"results_{config.env_name}.txt")
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("HYPOTHESIS TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Environment: {config.env_name}-{config.dataset_name}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Eval Episodes: {config.eval_episodes}\n")
        f.write(f"CVAE Latent Dim: {cvae.latent_dim} (action_dim={data['action_dim']} + {config.cvae_latent_dim_extra})\n")
        f.write("\n")
        # Only write collapse analysis if it was computed
        if args.mode != 'eval-only':
            f.write("COLLAPSE ANALYSIS:\n")
            f.write(f"  Effective dimensionality: {collapse_diagnostics['effective_dim']:.2f}/{cvae.latent_dim}\n")
            f.write(f"  Collapsed dimensions: {len(collapse_diagnostics['collapsed_dims'])}/{cvae.latent_dim}\n")
            f.write(f"  Extra dims collapsed: {collapse_diagnostics['extra_dim_collapsed']}/{config.cvae_latent_dim_extra}\n")
            f.write(f"  Avg correlation: {collapse_diagnostics['avg_abs_corr']:.3f}\n")
            f.write("\n")
        f.write("PERFORMANCE:\n")
        f.write(f"  Standard BC: {results_standard['mean_return']:.2f} ± {results_standard['std_return']:.2f}\n")
        f.write(f"  Latent BC:   {results_latent['mean_return']:.2f} ± {results_latent['std_return']:.2f}\n")
        f.write(f"  Improvement: {improvement:+.2f}%\n")

    print(f"\nResults saved to {results_path}")

    if config.use_wandb:
        wandb.finish()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
