"""
Simplified CVAE + INAC Pipeline for MuJoCo Testing

This script implements a simplified version of the CVAE progress reward + INAC pipeline
for testing on simple MuJoCo environments. It removes the complexity of residual learning
and furniture-specific processing to focus on the core inverse RL methodology.

Key components:
1. Simple CVAE that learns task progress from demonstrations
2. INAC agent trained with CVAE-derived rewards
3. Direct evaluation on MuJoCo environments

Test environments: HalfCheetah, Hopper, Walker2d, Ant
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
import gym  # Use old gym for D4RL compatibility
import d4rl
import pickle
from tqdm import tqdm
import wandb

# Add INAC to path
sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')
from core.agent.in_sample import InSampleAC
from core.utils import torch_utils, logger


class SimpleCVAE(nn.Module):
    """
    Simplified CVAE for learning task progress from state-action trajectories.
    Unlike the furniture-specific version, this works with any MuJoCo environment.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: (state, action) -> progress parameters
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

        # Progress latent parameters
        self.progress_mean = nn.Linear(hidden_dim, latent_dim)
        self.progress_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (state, progress) -> action
        decoder_layers = []
        decoder_input_dim = state_dim + latent_dim

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

        # Progress predictor: state -> progress (for rewards)
        predictor_layers = []
        predictor_layers.extend([
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(num_layers - 2):
            predictor_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        predictor_layers.extend([
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()  # Progress between 0 and 1
        ])
        self.progress_predictor = nn.Sequential(*predictor_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = torch.sigmoid(self.progress_mean(h))  # Constrain to [0, 1]
        logvar = self.progress_logvar(h)
        logvar = torch.clamp(logvar, -10, 10)  # Prevent explosion
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            progress = mean + eps * std
            progress = torch.clamp(progress, 0.0, 1.0)
            return progress
        else:
            return mean

    def decode(self, state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, progress], dim=-1)
        action = self.decoder(x)
        return action

    def predict_progress(self, state: torch.Tensor) -> torch.Tensor:
        """Predict progress from state alone - this will be used as reward signal."""
        return self.progress_predictor(state)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        progress_mean, progress_logvar = self.encode(state, action)
        progress = self.reparameterize(progress_mean, progress_logvar)
        recon_action = self.decode(state, progress)
        predicted_progress = self.predict_progress(state)

        return {
            'recon_action': recon_action,
            'progress_mean': progress_mean,
            'progress_logvar': progress_logvar,
            'progress': progress,
            'predicted_progress': predicted_progress
        }

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        true_progress: torch.Tensor,
        beta: float = 0.01,
        gamma: float = 1.0,
        use_recon_loss: bool = True,
        use_progress_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CVAE loss with optional ablations.

        Args:
            use_recon_loss: If False, removes reconstruction loss (tests if action reconstruction is needed)
            use_progress_loss: If False, removes progress supervision/prediction losses
                              (tests if latent encoding alone is sufficient)
        """
        outputs = self.forward(state, action)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['recon_action'], action)

        # KL divergence loss (standard VAE formulation)
        progress_mean = outputs['progress_mean']
        progress_logvar = outputs['progress_logvar']
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        # For progress in [0, 1], we compare to N(0.5, small_var) instead
        kl_loss = -0.5 * torch.mean(1 + progress_logvar - (progress_mean - 0.5).pow(2) - progress_logvar.exp())
        kl_loss = torch.clamp(kl_loss, 0, 100)  # Prevent explosion

        # Progress supervision loss
        progress_supervision_loss = F.mse_loss(outputs['progress_mean'], true_progress)

        # Progress prediction loss
        progress_prediction_loss = F.mse_loss(outputs['predicted_progress'], true_progress)

        # Build total loss based on ablation flags
        total_loss = beta * kl_loss  # Always keep KL loss for VAE regularization

        if use_recon_loss:
            total_loss = total_loss + recon_loss

        if use_progress_loss:
            total_loss = total_loss + gamma * progress_supervision_loss + gamma * progress_prediction_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'progress_supervision_loss': progress_supervision_loss,
            'progress_prediction_loss': progress_prediction_loss
        }


class CVAERewardComputer:
    """Computes CVAE-based progress rewards for RL training."""

    def __init__(self, cvae_model: SimpleCVAE, device: str = 'cuda'):
        self.cvae = cvae_model
        self.device = device
        self.cvae.eval()

    def compute_progress_reward(self, state: np.ndarray) -> float:
        """Compute progress reward from state using trained CVAE."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Normalize state using training statistics
        if hasattr(self.cvae, 'state_mean'):
            state_tensor = (state_tensor - self.cvae.state_mean) / self.cvae.state_std

        with torch.no_grad():
            progress_pred = self.cvae.predict_progress(state_tensor)
            reward = progress_pred.cpu().numpy()[0, 0]

        return float(reward)


def load_cvae_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> SimpleCVAE:
    """
    Load a trained CVAE model from checkpoint.

    Args:
        checkpoint_path: Path to the saved .pt file
        device: Device to load the model on

    Returns:
        Loaded CVAE model ready for inference

    Example:
        cvae = load_cvae_from_checkpoint('./cvae_checkpoints/halfcheetah/cvae_model.pt')
        reward_computer = CVAERewardComputer(cvae, device='cuda')
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with saved architecture
    cvae = SimpleCVAE(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dim=checkpoint['hidden_dim']
    ).to(device)

    # Load weights
    cvae.load_state_dict(checkpoint['model_state_dict'])

    # Load normalization stats if available
    if 'state_mean' in checkpoint:
        cvae.state_mean = checkpoint['state_mean'].to(device)
        cvae.state_std = checkpoint['state_std'].to(device)
        cvae.action_mean = checkpoint['action_mean'].to(device)
        cvae.action_std = checkpoint['action_std'].to(device)
        print(f"✓ Loaded normalization stats")

    cvae.eval()
    print(f"✓ Loaded CVAE from {checkpoint_path}")

    return cvae


class SimpleMuJoCoDataset:
    """
    Simplified dataset class for MuJoCo environments using D4RL.
    Creates progress labels and prepares data for CVAE + INAC training.
    """

    def __init__(self, env_name: str, dataset_name: str = 'medium-expert'):
        self.env_name = env_name
        self.dataset_name = dataset_name

        # Load D4RL dataset - try different naming conventions
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
            raise ValueError(f"Could not find environment for {env_name}-{dataset_name}. "
                           f"Tried: {env_id_options}")

        dataset = env.get_dataset()

        self.states = dataset['observations'].astype(np.float32)
        self.actions = dataset['actions'].astype(np.float32)
        self.rewards = dataset['rewards'].astype(np.float32)
        self.dones = dataset['terminals'].astype(bool)

        # D4RL also has 'timeouts' which should be treated as episode boundaries
        if 'timeouts' in dataset:
            self.timeouts = dataset['timeouts'].astype(bool)
            # Combine terminals and timeouts for episode boundaries
            self.dones = self.dones | self.timeouts

        # Find episode boundaries
        self.episode_starts = [0]
        self.episode_ends = []

        for i, done in enumerate(self.dones):
            if done:
                self.episode_ends.append(i + 1)
                if i + 1 < len(self.states):
                    self.episode_starts.append(i + 1)

        # Ensure we have complete episodes
        if len(self.episode_ends) != len(self.episode_starts):
            self.episode_ends.append(len(self.states))

        self.n_episodes = len(self.episode_ends)
        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        print(f"Loaded {env_name}-{dataset_name} dataset:")
        print(f"  States: {self.states.shape}")
        print(f"  Actions: {self.actions.shape}")
        print(f"  Episodes: {self.n_episodes}")
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")

        # Create progress labels (0 to 1 within each episode)
        self.progress_labels = self._create_progress_labels()

    def _create_progress_labels(self) -> np.ndarray:
        """Create progress labels based on cumulative reward normalized per episode."""
        progress_labels = np.zeros(len(self.states))

        for start, end in zip(self.episode_starts, self.episode_ends):
            episode_rewards = self.rewards[start:end]

            # Compute cumulative return normalized to [0, 1]
            cumulative_rewards = np.cumsum(episode_rewards)

            # Normalize to [0, 1] based on total episode return
            total_return = cumulative_rewards[-1]
            if abs(total_return) > 1e-6:
                # For negative returns, flip the sign
                if total_return < 0:
                    normalized_progress = 1.0 - (cumulative_rewards / total_return)
                else:
                    normalized_progress = cumulative_rewards / total_return
                # Clip to [0, 1]
                normalized_progress = np.clip(normalized_progress, 0.0, 1.0)
            else:
                # Fallback to timestep-based if no reward variation
                episode_length = end - start
                normalized_progress = np.linspace(0, 1, episode_length, endpoint=False)

            progress_labels[start:end] = normalized_progress

        print(f"  Progress labels: min={progress_labels.min():.3f}, max={progress_labels.max():.3f}, mean={progress_labels.mean():.3f}")
        return progress_labels.astype(np.float32)

    def get_train_data(self, subset_fraction: float = 1.0):
        """Get training data for CVAE."""
        if subset_fraction < 1.0:
            # Use contiguous samples to preserve temporal structure
            n_samples = int(len(self.states) * subset_fraction)
            # Take evenly spaced samples across the dataset
            indices = np.linspace(0, len(self.states) - 1, n_samples, dtype=int)
            return {
                'states': self.states[indices],
                'actions': self.actions[indices],
                'progress': self.progress_labels[indices].reshape(-1, 1)
            }
        else:
            return {
                'states': self.states,
                'actions': self.actions,
                'progress': self.progress_labels.reshape(-1, 1)
            }

    def get_inac_data(self, cvae_reward_computer: CVAERewardComputer, cache_path: str = None) -> Dict:
        """Convert to INAC format with CVAE-based rewards."""

        # Try to load cached rewards
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached CVAE rewards from {cache_path}")
            cached_data = np.load(cache_path)
            cvae_rewards = cached_data['cvae_rewards']
            print(f"✓ Loaded {len(cvae_rewards)} cached rewards")
        else:
            print("Computing CVAE progress rewards...")
            cvae_rewards = np.zeros(len(self.states))
            for i in tqdm(range(len(self.states)), desc="Computing rewards"):
                cvae_rewards[i] = cvae_reward_computer.compute_progress_reward(self.states[i])

            # Scale rewards
            cvae_rewards = cvae_rewards * 10.0  # Scale 0-1 to 0-10

            # Save to cache if path provided
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(cache_path, cvae_rewards=cvae_rewards)
                print(f"✓ Saved rewards to cache: {cache_path}")

        print(f"CVAE rewards - min: {cvae_rewards.min():.3f}, max: {cvae_rewards.max():.3f}, mean: {cvae_rewards.mean():.3f}")
        print(f"True env rewards - min: {self.rewards.min():.3f}, max: {self.rewards.max():.3f}, mean: {self.rewards.mean():.3f}")

        # Create next states and terminals
        next_states = np.roll(self.states, -1, axis=0)
        terminals = np.zeros(len(self.states), dtype=bool)

        # Handle episode boundaries
        for start, end in zip(self.episode_starts, self.episode_ends):
            if end - 1 < len(terminals):
                terminals[end - 1] = True
                next_states[end - 1] = self.states[end - 1]  # Terminal state points to itself

        return {
            'env': {
                'states': self.states,
                'actions': self.actions,
                'rewards': cvae_rewards,  # CVAE rewards for training
                'true_rewards': self.rewards,  # Original env rewards for logging
                'next_states': next_states,
                'terminations': terminals
            }
        }


def train_cvae(
    dataset: SimpleMuJoCoDataset,
    hidden_dim: int = 256,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    use_wandb: bool = False,
    beta: float = 0.01,
    gamma: float = 1.0,
    use_recon_loss: bool = True,
    use_progress_loss: bool = True
) -> SimpleCVAE:
    """
    Train CVAE on the dataset with optional ablations.

    Args:
        beta: Weight for KL divergence loss
        gamma: Weight for progress losses
        use_recon_loss: If False, removes reconstruction loss (ablation study)
        use_progress_loss: If False, removes progress supervision (tests latent-only rewards)
    """

    train_data = dataset.get_train_data()
    states = torch.FloatTensor(train_data['states']).to(device)
    actions = torch.FloatTensor(train_data['actions']).to(device)
    progress = torch.FloatTensor(train_data['progress']).to(device)

    # Normalize states and actions for numerical stability
    state_mean = states.mean(dim=0, keepdim=True)
    state_std = states.std(dim=0, keepdim=True) + 1e-6
    states = (states - state_mean) / state_std

    action_mean = actions.mean(dim=0, keepdim=True)
    action_std = actions.std(dim=0, keepdim=True) + 1e-6
    actions = (actions - action_mean) / action_std

    # Print diagnostics
    print(f"\nCVAE Training Data Statistics:")
    print(f"  States shape: {states.shape}")
    print(f"  States (normalized): mean={states.mean():.3f}, std={states.std():.3f}")
    print(f"  Actions (normalized): mean={actions.mean():.3f}, std={actions.std():.3f}")
    print(f"  Progress range: [{progress.min():.3f}, {progress.max():.3f}]")
    print(f"  Progress mean: {progress.mean():.3f}, std: {progress.std():.3f}")
    print(f"  Progress unique values: {len(torch.unique(progress))}")

    # Initialize model
    cvae = SimpleCVAE(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=learning_rate)

    print(f"\n{'='*50}")
    print("CVAE Training Configuration:")
    print(f"{'='*50}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"  Beta (KL weight): {beta}")
    print(f"  Gamma (Progress weight): {gamma}")
    print(f"  Use reconstruction loss: {use_recon_loss}")
    print(f"  Use progress supervision: {use_progress_loss}")
    print(f"{'='*50}\n")

    n_batches = len(states) // batch_size
    global_step = 0

    # Store normalization stats for later use
    cvae.state_mean = state_mean
    cvae.state_std = state_std
    cvae.action_mean = action_mean
    cvae.action_std = action_std

    for epoch in tqdm(range(num_epochs), desc="CVAE Epochs"):
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'progress_sup': 0, 'progress_pred': 0}

        # Shuffle data
        indices = torch.randperm(len(states))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_progress = progress[batch_indices]

            optimizer.zero_grad()

            loss_dict = cvae.compute_loss(
                batch_states, batch_actions, batch_progress,
                beta=beta, gamma=gamma,
                use_recon_loss=use_recon_loss,
                use_progress_loss=use_progress_loss
            )

            # Check for NaN in loss
            if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                print(f"\nWARNING: NaN/Inf detected at epoch {epoch}, batch {batch_idx}")
                print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
                print(f"  KL loss: {loss_dict['kl_loss'].item():.4f}")
                print(f"  Prog sup loss: {loss_dict['progress_supervision_loss'].item():.4f}")
                print(f"  Prog pred loss: {loss_dict['progress_prediction_loss'].item():.4f}")
                print("  Skipping this batch...")
                continue

            loss_dict['total_loss'].backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)

            # Compute gradient norm for monitoring
            total_norm = 0.0
            for p in cvae.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key == 'total':
                    epoch_losses[key] += loss_dict['total_loss'].item()
                elif key == 'recon':
                    epoch_losses[key] += loss_dict['recon_loss'].item()
                elif key == 'kl':
                    epoch_losses[key] += loss_dict['kl_loss'].item()
                elif key == 'progress_sup':
                    epoch_losses[key] += loss_dict['progress_supervision_loss'].item()
                elif key == 'progress_pred':
                    epoch_losses[key] += loss_dict['progress_prediction_loss'].item()

            # Log every batch to wandb for detailed monitoring
            if use_wandb and batch_idx % 10 == 0:
                # Get model predictions for statistics
                with torch.no_grad():
                    outputs = cvae.forward(batch_states, batch_actions)
                    pred_progress = outputs['predicted_progress'].cpu().numpy()
                    true_progress_np = batch_progress.cpu().numpy()

                    # Calculate correlation
                    if len(pred_progress) > 1:
                        correlation = np.corrcoef(pred_progress.flatten(), true_progress_np.flatten())[0, 1]
                    else:
                        correlation = 0.0

                wandb.log({
                    'cvae_batch/total_loss': loss_dict['total_loss'].item(),
                    'cvae_batch/recon_loss': loss_dict['recon_loss'].item(),
                    'cvae_batch/kl_loss': loss_dict['kl_loss'].item(),
                    'cvae_batch/progress_supervision_loss': loss_dict['progress_supervision_loss'].item(),
                    'cvae_batch/progress_prediction_loss': loss_dict['progress_prediction_loss'].item(),
                    'cvae_batch/gradient_norm': total_norm,
                    'cvae_batch/progress_correlation': correlation,
                    'cvae_batch/predicted_progress_mean': np.mean(pred_progress),
                    'cvae_batch/predicted_progress_std': np.std(pred_progress),
                }, step=global_step)

            global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # Evaluate on full dataset for epoch metrics
        with torch.no_grad():
            cvae.eval()
            outputs = cvae.forward(states[:10000], actions[:10000])  # Sample for efficiency
            pred_progress = outputs['predicted_progress'].cpu().numpy()
            true_progress_np = progress[:10000].cpu().numpy()
            correlation = np.corrcoef(pred_progress.flatten(), true_progress_np.flatten())[0, 1]
            cvae.train()

        # Print every epoch for better visibility
        print(f"\rEpoch {epoch+1}/{num_epochs}: "
              f"Loss={epoch_losses['total']:.4f}, "
              f"Recon={epoch_losses['recon']:.4f}, "
              f"Progress={epoch_losses['progress_pred']:.4f}, "
              f"Corr={correlation:.3f}", end="" if epoch < num_epochs - 1 else "\n")

        # Log epoch-level metrics to wandb
        if use_wandb:
            wandb.log({
                'cvae_epoch/total_loss': epoch_losses['total'],
                'cvae_epoch/recon_loss': epoch_losses['recon'],
                'cvae_epoch/kl_loss': epoch_losses['kl'],
                'cvae_epoch/progress_supervision_loss': epoch_losses['progress_sup'],
                'cvae_epoch/progress_prediction_loss': epoch_losses['progress_pred'],
                'cvae_epoch/progress_correlation': correlation,
                'cvae_epoch/epoch': epoch + 1,
                'cvae_epoch/learning_rate': optimizer.param_groups[0]['lr']
            }, step=global_step)

    return cvae


def train_inac_with_cvae_rewards(
    dataset: SimpleMuJoCoDataset,
    cvae_model: SimpleCVAE,
    max_steps: int = 50000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    tau: float = 0.1,
    device: str = 'cuda',
    use_wandb: bool = False,
    log_interval: int = 1000,
    eval_interval: int = 5000,
    eval_episodes: int = 5,
    env_name: str = "halfcheetah",
    rewards_cache_path: str = None
) -> InSampleAC:
    """Train INAC agent with CVAE-based rewards."""

    # Set up reward computer
    reward_computer = CVAERewardComputer(cvae_model, device)

    # Get INAC-formatted data with caching
    offline_data = dataset.get_inac_data(reward_computer, cache_path=rewards_cache_path)

    # Compute true episode returns for reference
    true_episode_returns = []
    for start, end in zip(dataset.episode_starts, dataset.episode_ends):
        episode_return = dataset.rewards[start:end].sum()
        true_episode_returns.append(episode_return)
    true_episode_returns = np.array(true_episode_returns)
    print(f"\nTrue episode returns from dataset:")
    print(f"  Mean: {true_episode_returns.mean():.2f} ± {true_episode_returns.std():.2f}")
    print(f"  Min: {true_episode_returns.min():.2f}, Max: {true_episode_returns.max():.2f}")
    print(f"  (Note: Training uses CVAE rewards, this is just for reference)\n")

    # INAC configuration
    class INACConfig:
        def __init__(self):
            self.seed = 42
            self.exp_path = "./mujoco_cvae_inac_results"
            self.tensorboard_logs = False  # Disable to avoid wandb conflicts
            self.env_fn = None  # Required by logger
            self.offline_data = None  # Required by logger

    config = INACConfig()
    torch_utils.set_one_thread()
    torch_utils.random_seed(config.seed)

    # Set up logger
    os.makedirs(config.exp_path, exist_ok=True)
    inac_logger = logger.Logger(config, config.exp_path)

    # Initialize INAC agent
    agent = InSampleAC(
        device=device,
        discrete_control=0,  # Continuous control
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_units=256,
        learning_rate=learning_rate,
        tau=tau,
        polyak=0.995,
        exp_path=config.exp_path,
        seed=config.seed,
        env_fn=lambda: None,  # Dummy env function
        timeout=1000,
        gamma=0.99,
        offline_data=offline_data,
        batch_size=batch_size,
        use_target_network=1,
        target_network_update_freq=1,
        evaluation_criteria='return',
        logger=inac_logger,
        lambdaVal='none'
    )

    print(f"Training INAC for {max_steps} steps...")

    # Tracking variables for better logging
    start_time = time.time()
    loss_accumulator = {'actor': [], 'critic': [], 'value': [], 'beta': []}

    pbar = tqdm(range(max_steps), desc="INAC training")
    for step in pbar:
        batch_data = agent.get_data()
        loss_dict = agent.update(batch_data)

        # Accumulate losses for averaging
        loss_accumulator['actor'].append(loss_dict['actor'])
        loss_accumulator['critic'].append(loss_dict['critic'])
        loss_accumulator['value'].append(loss_dict['value'])
        loss_accumulator['beta'].append(loss_dict['beta'])

        if (step + 1) % log_interval == 0:
            # Calculate average losses over interval
            avg_losses = {k: np.mean(v) for k, v in loss_accumulator.items()}

            # Calculate training speed
            elapsed_time = time.time() - start_time
            steps_per_sec = log_interval / elapsed_time if elapsed_time > 0 else 0

            # Update progress bar
            pbar.set_postfix({
                'Actor': f"{avg_losses['actor']:.3f}",
                'Critic': f"{avg_losses['critic']:.3f}",
                'Value': f"{avg_losses['value']:.3f}",
                'SPS': f"{steps_per_sec:.1f}"
            })

            # Comprehensive wandb logging
            if use_wandb:
                # Get batch statistics for detailed logging
                # INAC batch_data format: 'obs', 'act', 'reward', 'obs2', 'done'
                cvae_rewards = batch_data['reward'].cpu().numpy()

                wandb_log = {
                    # Training losses (averaged)
                    'train/actor_loss': avg_losses['actor'],
                    'train/critic_loss': avg_losses['critic'],
                    'train/value_loss': avg_losses['value'],
                    'train/beta_loss': avg_losses['beta'],

                    # CVAE Reward statistics (used for training)
                    'stats/cvae_reward_mean': np.mean(cvae_rewards),
                    'stats/cvae_reward_std': np.std(cvae_rewards),
                    'stats/cvae_reward_min': np.min(cvae_rewards),
                    'stats/cvae_reward_max': np.max(cvae_rewards),

                    # Training speed
                    'perf/steps_per_sec': steps_per_sec,
                    'perf/total_steps': step + 1,

                    # Progress
                    'progress/completion': (step + 1) / max_steps
                }

                wandb.log(wandb_log, step=step + 1)

            # Reset accumulators
            loss_accumulator = {'actor': [], 'critic': [], 'value': [], 'beta': []}
            start_time = time.time()

        # Periodic evaluation
        if (step + 1) % eval_interval == 0 or (step + 1) == max_steps:
            print(f"\n=== Evaluating at step {step + 1} ===")
            try:
                # Record one video at each checkpoint
                video_folder = f"./videos/{env_name}/checkpoints"
                eval_results = evaluate_policy(
                    agent,
                    env_name,
                    n_episodes=eval_episodes,
                    record_video=True,
                    video_folder=video_folder,
                    video_prefix=f"step{step+1:05d}"
                )

                if use_wandb:
                    wandb.log({
                        'eval/mean_return': eval_results['mean_return'],
                        'eval/std_return': eval_results['std_return'],
                        'eval/mean_length': eval_results['mean_length'],
                        'eval/best_return': max(eval_results['episodes']),
                        'eval/worst_return': min(eval_results['episodes'])
                    }, step=step + 1)

                print(f"Evaluation: Return={eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}\n")
            except Exception as e:
                print(f"Evaluation failed: {e}\n")

    return agent


def evaluate_policy(agent: InSampleAC, env_name: str, n_episodes: int = 10, record_video: bool = False, video_folder: str = "./videos", video_prefix: str = ""):
    """Evaluate the trained policy."""
    import imageio

    # Use the base environment name for evaluation (not D4RL dataset)
    # Correct capitalization for each environment
    env_name_map = {
        'halfcheetah': 'HalfCheetah',
        'hopper': 'Hopper',
        'walker2d': 'Walker2d',
        'ant': 'Ant'
    }
    base_env_name = env_name_map.get(env_name, env_name.title()) + "-v3"

    # Create environment
    env = gym.make(base_env_name)

    if record_video:
        os.makedirs(video_folder, exist_ok=True)

    episode_returns = []
    episode_lengths = []

    for episode in tqdm(range(n_episodes), desc="Evaluating", leave=False):
        # Only record the first episode when called from checkpoint evals
        should_record_this_ep = record_video and episode == 0
        frames = [] if should_record_this_ep else None

        # Handle both old and new gym reset formats
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
        episode_return = 0
        episode_length = 0
        done = False

        while not done:
            # Render frame if recording this episode
            if should_record_this_ep:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _ = agent.ac.pi(state_tensor)
            action = action.cpu().numpy().squeeze()

            # Handle both old and new gym return formats
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result

            episode_return += reward
            episode_length += 1

            if episode_length >= 1000:  # Prevent infinite episodes
                break

        # Save video with imageio (only first episode)
        if should_record_this_ep and frames:
            prefix = f"{video_prefix}_" if video_prefix else ""
            video_path = os.path.join(video_folder, f"{prefix}{env_name}_return{int(episode_return)}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"✓ Saved checkpoint video: {video_path}")

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    env.close()

    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episodes': episode_returns
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({n_episodes} episodes):")
    print(f"  [True Environment Rewards]")
    print(f"{'='*50}")
    print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Min Return:  {np.min(episode_returns):.2f}")
    print(f"  Max Return:  {np.max(episode_returns):.2f}")
    if record_video:
        print(f"\n  Videos saved to: {video_folder}/")
    print(f"{'='*50}\n")

    return results


def main():
    """Main function to test CVAE + INAC pipeline on MuJoCo."""

    # Configuration
    ENV_NAME = "halfcheetah"  # Can be: halfcheetah, hopper, walker2d, ant
    DATASET_NAME = "medium-expert"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing CVAE + INAC pipeline on {ENV_NAME}-{DATASET_NAME}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    dataset = SimpleMuJoCoDataset(ENV_NAME, DATASET_NAME)

    # Train CVAE
    print("\n=== Training CVAE ===")
    cvae_model = train_cvae(dataset, device=DEVICE)

    # Train INAC with CVAE rewards
    print("\n=== Training INAC with CVAE Rewards ===")
    inac_agent = train_inac_with_cvae_rewards(dataset, cvae_model, device=DEVICE)

    # Evaluate policy
    print("\n=== Evaluating Policy ===")
    eval_results = evaluate_policy(inac_agent, ENV_NAME)

    # Save models
    output_dir = Path("./mujoco_cvae_inac_results")
    output_dir.mkdir(exist_ok=True)

    torch.save({
        'cvae_state_dict': cvae_model.state_dict(),
        'cvae_config': {
            'state_dim': dataset.state_dim,
            'action_dim': dataset.action_dim,
            'hidden_dim': 256
        },
        'inac_actor_state_dict': inac_agent.ac.pi.state_dict(),
        'eval_results': eval_results,
        'env_name': ENV_NAME,
        'dataset_name': DATASET_NAME
    }, output_dir / f"{ENV_NAME}_cvae_inac_model.pt")

    print(f"\nModels saved to {output_dir}")
    print("Testing completed!")


if __name__ == "__main__":
    main()