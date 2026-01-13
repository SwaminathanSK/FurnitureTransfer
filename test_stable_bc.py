#!/usr/bin/env python3
"""
Stabilized Behavioral Cloning with CVAE
Implements variance reduction techniques for more stable training
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import gym
import d4rl
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from collections import deque


@dataclass
class StableTrainingConfig:
    """Configuration for stable BC training"""
    env_name: str
    dataset_name: str
    latent_extra_dim: int
    device: str

    # Network architecture
    bc_hidden_dim: int = 256
    bc_num_layers: int = 3
    bc_dropout: float = 0.1

    # Training stability features
    bc_learning_rate: float = 1e-4  # Lower LR for stability
    bc_batch_size: int = 256
    bc_epochs: int = 150
    weight_decay: float = 1e-4  # L2 regularization
    grad_clip: float = 1.0

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine' or 'plateau'

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Exponential moving average
    use_ema: bool = True
    ema_decay: float = 0.999

    # Validation split
    val_split: float = 0.1

    # Evaluation
    eval_episodes: int = 100
    save_dir: str = "./outputs/stable_bc"


class HigherDimLatentCVAE(nn.Module):
    """CVAE with higher-dimensional latent space"""
    def __init__(self, state_dim, action_dim, extra_latent_dim=1,
                 hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extra_latent_dim = extra_latent_dim
        self.latent_dim = action_dim + extra_latent_dim

        # Encoder
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

        # Decoder
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

    def decode(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def get_latent_action(self, state, action):
        mu, _ = self.encode(state, action)
        return mu


class BCPolicy(nn.Module):
    """BC policy with improved initialization"""
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

        # Better weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.network(state)


class EMAModel:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                   (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Replace model parameters with EMA values"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # Restore best model
                model.load_state_dict(self.best_state_dict)
                return True
        else:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
            self.counter = 0

        return False


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


def train_stable_bc(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_dim: int,
    action_dim: int,
    config: StableTrainingConfig,
    seed: int
) -> BCPolicy:
    """Train BC policy with stability improvements"""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Split into train/val
    num_samples = states.shape[0]
    num_val = int(num_samples * config.val_split)
    indices = np.random.permutation(num_samples)

    train_indices = indices[num_val:]
    val_indices = indices[:num_val]

    train_states = states[train_indices]
    train_actions = actions[train_indices]
    val_states = states[val_indices]
    val_actions = actions[val_indices]

    # Create dataloaders
    train_dataset = TensorDataset(train_states, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=config.bc_batch_size,
                             shuffle=True, drop_last=True)

    val_dataset = TensorDataset(val_states, val_actions)
    val_loader = DataLoader(val_dataset, batch_size=config.bc_batch_size,
                           shuffle=False)

    # Initialize model
    policy = BCPolicy(state_dim, action_dim, config.bc_hidden_dim,
                     config.bc_num_layers, config.bc_dropout).to(config.device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.bc_learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=config.bc_epochs)
        else:  # plateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                         patience=10, verbose=True)

    # EMA
    if config.use_ema:
        ema = EMAModel(policy, decay=config.ema_decay)

    # Early stopping
    if config.use_early_stopping:
        early_stopping = EarlyStopping(patience=config.patience,
                                      min_delta=config.min_delta)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training Stable BC (seed={seed})")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"LR: {config.bc_learning_rate}, Weight decay: {config.weight_decay}")
    print(f"Scheduler: {config.scheduler_type if config.use_lr_scheduler else 'None'}")
    print(f"EMA: {config.use_ema}, Early stopping: {config.use_early_stopping}")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config.bc_epochs):
        # Training
        policy.train()
        train_loss = 0
        num_batches = 0

        for batch_states, batch_actions in train_loader:
            pred_actions = policy(batch_states)
            loss = F.mse_loss(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)

            optimizer.step()

            # Update EMA
            if config.use_ema:
                ema.update()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        policy.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                pred_actions = policy(batch_states)
                loss = F.mse_loss(pred_actions, batch_actions)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # Update LR scheduler
        if config.use_lr_scheduler:
            if config.scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.bc_epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Early stopping
        if config.use_early_stopping:
            if early_stopping(avg_val_loss, policy):
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    # Use EMA weights for final model
    if config.use_ema:
        ema.apply_shadow()

    policy.eval()

    return policy, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses)
    }


def evaluate_policy(
    env, policy, cvae,
    state_mean, state_std,
    action_mean, action_std,
    num_episodes: int,
    device: str
) -> Dict:
    """Evaluate policy"""
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0

        while not done:
            state_normalized = (state - state_mean) / state_std
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(device)

            with torch.no_grad():
                latent_action = policy(state_tensor)
                action_pred = cvae.decode(state_tensor, latent_action)
                action = action_pred.cpu().numpy()[0]

            # Denormalize
            action = action * action_std + action_mean

            state, reward, done, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'all_returns': returns
    }


def main():
    parser = argparse.ArgumentParser(description='Stable BC with CVAE')
    parser.add_argument('--env', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium-replay')
    parser.add_argument('--cvae-checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bc-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--no-early-stop', action='store_true')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load CVAE
    print("Loading CVAE...")
    cvae, cvae_checkpoint = load_cvae_checkpoint(args.cvae_checkpoint, args.device)

    state_mean = cvae_checkpoint['state_mean'].cpu().numpy()
    state_std = cvae_checkpoint['state_std'].cpu().numpy()
    action_mean = cvae_checkpoint['action_mean'].cpu().numpy()
    action_std = cvae_checkpoint['action_std'].cpu().numpy()

    state_dim = cvae_checkpoint['state_dim']
    action_dim = cvae_checkpoint['action_dim']
    latent_dim = cvae.latent_dim

    # Load dataset
    env, dataset = load_d4rl_dataset(args.env, args.dataset)

    states = (dataset['observations'] - state_mean) / state_std
    actions_normalized = (dataset['actions'] - action_mean) / action_std

    states_tensor = torch.FloatTensor(states).to(args.device)
    actions_tensor = torch.FloatTensor(actions_normalized).to(args.device)

    # Encode to latent
    with torch.no_grad():
        latent_actions = cvae.get_latent_action(states_tensor, actions_tensor)

    # Config
    config = StableTrainingConfig(
        env_name=args.env,
        dataset_name=args.dataset,
        latent_extra_dim=cvae_checkpoint['extra_latent_dim'],
        device=args.device,
        bc_epochs=args.bc_epochs,
        bc_learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_ema=not args.no_ema,
        use_early_stopping=not args.no_early_stop,
        scheduler_type=args.scheduler,
        use_lr_scheduler=(args.scheduler != 'none'),
        eval_episodes=args.eval_episodes
    )

    # Train
    policy, train_info = train_stable_bc(
        states_tensor, latent_actions,
        state_dim, latent_dim,
        config, args.seed
    )

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating Policy")
    print(f"{'='*60}")

    results = evaluate_policy(
        env, policy, cvae,
        state_mean, state_std,
        action_mean, action_std,
        args.eval_episodes,
        args.device
    )

    d4rl_score = env.get_normalized_score(results['mean_return']) * 100

    print(f"\nReturn: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"D4RL Score: {d4rl_score:.2f}")
    print(f"Training stopped at epoch: {train_info['final_epoch']}")
    print(f"Best validation loss: {train_info['best_val_loss']:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.save_dir, f"{args.env}_stable_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    results_dict = {
        'return_mean': results['mean_return'],
        'return_std': results['std_return'],
        'd4rl_score': d4rl_score,
        'train_info': train_info,
        'config': vars(args)
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
