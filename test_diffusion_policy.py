#!/usr/bin/env python3
"""
Diffusion Policy for D4RL
Implements diffusion-based behavioral cloning for continuous control
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
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")


@dataclass
class DiffusionConfig:
    """Configuration for diffusion policy"""
    env_name: str
    dataset_name: str
    device: str

    # Diffusion parameters
    num_diffusion_iters: int = 100
    num_inference_steps: int = 10
    beta_schedule: str = 'squaredcos_cap_v2'

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    time_embed_dim: int = 128

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 256
    epochs: int = 100

    # Action chunking (temporal extension)
    pred_horizon: int = 4  # Predict 4 actions ahead
    obs_horizon: int = 1   # Use 1 observation
    action_horizon: int = 1  # Execute 1 action

    # Evaluation
    eval_episodes: int = 100
    save_dir: str = "./outputs/diffusion_policy"


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DiffusionUNet(nn.Module):
    """
    Simple conditional UNet for diffusion
    Conditions on: timestep and observation
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        pred_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Observation encoder
        obs_layers = []
        for i in range(2):
            obs_layers.extend([
                nn.Linear(obs_dim if i == 0 else hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout)
            ])
        self.obs_encoder = nn.Sequential(*obs_layers)

        # Action encoder (input noisy actions)
        self.action_encoder = nn.Linear(action_dim * pred_horizon, hidden_dim)

        # Main network: process concatenated [obs_emb, action_emb, time_emb]
        input_dim = hidden_dim * 2 + time_embed_dim

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout)
            ])

        self.network = nn.Sequential(*layers)

        # Output: predict noise
        self.output = nn.Linear(hidden_dim, action_dim * pred_horizon)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, noisy_action, timestep, obs):
        """
        noisy_action: (B, pred_horizon, action_dim)
        timestep: (B,)
        obs: (B, obs_dim)
        """
        B = noisy_action.shape[0]

        # Flatten actions: (B, pred_horizon, action_dim) -> (B, pred_horizon * action_dim)
        noisy_action_flat = noisy_action.reshape(B, -1)

        # Encode time
        time_emb = self.time_mlp(timestep)  # (B, time_embed_dim)

        # Encode observation
        obs_emb = self.obs_encoder(obs)  # (B, hidden_dim)

        # Encode noisy action
        action_emb = self.action_encoder(noisy_action_flat)  # (B, hidden_dim)

        # Concatenate all conditions
        x = torch.cat([obs_emb, action_emb, time_emb], dim=-1)

        # Process through network
        x = self.network(x)

        # Predict noise
        noise_pred = self.output(x)

        # Reshape back to (B, pred_horizon, action_dim)
        noise_pred = noise_pred.reshape(B, self.pred_horizon, self.action_dim)

        return noise_pred


class DiffusionPolicy:
    """Diffusion-based policy"""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DiffusionConfig
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.device

        # Diffusion model
        self.model = DiffusionUNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            pred_horizon=config.pred_horizon,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_embed_dim=config.time_embed_dim,
            dropout=config.dropout
        ).to(config.device)

        # Noise schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=True,
            prediction_type='epsilon'
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )

    def train_step(self, obs_batch, action_batch):
        """Single training step"""
        self.model.train()

        # obs_batch: (B, obs_dim)
        # action_batch: (B, pred_horizon, action_dim)

        # Sample noise
        noise = torch.randn_like(action_batch)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (obs_batch.shape[0],),
            device=self.device
        ).long()

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(action_batch, noise, timesteps)

        # Predict noise
        noise_pred = self.model(noisy_actions, timesteps, obs_batch)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, obs):
        """
        Predict action sequence using diffusion
        obs: (B, obs_dim)
        Returns: (B, pred_horizon, action_dim)
        """
        self.model.eval()

        B = obs.shape[0]

        # Start with random noise
        action = torch.randn(
            (B, self.config.pred_horizon, self.action_dim),
            device=self.device
        )

        # Set timesteps for inference
        self.inference_scheduler.set_timesteps(self.config.num_inference_steps)

        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            # Predict noise
            noise_pred = self.model(action, t.unsqueeze(0).repeat(B).to(self.device), obs)

            # Remove noise
            action = self.inference_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=action
            ).prev_sample

        return action


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


def prepare_diffusion_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    pred_horizon: int,
    obs_horizon: int = 1
):
    """
    Prepare dataset for diffusion policy with temporal extension
    Creates sequences of observations and action chunks
    """
    N = len(states)

    # For now, use simple implementation: obs_horizon=1
    # Create action chunks
    obs_list = []
    action_chunks = []

    for i in range(N - pred_horizon + 1):
        obs_list.append(states[i])
        action_chunks.append(actions[i:i + pred_horizon])

    return np.array(obs_list), np.array(action_chunks)


def train_diffusion_policy(
    states: torch.Tensor,
    actions: torch.Tensor,
    obs_dim: int,
    action_dim: int,
    config: DiffusionConfig,
    use_wandb: bool = False
) -> DiffusionPolicy:
    """Train diffusion policy"""

    print(f"\n{'='*60}")
    print("Training Diffusion Policy")
    print(f"{'='*60}")
    print(f"Observations: {states.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Prediction horizon: {config.pred_horizon}")
    print(f"Diffusion iters: {config.num_diffusion_iters}")
    print(f"Inference steps: {config.num_inference_steps}")

    # Create policy
    policy = DiffusionPolicy(obs_dim, action_dim, config)

    # Create dataloader
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Training loop
    for epoch in range(config.epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for obs_batch, action_batch in pbar:
            loss = policy.train_step(obs_batch, action_batch)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = total_loss / num_batches

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss
            })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    return policy


def evaluate_diffusion_policy(
    env,
    policy: DiffusionPolicy,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    num_episodes: int,
    use_wandb: bool = False
) -> Dict:
    """Evaluate diffusion policy"""

    returns = []

    pbar = tqdm(range(num_episodes), desc="Evaluating")
    for episode in pbar:
        state = env.reset()
        done = False
        episode_return = 0

        while not done:
            # Normalize state
            state_normalized = (state - state_mean) / state_std
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(policy.device)

            # Predict action sequence
            action_sequence = policy.predict(state_tensor)

            # Take first action only
            action = action_sequence[0, 0].cpu().numpy()

            # Denormalize action
            action = action * action_std + action_mean

            # Step environment
            state, reward, done, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)
        pbar.set_postfix({'return': f'{episode_return:.2f}', 'mean': f'{np.mean(returns):.2f}'})

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'all_returns': returns
    }


def main():
    parser = argparse.ArgumentParser(description='Diffusion Policy for D4RL')
    parser.add_argument('--env', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium-replay')
    parser.add_argument('--pred-horizon', type=int, default=4,
                       help='Number of actions to predict ahead')
    parser.add_argument('--num-diffusion-iters', type=int, default=100,
                       help='Number of diffusion steps during training')
    parser.add_argument('--num-inference-steps', type=int, default=10,
                       help='Number of denoising steps during inference')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    use_wandb = not args.no_wandb and WANDB_AVAILABLE

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="diffusion-policy-d4rl",
            entity="swami2004",
            name=f"{args.env}_{args.dataset}_diffusion_h{args.pred_horizon}_seed{args.seed}",
            config={
                'env': args.env,
                'dataset': args.dataset,
                'pred_horizon': args.pred_horizon,
                'num_diffusion_iters': args.num_diffusion_iters,
                'num_inference_steps': args.num_inference_steps,
                'epochs': args.epochs,
                'seed': args.seed,
                'method': 'diffusion_policy'
            }
        )

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    print("Loading D4RL dataset...")
    env, dataset = load_d4rl_dataset(args.env, args.dataset)

    obs_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    # Compute normalization stats
    state_mean = dataset['observations'].mean(axis=0)
    state_std = dataset['observations'].std(axis=0) + 1e-8
    action_mean = dataset['actions'].mean(axis=0)
    action_std = dataset['actions'].std(axis=0) + 1e-8

    # Normalize
    states_normalized = (dataset['observations'] - state_mean) / state_std
    actions_normalized = (dataset['actions'] - action_mean) / action_std

    # Prepare dataset with temporal extension
    print(f"\nPreparing dataset with pred_horizon={args.pred_horizon}...")
    obs_seq, action_chunks = prepare_diffusion_dataset(
        states_normalized,
        actions_normalized,
        pred_horizon=args.pred_horizon
    )

    print(f"Dataset size: {len(obs_seq)} sequences")

    # Convert to tensors
    states_tensor = torch.FloatTensor(obs_seq).to(args.device)
    actions_tensor = torch.FloatTensor(action_chunks).to(args.device)

    # Create config
    config = DiffusionConfig(
        env_name=args.env,
        dataset_name=args.dataset,
        device=args.device,
        pred_horizon=args.pred_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        num_inference_steps=args.num_inference_steps,
        epochs=args.epochs,
        eval_episodes=args.eval_episodes
    )

    # Train policy
    policy = train_diffusion_policy(
        states_tensor,
        actions_tensor,
        obs_dim,
        action_dim,
        config,
        use_wandb=use_wandb
    )

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating Diffusion Policy")
    print(f"{'='*60}")

    results = evaluate_diffusion_policy(
        env, policy,
        state_mean, state_std,
        action_mean, action_std,
        args.eval_episodes,
        use_wandb=use_wandb
    )

    d4rl_score = env.get_normalized_score(results['mean_return']) * 100

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'eval/return_mean': results['mean_return'],
            'eval/return_std': results['std_return'],
            'eval/d4rl_score': d4rl_score
        })
        wandb.finish()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"D4RL Score: {d4rl_score:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        config.save_dir,
        f"{args.env}_{args.dataset}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    results_dict = {
        'return_mean': results['mean_return'],
        'return_std': results['std_return'],
        'd4rl_score': d4rl_score,
        'config': {
            'env': args.env,
            'dataset': args.dataset,
            'pred_horizon': args.pred_horizon,
            'num_diffusion_iters': args.num_diffusion_iters,
            'num_inference_steps': args.num_inference_steps,
            'epochs': args.epochs,
            'seed': args.seed
        }
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save model
    torch.save({
        'model_state_dict': policy.model.state_dict(),
        'config': config,
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'obs_dim': obs_dim,
        'action_dim': action_dim
    }, os.path.join(save_dir, 'diffusion_policy.pt'))

    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
