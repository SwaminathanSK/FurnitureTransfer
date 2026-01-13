#!/usr/bin/env python3
"""
Diffusion Policy + CVAE for D4RL
Integrates CVAE latent representation with diffusion-based behavioral cloning
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
class DiffusionCVAEConfig:
    """Configuration for diffusion policy with CVAE"""
    env_name: str
    dataset_name: str
    device: str

    # CVAE parameters
    latent_extra_dim: int = 1  # Extra latent dimensions beyond action_dim
    cvae_hidden_dim: int = 256
    cvae_num_layers: int = 3
    cvae_dropout: float = 0.1
    cvae_epochs: int = 50
    cvae_learning_rate: float = 3e-4
    cvae_beta: float = 0.001  # KL weight

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
    save_dir: str = "./outputs/diffusion_cvae"


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
    Simple conditional UNet for diffusion in LATENT space
    Conditions on: timestep and observation
    Operates on LATENT actions (not raw actions)
    """
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,  # Use latent_dim instead of action_dim
        pred_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
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

        # Latent action encoder (input noisy latent actions)
        self.latent_encoder = nn.Linear(latent_dim * pred_horizon, hidden_dim)

        # Main network
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

        # Output: predict noise in latent space
        self.output = nn.Linear(hidden_dim, latent_dim * pred_horizon)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, noisy_latent, timestep, obs):
        """
        noisy_latent: (B, pred_horizon, latent_dim)
        timestep: (B,)
        obs: (B, obs_dim)
        """
        B = noisy_latent.shape[0]

        # Flatten latents
        noisy_latent_flat = noisy_latent.reshape(B, -1)

        # Encode time
        time_emb = self.time_mlp(timestep)

        # Encode observation
        obs_emb = self.obs_encoder(obs)

        # Encode noisy latent
        latent_emb = self.latent_encoder(noisy_latent_flat)

        # Concatenate all conditions
        x = torch.cat([obs_emb, latent_emb, time_emb], dim=-1)

        # Process through network
        x = self.network(x)

        # Predict noise
        noise_pred = self.output(x)

        # Reshape back
        noise_pred = noise_pred.reshape(B, self.pred_horizon, self.latent_dim)

        return noise_pred


class DiffusionCVAEPolicy:
    """Diffusion policy operating in CVAE latent space"""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cvae: HigherDimLatentCVAE,
        config: DiffusionCVAEConfig
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = cvae.latent_dim
        self.cvae = cvae
        self.config = config
        self.device = config.device

        # Freeze CVAE
        for param in self.cvae.parameters():
            param.requires_grad = False
        self.cvae.eval()

        # Diffusion model operates in LATENT space
        self.model = DiffusionUNet(
            obs_dim=obs_dim,
            latent_dim=self.latent_dim,  # Use latent_dim
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

    def train_step(self, obs_batch, latent_action_batch):
        """Single training step in LATENT space"""
        self.model.train()

        # obs_batch: (B, obs_dim)
        # latent_action_batch: (B, pred_horizon, latent_dim)

        # Sample noise
        noise = torch.randn_like(latent_action_batch)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (obs_batch.shape[0],),
            device=self.device
        ).long()

        # Add noise to latent actions
        noisy_latents = self.noise_scheduler.add_noise(latent_action_batch, noise, timesteps)

        # Predict noise
        noise_pred = self.model(noisy_latents, timesteps, obs_batch)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, obs, states_for_decode):
        """
        Predict action sequence using diffusion + CVAE
        obs: (B, obs_dim) - normalized observations for diffusion
        states_for_decode: (B, pred_horizon, obs_dim) - states for CVAE decoder
        Returns: (B, pred_horizon, action_dim) - predicted actions
        """
        self.model.eval()

        B = obs.shape[0]

        # Start with random noise in LATENT space
        latent = torch.randn(
            (B, self.config.pred_horizon, self.latent_dim),
            device=self.device
        )

        # Set timesteps for inference
        self.inference_scheduler.set_timesteps(self.config.num_inference_steps)

        # Iterative denoising in LATENT space
        for t in self.inference_scheduler.timesteps:
            # Predict noise
            noise_pred = self.model(latent, t.unsqueeze(0).repeat(B).to(self.device), obs)

            # Remove noise
            latent = self.inference_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latent
            ).prev_sample

        # Decode latent actions to action space using CVAE
        actions = []
        for h in range(self.config.pred_horizon):
            action = self.cvae.decode(states_for_decode[:, h], latent[:, h])
            actions.append(action)

        actions = torch.stack(actions, dim=1)  # (B, pred_horizon, action_dim)

        return actions


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


def train_cvae(
    states: torch.Tensor,
    actions: torch.Tensor,
    obs_dim: int,
    action_dim: int,
    config: DiffusionCVAEConfig,
    use_wandb: bool = False
) -> HigherDimLatentCVAE:
    """Train CVAE"""

    print(f"\n{'='*60}")
    print("Training CVAE")
    print(f"{'='*60}")
    print(f"Latent dim: {action_dim + config.latent_extra_dim} (action_dim + {config.latent_extra_dim})")

    cvae = HigherDimLatentCVAE(
        obs_dim, action_dim, config.latent_extra_dim,
        config.cvae_hidden_dim, config.cvae_num_layers, config.cvae_dropout
    ).to(config.device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=config.cvae_learning_rate)

    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    for epoch in range(config.cvae_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"CVAE Epoch {epoch+1}/{config.cvae_epochs}")
        for state_batch, action_batch in pbar:
            recon_action, mu, logvar = cvae(state_batch, action_batch)

            recon_loss = F.mse_loss(recon_action, action_batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state_batch.shape[0]
            loss = recon_loss + config.cvae_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'cvae_epoch': epoch + 1,
                'cvae_loss': avg_loss,
                'cvae_recon': avg_recon,
                'cvae_kl': avg_kl
            })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.cvae_epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

    cvae.eval()
    return cvae


def prepare_diffusion_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    pred_horizon: int,
    obs_horizon: int = 1
):
    """Prepare dataset for diffusion policy with temporal extension"""
    N = len(states)

    obs_list = []
    action_chunks = []
    state_chunks = []  # For CVAE decoding

    for i in range(N - pred_horizon + 1):
        obs_list.append(states[i])
        action_chunks.append(actions[i:i + pred_horizon])
        state_chunks.append(states[i:i + pred_horizon])  # States for each action in chunk

    return np.array(obs_list), np.array(action_chunks), np.array(state_chunks)


def train_diffusion_cvae_policy(
    states: torch.Tensor,
    latent_actions: torch.Tensor,
    state_chunks: torch.Tensor,
    obs_dim: int,
    action_dim: int,
    cvae: HigherDimLatentCVAE,
    config: DiffusionCVAEConfig,
    use_wandb: bool = False
) -> DiffusionCVAEPolicy:
    """Train diffusion policy in CVAE latent space"""

    print(f"\n{'='*60}")
    print("Training Diffusion Policy in Latent Space")
    print(f"{'='*60}")
    print(f"Observations: {states.shape}")
    print(f"Latent Actions: {latent_actions.shape}")
    print(f"Latent dim: {cvae.latent_dim}")
    print(f"Prediction horizon: {config.pred_horizon}")

    # Create policy
    policy = DiffusionCVAEPolicy(obs_dim, action_dim, cvae, config)

    # Create dataloader
    dataset = TensorDataset(states, latent_actions)
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

        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{config.epochs}")
        for obs_batch, latent_batch in pbar:
            loss = policy.train_step(obs_batch, latent_batch)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = total_loss / num_batches

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'diffusion_epoch': epoch + 1,
                'diffusion_loss': avg_loss
            })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    return policy


def evaluate_diffusion_cvae_policy(
    env,
    policy: DiffusionCVAEPolicy,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    num_episodes: int,
    use_wandb: bool = False
) -> Dict:
    """Evaluate diffusion+CVAE policy"""

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

            # For decoding, we need states for all pred_horizon steps
            # Since we don't have future states, repeat current state
            states_for_decode = state_tensor.unsqueeze(1).repeat(1, policy.config.pred_horizon, 1)

            # Predict action sequence
            action_sequence = policy.predict(state_tensor, states_for_decode)

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
    parser = argparse.ArgumentParser(description='Diffusion Policy + CVAE for D4RL')
    parser.add_argument('--env', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium-replay')
    parser.add_argument('--latent-extra', type=int, default=1,
                       help='Extra latent dimensions beyond action_dim')
    parser.add_argument('--pred-horizon', type=int, default=4,
                       help='Number of actions to predict ahead')
    parser.add_argument('--cvae-epochs', type=int, default=50)
    parser.add_argument('--diffusion-epochs', type=int, default=100)
    parser.add_argument('--num-diffusion-iters', type=int, default=100)
    parser.add_argument('--num-inference-steps', type=int, default=10)
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
            name=f"{args.env}_{args.dataset}_diffusion_cvae_latent{args.latent_extra}_h{args.pred_horizon}_seed{args.seed}",
            config={
                'env': args.env,
                'dataset': args.dataset,
                'latent_extra': args.latent_extra,
                'pred_horizon': args.pred_horizon,
                'cvae_epochs': args.cvae_epochs,
                'diffusion_epochs': args.diffusion_epochs,
                'num_diffusion_iters': args.num_diffusion_iters,
                'num_inference_steps': args.num_inference_steps,
                'seed': args.seed,
                'method': 'diffusion_cvae'
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

    # Convert to tensors for CVAE training
    states_tensor = torch.FloatTensor(states_normalized).to(args.device)
    actions_tensor = torch.FloatTensor(actions_normalized).to(args.device)

    # Create config
    config = DiffusionCVAEConfig(
        env_name=args.env,
        dataset_name=args.dataset,
        device=args.device,
        latent_extra_dim=args.latent_extra,
        pred_horizon=args.pred_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        num_inference_steps=args.num_inference_steps,
        cvae_epochs=args.cvae_epochs,
        epochs=args.diffusion_epochs,
        eval_episodes=args.eval_episodes
    )

    # Step 1: Train CVAE
    cvae = train_cvae(
        states_tensor,
        actions_tensor,
        obs_dim,
        action_dim,
        config,
        use_wandb=use_wandb
    )

    # Step 2: Encode actions to latent space
    print("\nEncoding actions to latent space...")
    with torch.no_grad():
        latent_actions_flat = cvae.get_latent_action(states_tensor, actions_tensor).cpu().numpy()

    # Step 3: Prepare dataset with temporal extension
    print(f"\nPreparing dataset with pred_horizon={args.pred_horizon}...")
    obs_seq, action_chunks, state_chunks = prepare_diffusion_dataset(
        states_normalized,
        latent_actions_flat,
        pred_horizon=args.pred_horizon
    )

    print(f"Dataset size: {len(obs_seq)} sequences")

    # Convert to tensors
    obs_seq_tensor = torch.FloatTensor(obs_seq).to(args.device)
    latent_chunks_tensor = torch.FloatTensor(action_chunks).to(args.device)
    state_chunks_tensor = torch.FloatTensor(state_chunks).to(args.device)

    # Step 4: Train diffusion policy in latent space
    policy = train_diffusion_cvae_policy(
        obs_seq_tensor,
        latent_chunks_tensor,
        state_chunks_tensor,
        obs_dim,
        action_dim,
        cvae,
        config,
        use_wandb=use_wandb
    )

    # Step 5: Evaluate
    print(f"\n{'='*60}")
    print("Evaluating Diffusion+CVAE Policy")
    print(f"{'='*60}")

    results = evaluate_diffusion_cvae_policy(
        env, policy,
        state_mean, state_std,
        action_mean, action_std,
        args.eval_episodes,
        use_wandb=use_wandb
    )

    d4rl_score = env.get_normalized_score(results['mean_return']) * 100

    # Log final results
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
        f"{args.env}_{args.dataset}_latent{args.latent_extra}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    results_dict = {
        'return_mean': results['mean_return'],
        'return_std': results['std_return'],
        'd4rl_score': d4rl_score,
        'config': {
            'env': args.env,
            'dataset': args.dataset,
            'latent_extra': args.latent_extra,
            'pred_horizon': args.pred_horizon,
            'cvae_epochs': args.cvae_epochs,
            'diffusion_epochs': args.diffusion_epochs,
            'num_diffusion_iters': args.num_diffusion_iters,
            'num_inference_steps': args.num_inference_steps,
            'seed': args.seed
        }
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
