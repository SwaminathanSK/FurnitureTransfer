"""
Conditional VAE for Learning 1D Latent Actions for One-Leg Assembly Task

Based on "Controlling Assistive Robots with Learned Latent Actions"
https://iliad.stanford.edu/blog/2019/11/12/controlling-assistive-robots-with-learned-latent-actions/

The c-VAE learns to encode state-action pairs into a 1D latent space,
conditioned on the current state. The latent variable can then be used as:
1. A reward signal indicating task progress
2. A compact action representation for policy learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for learning 1D latent actions.
    
    Architecture:
    - Encoder: (state, action) -> latent_mean, latent_logvar
    - Decoder: (state, latent) -> action
    - Latent dimension: 1 (scalar progress indicator)
    """
    
    def __init__(
        self,
        state_dim: int = 49,  # robot_state (14) + parts_poses (35)
        action_dim: int = 8,   # action dimension
        latent_dim: int = 1,   # 1D latent space
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: (state, action) -> latent params
        encoder_input_dim = state_dim + action_dim
        encoder_layers = []
        
        # Input layer
        encoder_layers.extend([
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
        # Output to latent parameters
        encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent mean and log variance heads
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (state, latent) -> action
        decoder_input_dim = state_dim + latent_dim
        decoder_layers = []
        
        # Input layer
        decoder_layers.extend([
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Dropout(dropout)
            ])
            
        # Output layer
        decoder_layers.append(nn.Linear(hidden_dim, action_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier initialization for better training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state-action pairs to latent parameters.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            latent_mean: (batch_size, latent_dim)
            latent_logvar: (batch_size, latent_dim)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Pass through encoder
        h = self.encoder(x)
        
        # Get latent parameters
        mean = self.latent_mean(h)
        logvar = self.latent_logvar(h)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mean: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
            
        Returns:
            latent: (batch_size, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode state + latent to action.
        
        Args:
            state: (batch_size, state_dim)
            latent: (batch_size, latent_dim)
            
        Returns:
            action: (batch_size, action_dim)
        """
        # Concatenate state and latent
        x = torch.cat([state, latent], dim=-1)
        
        # Pass through decoder
        action = self.decoder(x)
        
        return action
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through c-VAE.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            Dict containing:
                - recon_action: reconstructed action
                - latent_mean: mean of latent distribution
                - latent_logvar: log variance of latent distribution
                - latent: sampled latent vector
        """
        # Encode to latent parameters
        latent_mean, latent_logvar = self.encode(state, action)
        
        # Sample latent vector
        latent = self.reparameterize(latent_mean, latent_logvar)
        
        # Decode to action
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
        beta: float = 1.0,
        reconstruction_loss_type: str = 'mse'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute c-VAE loss with reconstruction and KL divergence terms.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            beta: weighting factor for KL loss (beta-VAE)
            reconstruction_loss_type: 'mse' or 'l1'
            
        Returns:
            Dict containing loss components
        """
        # Forward pass
        outputs = self.forward(state, action)
        
        # Reconstruction loss
        if reconstruction_loss_type == 'mse':
            recon_loss = F.mse_loss(outputs['recon_action'], action, reduction='mean')
        elif reconstruction_loss_type == 'l1':
            recon_loss = F.l1_loss(outputs['recon_action'], action, reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")
        
        # KL divergence loss (regularization)
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(
            1 + outputs['latent_logvar'] - outputs['latent_mean'].pow(2) - outputs['latent_logvar'].exp()
        )
        
        # Prevent KL collapse by ensuring minimum KL loss
        kl_loss = torch.maximum(kl_loss, torch.tensor(0.1, device=kl_loss.device))
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'latent_mean': outputs['latent_mean'].mean().item(),
            'latent_std': outputs['latent_mean'].std().item()
        }
    
    def get_latent_from_state_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation from state-action pairs (for reward computation).
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            latent: (batch_size, latent_dim)
        """
        with torch.no_grad():
            latent_mean, _ = self.encode(state, action)
            return latent_mean
    
    def sample_action_from_state(self, state: torch.Tensor, latent: torch.Tensor = None) -> torch.Tensor:
        """
        Sample action from state, optionally with given latent.
        
        Args:
            state: (batch_size, state_dim)
            latent: (batch_size, latent_dim) or None for random sampling
            
        Returns:
            action: (batch_size, action_dim)
        """
        with torch.no_grad():
            if latent is None:
                # Sample from prior N(0, 1)
                latent = torch.randn(state.size(0), self.latent_dim, device=state.device)
            
            action = self.decode(state, latent)
            return action


class CVAETrainer:
    """Trainer class for the conditional VAE."""
    
    def __init__(
        self,
        model: ConditionalVAE,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        beta_schedule: str = 'constant',  # 'constant', 'linear', 'cyclical'
        beta_max: float = 1.0,
        beta_min: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Beta scheduling for KL loss
        self.beta_schedule = beta_schedule
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.step_count = 0
        
    def get_beta(self, epoch: int, max_epochs: int) -> float:
        """Get beta value based on schedule."""
        if self.beta_schedule == 'constant':
            return self.beta_max
        elif self.beta_schedule == 'linear':
            # Slower ramp-up to prevent KL collapse
            warmup_epochs = max_epochs * 0.7  # Use 70% of epochs for warmup
            return self.beta_min + (self.beta_max - self.beta_min) * min(epoch / warmup_epochs, 1.0)
        elif self.beta_schedule == 'cyclical':
            cycle_length = max_epochs // 4
            cycle_pos = epoch % cycle_length
            return self.beta_min + (self.beta_max - self.beta_min) * (cycle_pos / cycle_length)
        elif self.beta_schedule == 'warmup':
            # Start with very low beta, slowly increase
            if epoch < max_epochs * 0.5:
                return 0.01 * (epoch / (max_epochs * 0.5))
            else:
                remaining_progress = (epoch - max_epochs * 0.5) / (max_epochs * 0.5)
                return 0.01 + (self.beta_max - 0.01) * remaining_progress
        else:
            return self.beta_max
    
    def train_step(self, state_batch: torch.Tensor, action_batch: torch.Tensor, beta: float) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss_dict = self.model.compute_loss(state_batch, action_batch, beta=beta)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.step_count += 1
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def evaluate(self, val_state: torch.Tensor, val_action: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        with torch.no_grad():
            loss_dict = self.model.compute_loss(val_state, val_action, beta=1.0)
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def visualize_latent_space(
    model: ConditionalVAE,
    states: torch.Tensor,
    actions: torch.Tensor,
    save_path: str = None,
    title: str = "Latent Space Visualization"
):
    """
    Visualize the 1D latent space learned by the c-VAE.
    
    Args:
        model: trained c-VAE model
        states: (N, state_dim)
        actions: (N, action_dim)
        save_path: path to save the plot
        title: plot title
    """
    model.eval()
    with torch.no_grad():
        latents = model.get_latent_from_state_action(states, actions)
    
    latents = latents.cpu().numpy().flatten()
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Histogram of latent values
    plt.subplot(1, 2, 1)
    plt.hist(latents, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Latent Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Latent Values')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Latent values over time (assuming sequential data)
    plt.subplot(1, 2, 2)
    plt.plot(latents, alpha=0.7, color='red', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Latent Value')
    plt.title('Latent Values Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ConditionalVAE(
        state_dim=49,
        action_dim=8,
        latent_dim=1,
        hidden_dim=256,
        num_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    state = torch.randn(batch_size, 49)
    action = torch.randn(batch_size, 8)
    
    outputs = model(state, action)
    loss_dict = model.compute_loss(state, action)
    
    print("Output shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v}")