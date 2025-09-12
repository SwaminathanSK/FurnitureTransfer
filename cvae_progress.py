"""
Progress-Supervised Conditional VAE for One-Leg Assembly Task

This version explicitly supervises the 1D latent variable to represent task progress
(timestep / episode_length), making it a direct progress indicator for rewards.

Key differences from standard c-VAE:
1. Latent variable is supervised to predict progress (0 to 1)
2. Prior is uniform U(0,1) instead of normal N(0,1) 
3. Additional progress prediction loss
4. Can infer progress from state alone (without actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class ProgressConditionalVAE(nn.Module):
    """
    Progress-supervised conditional VAE for learning task progress as 1D latent.
    
    Architecture:
    - Encoder: (state, action) -> progress_mean, progress_logvar  
    - Decoder: (state, progress) -> action
    - Progress predictor: state -> progress (for inference without actions)
    """
    
    def __init__(
        self,
        state_dim: int = 58,   # robot_state (16) + parts_poses (42)
        action_dim: int = 10,   # action dimension
        latent_dim: int = 1,    # 1D progress variable
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: (state, action) -> progress parameters
        encoder_input_dim = state_dim + action_dim
        encoder_layers = []
        
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
        
        # Progress parameters (mean and log variance)
        self.progress_mean = nn.Linear(hidden_dim, latent_dim)
        self.progress_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (state, progress) -> action
        decoder_input_dim = state_dim + latent_dim
        decoder_layers = []
        
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
        
        # Progress predictor: state -> progress (for inference without actions)
        progress_predictor_layers = []
        
        progress_predictor_layers.extend([
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        for _ in range(num_layers - 2):
            progress_predictor_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        progress_predictor_layers.extend([
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()  # Output between 0 and 1
        ])
        self.progress_predictor = nn.Sequential(*progress_predictor_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier initialization for better training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state-action pairs to progress parameters.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            progress_mean: (batch_size, latent_dim) - should be between 0 and 1
            progress_logvar: (batch_size, latent_dim)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Pass through encoder
        h = self.encoder(x)
        
        # Get progress parameters
        mean = torch.sigmoid(self.progress_mean(h))  # Constrain to [0, 1]
        logvar = self.progress_logvar(h)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mean: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
            
        Returns:
            progress: (batch_size, latent_dim) - between 0 and 1
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            progress = mean + eps * std
            # Clamp to [0, 1] range
            progress = torch.clamp(progress, 0.0, 1.0)
            return progress
        else:
            return mean
    
    def decode(self, state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Decode state + progress to action.
        
        Args:
            state: (batch_size, state_dim)
            progress: (batch_size, latent_dim) - between 0 and 1
            
        Returns:
            action: (batch_size, action_dim)
        """
        # Concatenate state and progress
        x = torch.cat([state, progress], dim=-1)
        
        # Pass through decoder
        action = self.decoder(x)
        
        return action
    
    def predict_progress(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict progress from state alone (for inference without actions).
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            progress: (batch_size, latent_dim) - between 0 and 1
        """
        return self.progress_predictor(state)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through progress c-VAE.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            Dict containing outputs
        """
        # Encode to progress parameters
        progress_mean, progress_logvar = self.encode(state, action)
        
        # Sample progress
        progress = self.reparameterize(progress_mean, progress_logvar)
        
        # Decode to action
        recon_action = self.decode(state, progress)
        
        # Predict progress from state alone
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
        beta: float = 1.0,
        gamma: float = 1.0,
        reconstruction_loss_type: str = 'mse'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute progress-supervised c-VAE loss.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            true_progress: (batch_size, 1) - ground truth progress (0 to 1)
            beta: weighting factor for KL loss
            gamma: weighting factor for progress prediction loss
            reconstruction_loss_type: 'mse' or 'l1'
            
        Returns:
            Dict containing loss components
        """
        # Forward pass
        outputs = self.forward(state, action)
        
        # 1. Reconstruction loss
        if reconstruction_loss_type == 'mse':
            recon_loss = F.mse_loss(outputs['recon_action'], action, reduction='mean')
        elif reconstruction_loss_type == 'l1':
            recon_loss = F.l1_loss(outputs['recon_action'], action, reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")
        
        # 2. KL divergence loss - against uniform prior U(0, 1)
        # For uniform prior U(0,1), we want the latent to be uniformly distributed
        # We approximate this by encouraging the mean to be spread out and variance to be reasonable
        progress_mean = outputs['progress_mean']
        progress_logvar = outputs['progress_logvar']
        
        # KL divergence against uniform distribution is tricky, so we use a proxy:
        # Encourage diversity in means while keeping reasonable variance
        kl_loss = torch.mean(progress_logvar.exp()) + torch.mean((progress_mean - 0.5).pow(2))
        
        # 3. Progress supervision loss - supervise latent to match true progress
        progress_supervision_loss = F.mse_loss(outputs['progress_mean'], true_progress, reduction='mean')
        
        # 4. Progress prediction loss - state-only progress predictor
        progress_prediction_loss = F.mse_loss(outputs['predicted_progress'], true_progress, reduction='mean')
        
        # Total loss
        total_loss = (recon_loss + 
                     beta * kl_loss + 
                     gamma * progress_supervision_loss +
                     gamma * progress_prediction_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'progress_supervision_loss': progress_supervision_loss,
            'progress_prediction_loss': progress_prediction_loss,
            'progress_mean': outputs['progress_mean'].mean().item(),
            'progress_std': outputs['progress_mean'].std().item()
        }
    
    def get_progress_from_state_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Extract progress from state-action pairs.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            
        Returns:
            progress: (batch_size, latent_dim) - between 0 and 1
        """
        with torch.no_grad():
            progress_mean, _ = self.encode(state, action)
            return progress_mean
    
    def get_progress_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict progress from state alone (most useful for rewards).
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            progress: (batch_size, latent_dim) - between 0 and 1
        """
        with torch.no_grad():
            return self.predict_progress(state)


def create_progress_labels(episode_ends: np.ndarray, total_timesteps: int) -> np.ndarray:
    """
    Create progress labels (timestep / episode_length) for each sample.
    
    Args:
        episode_ends: array of episode end indices
        total_timesteps: total number of timesteps
        
    Returns:
        progress_labels: array of progress values (0 to 1) for each timestep
    """
    progress_labels = np.zeros(total_timesteps)
    
    start_idx = 0
    for episode_idx, end_idx in enumerate(episode_ends):
        episode_length = end_idx - start_idx
        
        # Create progress from 0 to 1 for this episode
        episode_progress = np.linspace(0, 1, episode_length, endpoint=False)
        progress_labels[start_idx:end_idx] = episode_progress
        
        start_idx = end_idx
    
    return progress_labels


def visualize_progress_learning(
    model: ProgressConditionalVAE,
    states: torch.Tensor,
    actions: torch.Tensor,
    true_progress: torch.Tensor,
    episode_ends: np.ndarray,
    save_path: str = None,
    title: str = "Progress Learning Visualization"
):
    """
    Visualize how well the model learned to predict task progress.
    
    Args:
        model: trained progress c-VAE
        states: (N, state_dim)
        actions: (N, action_dim) 
        true_progress: (N, 1) ground truth progress
        episode_ends: episode boundaries
        save_path: path to save plot
        title: plot title
    """
    model.eval()
    with torch.no_grad():
        # Get predictions
        predicted_from_state_action = model.get_progress_from_state_action(states, actions)
        predicted_from_state = model.get_progress_from_state(states)
    
    true_progress = true_progress.cpu().numpy().flatten()
    pred_sa = predicted_from_state_action.cpu().numpy().flatten()
    pred_s = predicted_from_state.cpu().numpy().flatten()
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Progress over time for first few episodes
    plt.subplot(2, 3, 1)
    start_idx = 0
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for ep_idx in range(min(5, len(episode_ends))):
        end_idx = min(episode_ends[ep_idx], len(true_progress))  # Handle subset bounds
        ep_length = end_idx - start_idx
        
        if ep_length > 0:  # Only plot if episode exists in subset
            plt.plot(range(ep_length), true_progress[start_idx:end_idx], 
                    color=colors[ep_idx], linestyle='-', alpha=0.7, label=f'True Ep {ep_idx+1}')
            plt.plot(range(ep_length), pred_sa[start_idx:end_idx], 
                    color=colors[ep_idx], linestyle='--', alpha=0.7, label=f'Pred Ep {ep_idx+1}')
        
        start_idx = end_idx
        if start_idx >= len(true_progress):  # Stop if we've used all data
            break
    
    plt.xlabel('Timestep within Episode')
    plt.ylabel('Progress')
    plt.title('Progress Over Time (First 5 Episodes)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: State-action progress correlation
    plt.subplot(2, 3, 2)
    plt.scatter(true_progress, pred_sa, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('True Progress')
    plt.ylabel('Predicted Progress (State+Action)')
    plt.title('State+Action Progress Prediction')
    correlation_sa = np.corrcoef(true_progress, pred_sa)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_sa:.3f}', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: State-only progress correlation  
    plt.subplot(2, 3, 3)
    plt.scatter(true_progress, pred_s, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('True Progress')
    plt.ylabel('Predicted Progress (State Only)')
    plt.title('State-Only Progress Prediction')
    correlation_s = np.corrcoef(true_progress, pred_s)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_s:.3f}', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Progress distribution
    plt.subplot(2, 3, 4)
    plt.hist(true_progress, bins=50, alpha=0.7, label='True', density=True)
    plt.hist(pred_sa, bins=50, alpha=0.7, label='Pred (S+A)', density=True)
    plt.hist(pred_s, bins=50, alpha=0.7, label='Pred (S)', density=True)
    plt.xlabel('Progress')
    plt.ylabel('Density')
    plt.title('Progress Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(2, 3, 5)
    residuals_sa = pred_sa - true_progress
    residuals_s = pred_s - true_progress
    plt.scatter(true_progress, residuals_sa, alpha=0.5, s=1, label='State+Action')
    plt.scatter(true_progress, residuals_s, alpha=0.5, s=1, label='State Only')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('True Progress')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Statistics
    plt.subplot(2, 3, 6)
    stats_text = f"""
    Progress Prediction Statistics:
    
    State + Action:
    - Correlation: {correlation_sa:.4f}
    - MSE: {np.mean((pred_sa - true_progress)**2):.4f}
    - MAE: {np.mean(np.abs(pred_sa - true_progress)):.4f}
    
    State Only:
    - Correlation: {correlation_s:.4f}  
    - MSE: {np.mean((pred_s - true_progress)**2):.4f}
    - MAE: {np.mean(np.abs(pred_s - true_progress)):.4f}
    
    Progress Range:
    - True: [{true_progress.min():.3f}, {true_progress.max():.3f}]
    - Pred (S+A): [{pred_sa.min():.3f}, {pred_sa.max():.3f}]
    - Pred (S): [{pred_s.min():.3f}, {pred_s.max():.3f}]
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Progress visualization saved to {save_path}")
    
    plt.show()
    
    return {
        'correlation_state_action': correlation_sa,
        'correlation_state_only': correlation_s,
        'mse_state_action': np.mean((pred_sa - true_progress)**2),
        'mse_state_only': np.mean((pred_s - true_progress)**2)
    }


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ProgressConditionalVAE(
        state_dim=58,
        action_dim=10,
        latent_dim=1,
        hidden_dim=256,
        num_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    state = torch.randn(batch_size, 58)
    action = torch.randn(batch_size, 10)
    true_progress = torch.rand(batch_size, 1)  # Random progress for testing
    
    outputs = model(state, action)
    loss_dict = model.compute_loss(state, action, true_progress)
    
    print("Output shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v}")