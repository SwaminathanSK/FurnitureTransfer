"""
Training script for Progress-Supervised Conditional VAE

This trains a c-VAE where the 1D latent variable is explicitly supervised
to represent task progress (timestep / episode_length).
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from typing import Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvae_progress import ProgressConditionalVAE, create_progress_labels, visualize_progress_learning


class ProgressOneLegDataset(Dataset):
    """
    Dataset that includes progress labels for supervised training.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        normalize: bool = True,
        data_subset: int = None,
        action_type: str = 'pos'
    ):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        print(f"Loading progress-supervised dataset from {dataset_path}")
        dataset = zarr.open(dataset_path, mode='r')
        
        # Load state-action data
        robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])
        self.states = np.concatenate([robot_states, parts_poses], axis=-1)
        self.actions = np.array(dataset[f'action/{action_type}'][:])
        
        # Load episode information
        self.episode_ends = np.array(dataset['episode_ends'][:])
        self.rewards = np.array(dataset['reward'][:])
        self.success = np.array(dataset['success'][:])
        
        # Create progress labels
        print("Creating progress labels...")
        self.progress_labels = create_progress_labels(self.episode_ends, len(self.states))
        
        print(f"Progress labels stats: min={self.progress_labels.min():.3f}, "
              f"max={self.progress_labels.max():.3f}, mean={self.progress_labels.mean():.3f}")
        
        # Apply data subset if specified
        if data_subset is not None:
            self.states = self.states[:data_subset]
            self.actions = self.actions[:data_subset]
            self.progress_labels = self.progress_labels[:data_subset]
            # Adjust episode_ends for subset
            subset_episode_ends = []
            for end_idx in self.episode_ends:
                if end_idx <= data_subset:
                    subset_episode_ends.append(end_idx)
                else:
                    subset_episode_ends.append(data_subset)
                    break
            self.episode_ends = np.array(subset_episode_ends)
        
        print(f"Loaded {len(self.states)} state-action pairs with progress labels")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
        print(f"Episodes: {len(self.episode_ends)}")
        
        # Compute normalization statistics
        if normalize:
            self.state_mean = np.mean(self.states, axis=0)
            self.state_std = np.std(self.states, axis=0) + 1e-8
            
            self.action_mean = np.mean(self.actions, axis=0)
            self.action_std = np.std(self.actions, axis=0) + 1e-8
            
            # Normalize data
            self.states = (self.states - self.state_mean) / self.state_std
            self.actions = (self.actions - self.action_mean) / self.action_std
            
            self.normalize = True
            print("Data normalized")
        else:
            self.normalize = False
        
        # Convert to torch tensors
        self.states = torch.FloatTensor(self.states)
        self.actions = torch.FloatTensor(self.actions)
        self.progress_labels = torch.FloatTensor(self.progress_labels).unsqueeze(1)  # (N, 1)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.progress_labels[idx]
    
    def get_stats(self):
        """Return normalization statistics."""
        if self.normalize:
            return {
                'state_mean': self.state_mean,
                'state_std': self.state_std,
                'action_mean': self.action_mean,
                'action_std': self.action_std
            }
        else:
            return None


class ProgressCVAETrainer:
    """Trainer for progress-supervised c-VAE."""
    
    def __init__(
        self,
        model: ProgressConditionalVAE,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        beta: float = 0.1,      # Lower for progress supervision
        gamma: float = 10.0,    # High weight for progress supervision
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.gamma = gamma
        
    def train_step(
        self, 
        state_batch: torch.Tensor, 
        action_batch: torch.Tensor, 
        progress_batch: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss_dict = self.model.compute_loss(
            state_batch, action_batch, progress_batch,
            beta=self.beta, gamma=self.gamma
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def evaluate(
        self, 
        val_state: torch.Tensor, 
        val_action: torch.Tensor, 
        val_progress: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        with torch.no_grad():
            loss_dict = self.model.compute_loss(
                val_state, val_action, val_progress,
                beta=self.beta, gamma=self.gamma
            )
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def plot_progress_training_curves(train_losses, val_losses, save_path=None):
    """Plot training curves including progress losses."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total loss
    axes[0, 0].plot(epochs, [l['total_loss'] for l in train_losses], 'b-', label='Train')
    axes[0, 0].plot(epochs, [l['total_loss'] for l in val_losses], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, [l['recon_loss'] for l in train_losses], 'b-', label='Train')
    axes[0, 1].plot(epochs, [l['recon_loss'] for l in val_losses], 'r-', label='Val')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL loss
    axes[0, 2].plot(epochs, [l['kl_loss'] for l in train_losses], 'b-', label='Train')
    axes[0, 2].plot(epochs, [l['kl_loss'] for l in val_losses], 'r-', label='Val')
    axes[0, 2].set_title('KL Divergence Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Progress supervision loss
    axes[1, 0].plot(epochs, [l['progress_supervision_loss'] for l in train_losses], 'b-', label='Train')
    axes[1, 0].plot(epochs, [l['progress_supervision_loss'] for l in val_losses], 'r-', label='Val')
    axes[1, 0].set_title('Progress Supervision Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Progress prediction loss
    axes[1, 1].plot(epochs, [l['progress_prediction_loss'] for l in train_losses], 'b-', label='Train')
    axes[1, 1].plot(epochs, [l['progress_prediction_loss'] for l in val_losses], 'r-', label='Val')
    axes[1, 1].set_title('Progress Prediction Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Progress statistics
    axes[1, 2].plot(epochs, [l['progress_mean'] for l in train_losses], 'b-', label='Mean')
    axes[1, 2].plot(epochs, [l['progress_std'] for l in train_losses], 'g-', label='Std')
    axes[1, 2].set_title('Progress Statistics (Train)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def train_progress_cvae(
    data_path: str,
    output_dir: str = "./progress_cvae_results",
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 0.1,
    gamma: float = 10.0,
    hidden_dim: int = 256,
    num_layers: int = 3,
    data_subset: int = None,
    val_split: float = 0.1,
    device: str = None
):
    """Train the progress-supervised c-VAE."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ProgressOneLegDataset(
        dataset_path=data_path,
        normalize=True,
        data_subset=data_subset
    )
    
    # Split into train/validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and trainer
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    
    model = ProgressConditionalVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    trainer = ProgressCVAETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        beta=beta,
        gamma=gamma
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Beta (KL weight): {beta}, Gamma (Progress weight): {gamma}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (states, actions, progress) in enumerate(train_pbar):
            states = states.to(device)
            actions = actions.to(device)
            progress = progress.to(device)
            
            loss_dict = trainer.train_step(states, actions, progress)
            train_loss_epoch.append(loss_dict)
            
            # Update progress bar
            train_pbar.set_postfix({
                'total': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'prog_sup': f"{loss_dict['progress_supervision_loss']:.4f}",
                'prog_pred': f"{loss_dict['progress_prediction_loss']:.4f}"
            })
        
        # Average training losses
        avg_train_loss = {
            key: np.mean([loss[key] for loss in train_loss_epoch])
            for key in train_loss_epoch[0].keys()
        }
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss_epoch = []
        
        with torch.no_grad():
            for states, actions, progress in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                progress = progress.to(device)
                
                loss_dict = trainer.evaluate(states, actions, progress)
                val_loss_epoch.append(loss_dict)
        
        # Average validation losses
        avg_val_loss = {
            key: np.mean([loss[key] for loss in val_loss_epoch])
            for key in val_loss_epoch[0].keys()
        }
        val_losses.append(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Total: {avg_train_loss['total_loss']:.4f}, "
              f"Recon: {avg_train_loss['recon_loss']:.4f}, "
              f"Prog Sup: {avg_train_loss['progress_supervision_loss']:.4f}, "
              f"Prog Pred: {avg_train_loss['progress_prediction_loss']:.4f}")
        print(f"  Val   - Total: {avg_val_loss['total_loss']:.4f}, "
              f"Recon: {avg_val_loss['recon_loss']:.4f}, "
              f"Prog Sup: {avg_val_loss['progress_supervision_loss']:.4f}, "
              f"Prog Pred: {avg_val_loss['progress_prediction_loss']:.4f}")
        
        # Save best model
        if avg_val_loss['total_loss'] < best_val_loss:
            best_val_loss = avg_val_loss['total_loss']
            model_path = output_dir / 'best_progress_cvae_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'latent_dim': 1,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                },
                'training_config': {
                    'beta': beta,
                    'gamma': gamma,
                    'learning_rate': learning_rate,
                },
                'data_stats': dataset.get_stats()
            }, model_path)
            print(f"  Saved best model to {model_path}")
    
    print("Training completed!")
    
    # Plot training curves
    curves_path = output_dir / 'progress_training_curves.png'
    plot_progress_training_curves(train_losses, val_losses, save_path=curves_path)
    
    # Visualize progress learning on validation data
    val_states = []
    val_actions = []
    val_progress = []
    for states, actions, progress in val_loader:
        val_states.append(states)
        val_actions.append(actions)
        val_progress.append(progress)
    
    val_states = torch.cat(val_states, dim=0).to(device)
    val_actions = torch.cat(val_actions, dim=0).to(device)
    val_progress = torch.cat(val_progress, dim=0)
    
    progress_viz_path = output_dir / 'progress_learning_visualization.png'
    metrics = visualize_progress_learning(
        model, val_states, val_actions, val_progress, dataset.episode_ends,
        save_path=progress_viz_path,
        title="Progress Learning - One-Leg Assembly Task"
    )
    
    # Save final model
    final_model_path = output_dir / 'final_progress_cvae_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': epochs,
        'final_train_loss': avg_train_loss['total_loss'],
        'final_val_loss': avg_val_loss['total_loss'],
        'model_config': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'latent_dim': 1,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
        },
        'training_config': {
            'beta': beta,
            'gamma': gamma,
            'learning_rate': learning_rate,
        },
        'data_stats': dataset.get_stats(),
        'progress_metrics': metrics
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Progress prediction correlation (state-only): {metrics['correlation_state_only']:.4f}")
    
    return model, dataset, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Progress-Supervised c-VAE')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to zarr dataset')
    parser.add_argument('--output_dir', type=str, default='./progress_cvae_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='KL loss weight (lower for supervised setting)')
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='Progress supervision loss weight')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of networks')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--data_subset', type=int, default=None,
                        help='Use only first N samples (for debugging)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    # Train the model
    train_progress_cvae(**vars(args))


if __name__ == '__main__':
    main()