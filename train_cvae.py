"""
Training script for Conditional VAE on One-Leg Assembly Task

This script trains a c-VAE to learn 1D latent actions from state-action pairs
collected from the one-leg assembly task. The learned latent representation
can be used as:
1. A reward signal indicating task progress
2. A compact action representation for RL
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
import wandb
from datetime import datetime
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvae_latent_actions import ConditionalVAE, CVAETrainer, visualize_latent_space


class OneLegStateActionDataset(Dataset):
    """
    Dataset for One-Leg assembly state-action pairs.
    
    Loads data from zarr format and provides state-action pairs for c-VAE training.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        normalize: bool = True,
        data_subset: int = None,
        action_type: str = 'pos'  # 'pos' or 'delta'
    ):
        """
        Args:
            dataset_path: path to zarr dataset
            normalize: whether to normalize states and actions
            data_subset: if not None, use only first N samples
            action_type: which action type to use ('pos' or 'delta')
        """
        # Load zarr dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        print(f"Loading dataset from {dataset_path}")
        dataset = zarr.open(dataset_path, mode='r')
        
        # Load state-action data
        if 'robot_state' in dataset and 'parts_poses' in dataset and f'action/{action_type}' in dataset:
            # New format with separate robot_state and parts_poses
            robot_states = np.array(dataset['robot_state'][:])
            parts_poses = np.array(dataset['parts_poses'][:])
            self.states = np.concatenate([robot_states, parts_poses], axis=-1)
            self.actions = np.array(dataset[f'action/{action_type}'][:])
            
            # Also load episode information and rewards for analysis
            self.episode_ends = np.array(dataset['episode_ends'][:])
            self.rewards = np.array(dataset['reward'][:])
            self.success = np.array(dataset['success'][:])
            
        elif 'observations' in dataset and 'actions' in dataset:
            # Legacy format
            self.states = np.array(dataset['observations'][:])
            self.actions = np.array(dataset['actions'][:])
            self.episode_ends = np.array(dataset['episode_ends'][:])
            self.rewards = None
            self.success = None
        else:
            raise ValueError("Dataset format not recognized. Expected 'robot_state'+'parts_poses'+'action/pos' or 'observations'+'actions'")
        
        # Apply data subset if specified
        if data_subset is not None:
            self.states = self.states[:data_subset]
            self.actions = self.actions[:data_subset]
            if self.rewards is not None:
                self.rewards = self.rewards[:data_subset]
        
        print(f"Loaded {len(self.states)} state-action pairs")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
        
        # Compute normalization statistics
        if normalize:
            self.state_mean = np.mean(self.states, axis=0)
            self.state_std = np.std(self.states, axis=0) + 1e-8  # Avoid division by zero
            
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
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
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
    
    def denormalize_action(self, normalized_action):
        """Denormalize actions back to original scale."""
        if self.normalize:
            return normalized_action * self.action_std + self.action_mean
        else:
            return normalized_action
    
    def denormalize_state(self, normalized_state):
        """Denormalize states back to original scale."""
        if self.normalize:
            return normalized_state * self.state_std + self.state_mean
        else:
            return normalized_state


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
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
    axes[1, 0].plot(epochs, [l['kl_loss'] for l in train_losses], 'b-', label='Train')
    axes[1, 0].plot(epochs, [l['kl_loss'] for l in val_losses], 'r-', label='Val')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Latent statistics
    axes[1, 1].plot(epochs, [l['latent_mean'] for l in train_losses], 'b-', label='Mean')
    axes[1, 1].plot(epochs, [l['latent_std'] for l in train_losses], 'g-', label='Std')
    axes[1, 1].set_title('Latent Statistics (Train)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def train_cvae(
    data_path: str,
    output_dir: str = "./cvae_results",
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta_schedule: str = 'linear',
    beta_max: float = 1.0,
    hidden_dim: int = 256,
    num_layers: int = 3,
    data_subset: int = None,
    use_wandb: bool = False,
    val_split: float = 0.1,
    device: str = None
):
    """
    Train the conditional VAE model.
    
    Args:
        data_path: path to the zarr dataset
        output_dir: directory to save results
        batch_size: training batch size
        epochs: number of training epochs
        learning_rate: learning rate for optimizer
        beta_schedule: schedule for beta parameter ('constant', 'linear', 'cyclical')
        beta_max: maximum beta value
        hidden_dim: hidden dimension of the networks
        num_layers: number of layers in encoder/decoder
        data_subset: if not None, use only first N samples
        use_wandb: whether to log to wandb
        val_split: fraction of data to use for validation
        device: device to use for training
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="cvae-latent-actions",
            config={
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'beta_schedule': beta_schedule,
                'beta_max': beta_max,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'data_subset': data_subset,
            }
        )
    
    # Load dataset
    dataset = OneLegStateActionDataset(
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
    
    model = ConditionalVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    trainer = CVAETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        beta_schedule=beta_schedule,
        beta_max=beta_max
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Get beta for this epoch
        beta = trainer.get_beta(epoch, epochs)
        
        # Training
        model.train()
        train_loss_epoch = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (states, actions) in enumerate(train_pbar):
            states = states.to(device)
            actions = actions.to(device)
            
            loss_dict = trainer.train_step(states, actions, beta=beta)
            train_loss_epoch.append(loss_dict)
            
            # Update progress bar
            train_pbar.set_postfix({
                'total': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}",
                'beta': f"{beta:.3f}"
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
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                
                loss_dict = trainer.evaluate(states, actions)
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
              f"KL: {avg_train_loss['kl_loss']:.4f}")
        print(f"  Val   - Total: {avg_val_loss['total_loss']:.4f}, "
              f"Recon: {avg_val_loss['recon_loss']:.4f}, "
              f"KL: {avg_val_loss['kl_loss']:.4f}")
        print(f"  Beta: {beta:.3f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'beta': beta,
                'train/total_loss': avg_train_loss['total_loss'],
                'train/recon_loss': avg_train_loss['recon_loss'],
                'train/kl_loss': avg_train_loss['kl_loss'],
                'val/total_loss': avg_val_loss['total_loss'],
                'val/recon_loss': avg_val_loss['recon_loss'],
                'val/kl_loss': avg_val_loss['kl_loss'],
            })
        
        # Save best model
        if avg_val_loss['total_loss'] < best_val_loss:
            best_val_loss = avg_val_loss['total_loss']
            model_path = output_dir / 'best_cvae_model.pt'
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
                'data_stats': dataset.get_stats()
            }, model_path)
            print(f"  Saved best model to {model_path}")
    
    # Final visualizations and save
    print("Training completed!")
    
    # Plot training curves
    curves_path = output_dir / 'training_curves.png'
    plot_training_curves(train_losses, val_losses, save_path=curves_path)
    
    # Visualize latent space on validation data
    val_states = []
    val_actions = []
    for states, actions in val_loader:
        val_states.append(states)
        val_actions.append(actions)
    
    val_states = torch.cat(val_states, dim=0).to(device)
    val_actions = torch.cat(val_actions, dim=0).to(device)
    
    latent_viz_path = output_dir / 'latent_space_visualization.png'
    visualize_latent_space(
        model, val_states, val_actions, 
        save_path=latent_viz_path,
        title="Learned 1D Latent Space for One-Leg Assembly"
    )
    
    # Save final model
    final_model_path = output_dir / 'final_cvae_model.pt'
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
        'data_stats': dataset.get_stats()
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if use_wandb:
        wandb.finish()
    
    return model, dataset


def main():
    parser = argparse.ArgumentParser(description='Train Conditional VAE for One-Leg Assembly')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to zarr dataset')
    parser.add_argument('--output_dir', type=str, default='./cvae_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        choices=['constant', 'linear', 'cyclical', 'warmup'],
                        help='Beta scheduling strategy')
    parser.add_argument('--beta_max', type=float, default=1.0,
                        help='Maximum beta value')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of networks')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--data_subset', type=int, default=None,
                        help='Use only first N samples (for debugging)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    # Train the model
    train_cvae(**vars(args))


if __name__ == '__main__':
    main()