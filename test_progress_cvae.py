"""
Test the trained progress c-VAE model.

Quick script to load the model and see how well it learned to predict task progress.
"""

import torch
import numpy as np
import zarr
from cvae_progress import ProgressConditionalVAE, create_progress_labels
import matplotlib.pyplot as plt


def test_progress_cvae(
    model_path: str = './progress_cvae_results/best_progress_cvae_model.pt',
    data_path: str = './robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/high/success.zarr'
):
    """Test the trained progress c-VAE."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint['model_config']
    model = ProgressConditionalVAE(
        state_dim=model_config['state_dim'],
        action_dim=model_config['action_dim'],
        latent_dim=model_config['latent_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training config: {checkpoint.get('training_config', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Load test data
    print(f"Loading test data from {data_path}")
    dataset = zarr.open(data_path, mode='r')
    
    # Use a subset for testing
    subset_size = 2000
    robot_states = np.array(dataset['robot_state'][:subset_size])
    parts_poses = np.array(dataset['parts_poses'][:subset_size])
    states = np.concatenate([robot_states, parts_poses], axis=-1)
    actions = np.array(dataset['action/pos'][:subset_size])
    
    episode_ends = np.array(dataset['episode_ends'][:])
    
    # Create progress labels for the full dataset, then take subset
    total_samples = len(dataset['robot_state'])
    full_progress_labels = create_progress_labels(episode_ends, total_samples)
    progress_labels = full_progress_labels[:subset_size]
    
    print(f"Test data: {len(states)} samples")
    
    # Normalize data (using stored stats)
    data_stats = checkpoint.get('data_stats')
    if data_stats:
        states = (states - data_stats['state_mean']) / data_stats['state_std']
        actions = (actions - data_stats['action_mean']) / data_stats['action_std']
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.FloatTensor(actions).to(device)
    progress_tensor = torch.FloatTensor(progress_labels).unsqueeze(1)
    
    # Test progress prediction
    print("Testing progress prediction...")
    with torch.no_grad():
        # Predict progress from state + action
        pred_progress_sa = model.get_progress_from_state_action(states_tensor, actions_tensor)
        
        # Predict progress from state only
        pred_progress_s = model.get_progress_from_state(states_tensor)
        
        # Test reconstruction
        outputs = model(states_tensor[:100], actions_tensor[:100])  # Test on smaller batch
    
    # Convert to numpy
    pred_sa = pred_progress_sa.cpu().numpy().flatten()
    pred_s = pred_progress_s.cpu().numpy().flatten()
    true_progress = progress_labels
    
    # Compute metrics
    correlation_sa = np.corrcoef(true_progress, pred_sa)[0, 1]
    correlation_s = np.corrcoef(true_progress, pred_s)[0, 1]
    mse_sa = np.mean((pred_sa - true_progress)**2)
    mse_s = np.mean((pred_s - true_progress)**2)
    
    print(f"\nProgress Prediction Results:")
    print(f"  State + Action:")
    print(f"    Correlation: {correlation_sa:.4f}")
    print(f"    MSE: {mse_sa:.6f}")
    print(f"    Range: [{pred_sa.min():.3f}, {pred_sa.max():.3f}]")
    
    print(f"  State Only:")
    print(f"    Correlation: {correlation_s:.4f}")
    print(f"    MSE: {mse_s:.6f}")
    print(f"    Range: [{pred_s.min():.3f}, {pred_s.max():.3f}]")
    
    print(f"  True Progress:")
    print(f"    Range: [{true_progress.min():.3f}, {true_progress.max():.3f}]")
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Correlation plot (State + Action)
    plt.subplot(1, 3, 1)
    plt.scatter(true_progress, pred_sa, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('True Progress')
    plt.ylabel('Predicted (State+Action)')
    plt.title(f'S+A: Corr={correlation_sa:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Correlation plot (State Only)
    plt.subplot(1, 3, 2)
    plt.scatter(true_progress, pred_s, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('True Progress')
    plt.ylabel('Predicted (State Only)')
    plt.title(f'State Only: Corr={correlation_s:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Progress over time for first episode
    plt.subplot(1, 3, 3)
    first_ep_end = min(episode_ends[0], len(true_progress))
    plt.plot(true_progress[:first_ep_end], 'b-', label='True', linewidth=2)
    plt.plot(pred_sa[:first_ep_end], 'r--', label='Pred (S+A)', linewidth=2)
    plt.plot(pred_s[:first_ep_end], 'g:', label='Pred (S)', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Progress')
    plt.title('Progress Over Time (Episode 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./progress_cvae_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTest visualization saved to: progress_cvae_test_results.png")
    
    # Success criteria
    if correlation_s > 0.7:
        print(f"\n✅ SUCCESS! State-only progress prediction correlation > 0.7")
        print(f"   This model can be used as a dense reward signal!")
    else:
        print(f"\n⚠️  Warning: State-only correlation is {correlation_s:.3f} < 0.7")
        print(f"   May need more training or different hyperparameters.")
    
    return {
        'correlation_state_action': correlation_sa,
        'correlation_state_only': correlation_s,
        'mse_state_action': mse_sa,
        'mse_state_only': mse_s
    }


if __name__ == '__main__':
    results = test_progress_cvae()