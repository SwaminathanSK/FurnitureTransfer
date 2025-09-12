"""
Test Out-of-Distribution Behavior of Progress c-VAE

This tests how the progress c-VAE behaves when given:
1. States/actions close to expert distribution (interpolation)
2. States/actions far from expert distribution (extrapolation)
3. Random noise inputs

This is crucial for understanding reward behavior during RL training.
"""

import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt
from cvae_progress import ProgressConditionalVAE, create_progress_labels


def test_ood_behavior(
    model_path: str = './progress_cvae_results/best_progress_cvae_model.pt',
    data_path: str = './robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/high/success.zarr'
):
    """Test out-of-distribution behavior."""
    
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
    
    # Load expert data for distribution analysis
    print(f"Loading expert data from {data_path}")
    dataset = zarr.open(data_path, mode='r')
    
    # Load full expert data
    robot_states = np.array(dataset['robot_state'][:])
    parts_poses = np.array(dataset['parts_poses'][:])
    expert_states = np.concatenate([robot_states, parts_poses], axis=-1)
    expert_actions = np.array(dataset['action/pos'][:])
    
    # Normalize using training stats
    data_stats = checkpoint.get('data_stats')
    if data_stats:
        expert_states_norm = (expert_states - data_stats['state_mean']) / data_stats['state_std']
        expert_actions_norm = (expert_actions - data_stats['action_mean']) / data_stats['action_std']
    else:
        expert_states_norm = expert_states
        expert_actions_norm = expert_actions
    
    # Compute expert distribution statistics
    state_mean = np.mean(expert_states_norm, axis=0)
    state_std = np.std(expert_states_norm, axis=0)
    action_mean = np.mean(expert_actions_norm, axis=0)
    action_std = np.std(expert_actions_norm, axis=0)
    
    print(f"Expert distribution stats computed from {len(expert_states)} samples")
    
    # Create test cases
    n_samples = 100
    
    # 1. In-distribution (interpolation): Sample from expert data with small noise
    print("\n1. Testing IN-DISTRIBUTION behavior (expert + small noise)...")
    
    # Pick random expert samples and add small noise
    expert_indices = np.random.choice(len(expert_states_norm), n_samples)
    in_dist_states = expert_states_norm[expert_indices] + 0.1 * np.random.randn(n_samples, expert_states_norm.shape[1])
    in_dist_actions = expert_actions_norm[expert_indices] + 0.1 * np.random.randn(n_samples, expert_actions_norm.shape[1])
    
    # 2. Near out-of-distribution: Expert distribution but with larger noise
    print("2. Testing NEAR OOD behavior (expert + medium noise)...")
    
    expert_indices = np.random.choice(len(expert_states_norm), n_samples)
    near_ood_states = expert_states_norm[expert_indices] + 0.5 * np.random.randn(n_samples, expert_states_norm.shape[1])
    near_ood_actions = expert_actions_norm[expert_indices] + 0.5 * np.random.randn(n_samples, expert_actions_norm.shape[1])
    
    # 3. Far out-of-distribution: Sample from different distribution
    print("3. Testing FAR OOD behavior (completely different distribution)...")
    
    # States/actions with different means and larger variance
    far_ood_states = np.random.randn(n_samples, expert_states_norm.shape[1]) * 2.0 + 1.0  # Different mean
    far_ood_actions = np.random.randn(n_samples, expert_actions_norm.shape[1]) * 2.0 + 1.0
    
    # 4. Extreme out-of-distribution: Very large values
    print("4. Testing EXTREME OOD behavior (very large values)...")
    
    extreme_ood_states = np.random.randn(n_samples, expert_states_norm.shape[1]) * 10.0
    extreme_ood_actions = np.random.randn(n_samples, expert_actions_norm.shape[1]) * 10.0
    
    # Test all cases
    test_cases = {
        'Expert (baseline)': (expert_states_norm[:n_samples], expert_actions_norm[:n_samples]),
        'In-Distribution': (in_dist_states, in_dist_actions),
        'Near OOD': (near_ood_states, near_ood_actions),
        'Far OOD': (far_ood_states, far_ood_actions),
        'Extreme OOD': (extreme_ood_states, extreme_ood_actions)
    }
    
    results = {}
    
    print("\nTesting progress predictions...")
    
    for name, (test_states, test_actions) in test_cases.items():
        # Convert to tensors
        states_tensor = torch.FloatTensor(test_states).to(device)
        actions_tensor = torch.FloatTensor(test_actions).to(device)
        
        with torch.no_grad():
            # Predict progress from state + action
            pred_progress_sa = model.get_progress_from_state_action(states_tensor, actions_tensor)
            # Predict progress from state only
            pred_progress_s = model.get_progress_from_state(states_tensor)
        
        progress_sa = pred_progress_sa.cpu().numpy().flatten()
        progress_s = pred_progress_s.cpu().numpy().flatten()
        
        results[name] = {
            'progress_sa': progress_sa,
            'progress_s': progress_s,
            'mean_sa': np.mean(progress_sa),
            'std_sa': np.std(progress_sa),
            'min_sa': np.min(progress_sa),
            'max_sa': np.max(progress_sa),
            'mean_s': np.mean(progress_s),
            'std_s': np.std(progress_s),
            'min_s': np.min(progress_s),
            'max_s': np.max(progress_s),
        }
        
        print(f"\n{name}:")
        print(f"  State+Action: μ={results[name]['mean_sa']:.3f}, σ={results[name]['std_sa']:.3f}, "
              f"range=[{results[name]['min_sa']:.3f}, {results[name]['max_sa']:.3f}]")
        print(f"  State Only:   μ={results[name]['mean_s']:.3f}, σ={results[name]['std_s']:.3f}, "
              f"range=[{results[name]['min_s']:.3f}, {results[name]['max_s']:.3f}]")
    
    # Visualize results
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Progress distributions (State+Action)
    plt.subplot(2, 3, 1)
    for name in test_cases.keys():
        plt.hist(results[name]['progress_sa'], bins=20, alpha=0.7, density=True, label=name)
    plt.xlabel('Progress (State+Action)')
    plt.ylabel('Density')
    plt.title('Progress Distributions (State+Action)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Progress distributions (State Only)
    plt.subplot(2, 3, 2)
    for name in test_cases.keys():
        plt.hist(results[name]['progress_s'], bins=20, alpha=0.7, density=True, label=name)
    plt.xlabel('Progress (State Only)')
    plt.ylabel('Density')
    plt.title('Progress Distributions (State Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Mean progress comparison
    plt.subplot(2, 3, 3)
    names = list(test_cases.keys())
    means_sa = [results[name]['mean_sa'] for name in names]
    means_s = [results[name]['mean_s'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, means_sa, width, label='State+Action', alpha=0.8)
    plt.bar(x + width/2, means_s, width, label='State Only', alpha=0.8)
    plt.xlabel('Test Case')
    plt.ylabel('Mean Progress')
    plt.title('Mean Progress by Distribution')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation comparison
    plt.subplot(2, 3, 4)
    stds_sa = [results[name]['std_sa'] for name in names]
    stds_s = [results[name]['std_s'] for name in names]
    
    plt.bar(x - width/2, stds_sa, width, label='State+Action', alpha=0.8)
    plt.bar(x + width/2, stds_s, width, label='State Only', alpha=0.8)
    plt.xlabel('Test Case')
    plt.ylabel('Progress Std Dev')
    plt.title('Progress Uncertainty by Distribution')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Range comparison (min/max)
    plt.subplot(2, 3, 5)
    mins_s = [results[name]['min_s'] for name in names]
    maxs_s = [results[name]['max_s'] for name in names]
    
    plt.bar(x, maxs_s, label='Max', alpha=0.8)
    plt.bar(x, mins_s, label='Min', alpha=0.8)
    plt.xlabel('Test Case')
    plt.ylabel('Progress Range (State Only)')
    plt.title('Progress Range by Distribution')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Individual sample scatter (Expert vs OOD)
    plt.subplot(2, 3, 6)
    plt.scatter(results['Expert (baseline)']['progress_sa'], 
                results['Expert (baseline)']['progress_s'], 
                alpha=0.6, label='Expert', s=20)
    plt.scatter(results['Far OOD']['progress_sa'], 
                results['Far OOD']['progress_s'], 
                alpha=0.6, label='Far OOD', s=20)
    plt.xlabel('Progress (State+Action)')
    plt.ylabel('Progress (State Only)')
    plt.title('State+Action vs State Only')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./ood_behavior_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis summary
    print(f"\n" + "="*60)
    print("OUT-OF-DISTRIBUTION BEHAVIOR ANALYSIS")
    print("="*60)
    
    expert_mean_s = results['Expert (baseline)']['mean_s']
    expert_std_s = results['Expert (baseline)']['std_s']
    
    for name in ['In-Distribution', 'Near OOD', 'Far OOD', 'Extreme OOD']:
        mean_diff = results[name]['mean_s'] - expert_mean_s
        std_ratio = results[name]['std_s'] / expert_std_s
        in_bounds = np.sum((results[name]['progress_s'] >= 0) & 
                          (results[name]['progress_s'] <= 1)) / len(results[name]['progress_s'])
        
        print(f"\n{name}:")
        print(f"  Mean shift from expert: {mean_diff:+.3f}")
        print(f"  Std ratio to expert: {std_ratio:.2f}x")
        print(f"  Samples in [0,1] bounds: {in_bounds:.1%}")
        
        if name == 'Far OOD' or name == 'Extreme OOD':
            if in_bounds < 0.5:
                print(f"  ⚠️  WARNING: Many samples outside [0,1] bounds!")
            if abs(mean_diff) > 0.5:
                print(f"  ⚠️  WARNING: Large mean shift from expert!")
    
    print(f"\n" + "="*60)
    print("IMPLICATIONS FOR RL TRAINING:")
    print("="*60)
    
    far_ood_mean = results['Far OOD']['mean_s']
    if far_ood_mean > expert_mean_s + 0.2:
        print("🔥 GOOD: Far OOD states get higher progress (could encourage exploration)")
    elif far_ood_mean < expert_mean_s - 0.2:
        print("❄️  BAD: Far OOD states get lower progress (might discourage exploration)")
    else:
        print("⚖️  NEUTRAL: Far OOD states get similar progress to expert")
    
    extreme_in_bounds = np.sum((results['Extreme OOD']['progress_s'] >= 0) & 
                              (results['Extreme OOD']['progress_s'] <= 1)) / len(results['Extreme OOD']['progress_s'])
    
    if extreme_in_bounds < 0.3:
        print("⚠️  CONCERN: Extreme OOD gives unbounded progress values")
        print("   Consider clipping progress rewards to [0,1] in RL training")
    else:
        print("✅ GOOD: Even extreme inputs stay reasonably bounded")
    
    print(f"\nResults saved to: ood_behavior_analysis.png")
    
    return results


if __name__ == '__main__':
    results = test_ood_behavior()