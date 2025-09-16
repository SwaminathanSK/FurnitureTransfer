"""
Create Balanced Dataset with Negative Examples

This augments the expert demonstrations with negative examples to fix OOD behavior:
1. Expert demonstrations (positive examples, progress 0→1)
2. Random actions (negative examples, low progress ~0.1)
3. Corrupted expert actions (negative examples, progress ~0)
4. Early terminations (truncated progress)

This should fix the overestimation problem for OOD states.
"""

import numpy as np
import torch
import zarr
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def generate_random_trajectory(
    initial_state: np.ndarray,
    length: int,
    state_bounds: dict,
    action_bounds: dict,
    noise_level: float = 1.0
) -> tuple:
    """Generate a random trajectory starting from initial_state."""
    
    states = [initial_state.copy()]
    actions = []
    
    current_state = initial_state.copy()
    
    for step in range(length):
        # Generate random action within bounds
        action = np.random.uniform(
            action_bounds['min'], 
            action_bounds['max'], 
            size=action_bounds['dim']
        )
        actions.append(action)
        
        # Simulate simple state evolution (random walk with bounds)
        # This is a crude approximation - in reality you'd use the actual environment
        state_noise = np.random.normal(0, noise_level, size=current_state.shape)
        next_state = current_state + 0.1 * state_noise
        
        # Keep states within reasonable bounds
        next_state = np.clip(next_state, state_bounds['min'], state_bounds['max'])
        
        states.append(next_state)
        current_state = next_state
    
    return np.array(states[:-1]), np.array(actions)  # Return matching state-action pairs


def corrupt_expert_actions(
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    corruption_ratio: float = 0.5,
    noise_scale: float = 2.0
) -> tuple:
    """Corrupt expert actions to create negative examples."""
    
    corrupted_actions = expert_actions.copy()
    n_corrupt = int(len(expert_actions) * corruption_ratio)
    
    # Randomly select actions to corrupt
    corrupt_indices = np.random.choice(len(expert_actions), n_corrupt, replace=False)
    
    # Add large noise to selected actions
    noise = np.random.normal(0, noise_scale, size=(n_corrupt, expert_actions.shape[1]))
    corrupted_actions[corrupt_indices] += noise
    
    return expert_states, corrupted_actions, corrupt_indices


def create_balanced_dataset(
    expert_data_path: str,
    output_path: str,
    negative_ratio: float = 0.5,  # Fraction of negative examples
    random_traj_length_range: tuple = (50, 200),
    data_subset: int = None
):
    """Create balanced dataset with positive and negative examples."""
    
    print(f"Loading expert data from {expert_data_path}")
    dataset = zarr.open(expert_data_path, mode='r')
    
    # Load expert data
    if data_subset:
        robot_states = np.array(dataset['robot_state'][:data_subset])
        parts_poses = np.array(dataset['parts_poses'][:data_subset])
        expert_actions = np.array(dataset['action/pos'][:data_subset])
        episode_ends = dataset['episode_ends'][:]
        # Adjust episode_ends for subset
        episode_ends = episode_ends[episode_ends <= data_subset]
    else:
        robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])
        expert_actions = np.array(dataset['action/pos'][:])
        episode_ends = np.array(dataset['episode_ends'][:])
    
    expert_states = np.concatenate([robot_states, parts_poses], axis=-1)
    
    print(f"Loaded {len(expert_states)} expert examples")
    print(f"Episodes: {len(episode_ends)}")
    
    # Compute data bounds for generating negative examples
    state_bounds = {
        'min': np.percentile(expert_states, 5, axis=0),   # 5th percentile
        'max': np.percentile(expert_states, 95, axis=0),  # 95th percentile
        'dim': expert_states.shape[1]
    }
    
    action_bounds = {
        'min': np.percentile(expert_actions, 5, axis=0),
        'max': np.percentile(expert_actions, 95, axis=0),
        'dim': expert_actions.shape[1]
    }
    
    # 1. Expert trajectories (positive examples)
    print("\n1. Processing expert trajectories...")
    
    expert_progress = []
    start_idx = 0
    for episode_idx, end_idx in enumerate(episode_ends):
        episode_length = end_idx - start_idx
        # Create progress from 0 to 1 for this episode
        episode_progress = np.linspace(0, 1, episode_length, endpoint=False)
        expert_progress.extend(episode_progress)
        start_idx = end_idx
    
    expert_progress = np.array(expert_progress)
    
    # 2. Generate random trajectories (negative examples)
    print("\n2. Generating random trajectories...")
    
    n_negative = int(len(expert_states) * negative_ratio)
    n_random_trajs = max(10, n_negative // 3)  # At least 10 random trajectories
    
    random_states = []
    random_actions = []
    random_progress = []
    
    for traj_idx in tqdm(range(n_random_trajs), desc="Random trajectories"):
        # Pick random starting state from expert distribution
        start_state_idx = np.random.choice(len(expert_states))
        start_state = expert_states[start_state_idx]
        
        # Random trajectory length
        traj_length = np.random.randint(*random_traj_length_range)
        
        # Generate random trajectory
        traj_states, traj_actions = generate_random_trajectory(
            start_state, traj_length, state_bounds, action_bounds
        )
        
        # Ensure same length for state-action pairs
        assert len(traj_states) == len(traj_actions), f"Length mismatch: {len(traj_states)} states vs {len(traj_actions)} actions"
        
        # Low progress for random actions (0 to 0.2)
        traj_progress = np.linspace(0, 0.2, len(traj_states), endpoint=False)
        
        random_states.extend(traj_states)
        random_actions.extend(traj_actions)
        random_progress.extend(traj_progress)
    
    random_states = np.array(random_states)
    random_actions = np.array(random_actions)
    random_progress = np.array(random_progress)
    
    print(f"Generated {len(random_states)} random examples")
    
    # 3. Corrupted expert actions (negative examples)
    print("\n3. Creating corrupted expert actions...")
    
    n_corrupted = min(n_negative - len(random_states), len(expert_states) // 2)
    if n_corrupted > 0:
        # Select subset of expert data to corrupt
        corrupt_indices = np.random.choice(len(expert_states), n_corrupted, replace=False)
        corrupt_states = expert_states[corrupt_indices]
        corrupt_actions_clean = expert_actions[corrupt_indices]
        
        # Corrupt the actions
        _, corrupt_actions, _ = corrupt_expert_actions(
            corrupt_states, corrupt_actions_clean, 
            corruption_ratio=1.0,  # Corrupt all selected actions
            noise_scale=3.0        # Heavy corruption
        )
        
        # Very low progress for corrupted actions (0 to 0.1)
        corrupt_progress = np.random.uniform(0, 0.1, size=n_corrupted)
    else:
        corrupt_states = np.empty((0, expert_states.shape[1]))
        corrupt_actions = np.empty((0, expert_actions.shape[1]))
        corrupt_progress = np.empty(0)
    
    print(f"Generated {len(corrupt_states)} corrupted examples")
    
    # 4. Combine all data
    print("\n4. Combining all data...")
    
    all_states = np.vstack([expert_states, random_states, corrupt_states])
    all_actions = np.vstack([expert_actions, random_actions, corrupt_actions])
    all_progress = np.concatenate([expert_progress, random_progress, corrupt_progress])
    
    # Create labels for data source
    data_sources = (['expert'] * len(expert_states) + 
                   ['random'] * len(random_states) + 
                   ['corrupt'] * len(corrupt_states))
    
    print(f"\nDataset composition:")
    print(f"  Expert examples: {len(expert_states)} ({len(expert_states)/len(all_states)*100:.1f}%)")
    print(f"  Random examples: {len(random_states)} ({len(random_states)/len(all_states)*100:.1f}%)")
    print(f"  Corrupt examples: {len(corrupt_states)} ({len(corrupt_states)/len(all_states)*100:.1f}%)")
    print(f"  Total examples: {len(all_states)}")
    
    # Progress distribution analysis
    expert_progress_stats = f"μ={np.mean(expert_progress):.3f}, σ={np.std(expert_progress):.3f}"
    if len(random_progress) > 0:
        random_progress_stats = f"μ={np.mean(random_progress):.3f}, σ={np.std(random_progress):.3f}"
    else:
        random_progress_stats = "N/A"
    if len(corrupt_progress) > 0:
        corrupt_progress_stats = f"μ={np.mean(corrupt_progress):.3f}, σ={np.std(corrupt_progress):.3f}"
    else:
        corrupt_progress_stats = "N/A"
    
    print(f"\nProgress distribution:")
    print(f"  Expert: {expert_progress_stats}")
    print(f"  Random: {random_progress_stats}")
    print(f"  Corrupt: {corrupt_progress_stats}")
    
    # 5. Save balanced dataset
    print(f"\n5. Saving balanced dataset to {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as zarr dataset
    balanced_dataset = zarr.open(str(output_path), mode='w')
    
    # Save data
    balanced_dataset.create_dataset('states', data=all_states, chunks=True)
    balanced_dataset.create_dataset('actions', data=all_actions, chunks=True)
    balanced_dataset.create_dataset('progress', data=all_progress, chunks=True)
    balanced_dataset.create_dataset('data_sources', data=data_sources, chunks=True)
    
    # Save metadata
    balanced_dataset.attrs['expert_count'] = len(expert_states)
    balanced_dataset.attrs['random_count'] = len(random_states)
    balanced_dataset.attrs['corrupt_count'] = len(corrupt_states)
    balanced_dataset.attrs['total_count'] = len(all_states)
    balanced_dataset.attrs['negative_ratio'] = negative_ratio
    
    print(f"Saved {len(all_states)} samples to {output_path}")
    
    # 6. Create visualization
    print("\n6. Creating visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Progress distribution by data source
    plt.subplot(2, 3, 1)
    plt.hist(expert_progress, bins=50, alpha=0.7, label=f'Expert (n={len(expert_progress)})', density=True)
    if len(random_progress) > 0:
        plt.hist(random_progress, bins=50, alpha=0.7, label=f'Random (n={len(random_progress)})', density=True)
    if len(corrupt_progress) > 0:
        plt.hist(corrupt_progress, bins=50, alpha=0.7, label=f'Corrupt (n={len(corrupt_progress)})', density=True)
    plt.xlabel('Progress')
    plt.ylabel('Density')
    plt.title('Progress Distribution by Data Source')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Progress vs time for different sources
    plt.subplot(2, 3, 2)
    expert_indices = np.where(np.array(data_sources) == 'expert')[0]
    if len(random_states) > 0:
        random_indices = np.where(np.array(data_sources) == 'random')[0]
        plt.scatter(random_indices[:1000], all_progress[random_indices[:1000]], 
                   alpha=0.5, s=1, label='Random', color='orange')
    
    plt.scatter(expert_indices[::10], all_progress[expert_indices[::10]], 
               alpha=0.7, s=1, label='Expert', color='blue')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Progress')
    plt.title('Progress Values Across Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: State space coverage (PCA)
    from sklearn.decomposition import PCA
    
    plt.subplot(2, 3, 3)
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(all_states)
    
    expert_mask = np.array(data_sources) == 'expert'
    random_mask = np.array(data_sources) == 'random'
    corrupt_mask = np.array(data_sources) == 'corrupt'
    
    plt.scatter(states_2d[expert_mask, 0], states_2d[expert_mask, 1], 
               alpha=0.6, s=1, label='Expert', color='blue')
    if len(random_states) > 0:
        plt.scatter(states_2d[random_mask, 0], states_2d[random_mask, 1], 
                   alpha=0.6, s=1, label='Random', color='orange')
    if len(corrupt_states) > 0:
        plt.scatter(states_2d[corrupt_mask, 0], states_2d[corrupt_mask, 1], 
                   alpha=0.6, s=1, label='Corrupt', color='red')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('State Space Coverage (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Action space coverage
    plt.subplot(2, 3, 4)
    pca_actions = PCA(n_components=2)
    actions_2d = pca_actions.fit_transform(all_actions)
    
    plt.scatter(actions_2d[expert_mask, 0], actions_2d[expert_mask, 1], 
               alpha=0.6, s=1, label='Expert', color='blue')
    if len(random_actions) > 0:
        plt.scatter(actions_2d[random_mask, 0], actions_2d[random_mask, 1], 
                   alpha=0.6, s=1, label='Random', color='orange')
    if len(corrupt_actions) > 0:
        plt.scatter(actions_2d[corrupt_mask, 0], actions_2d[corrupt_mask, 1], 
                   alpha=0.6, s=1, label='Corrupt', color='red')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Action Space Coverage (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Data composition pie chart
    plt.subplot(2, 3, 5)
    sizes = [len(expert_states), len(random_states), len(corrupt_states)]
    labels = ['Expert', 'Random', 'Corrupt']
    colors = ['blue', 'orange', 'red']
    
    # Remove zero entries
    sizes_nonzero = [s for s in sizes if s > 0]
    labels_nonzero = [l for s, l in zip(sizes, labels) if s > 0]
    colors_nonzero = [c for s, c in zip(sizes, colors) if s > 0]
    
    plt.pie(sizes_nonzero, labels=labels_nonzero, colors=colors_nonzero, autopct='%1.1f%%')
    plt.title('Dataset Composition')
    
    # Plot 6: Progress statistics summary
    plt.subplot(2, 3, 6)
    progress_by_source = {
        'Expert': expert_progress,
        'Random': random_progress if len(random_progress) > 0 else np.array([]),
        'Corrupt': corrupt_progress if len(corrupt_progress) > 0 else np.array([])
    }
    
    means = [np.mean(prog) if len(prog) > 0 else 0 for prog in progress_by_source.values()]
    stds = [np.std(prog) if len(prog) > 0 else 0 for prog in progress_by_source.values()]
    source_names = list(progress_by_source.keys())
    
    x = np.arange(len(source_names))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    plt.bar(x + width/2, stds, width, label='Std', alpha=0.8)
    plt.xlabel('Data Source')
    plt.ylabel('Progress Value')
    plt.title('Progress Statistics by Source')
    plt.xticks(x, source_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = output_path.parent / 'balanced_dataset_analysis.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Analysis visualization saved to: {viz_path}")
    
    return {
        'total_samples': len(all_states),
        'expert_samples': len(expert_states),
        'random_samples': len(random_states),
        'corrupt_samples': len(corrupt_states),
        'expert_progress_mean': np.mean(expert_progress),
        'random_progress_mean': np.mean(random_progress) if len(random_progress) > 0 else 0,
        'corrupt_progress_mean': np.mean(corrupt_progress) if len(corrupt_progress) > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Create balanced dataset with negative examples')
    parser.add_argument('--expert_data_path', type=str, 
                        default='./robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/high/success.zarr',
                        help='Path to expert data')
    parser.add_argument('--output_path', type=str, 
                        default='./balanced_dataset/one_leg_balanced.zarr',
                        help='Output path for balanced dataset')
    parser.add_argument('--negative_ratio', type=float, default=0.5,
                        help='Fraction of negative examples (0.5 = equal positive/negative)')
    parser.add_argument('--data_subset', type=int, default=None,
                        help='Use only first N samples from expert data')
    parser.add_argument('--random_traj_min', type=int, default=50,
                        help='Minimum length for random trajectories')
    parser.add_argument('--random_traj_max', type=int, default=200,
                        help='Maximum length for random trajectories')
    
    args = parser.parse_args()
    
    results = create_balanced_dataset(
        expert_data_path=args.expert_data_path,
        output_path=args.output_path,
        negative_ratio=args.negative_ratio,
        random_traj_length_range=(args.random_traj_min, args.random_traj_max),
        data_subset=args.data_subset
    )
    
    print(f"\n" + "="*50)
    print("BALANCED DATASET CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"Total samples: {results['total_samples']}")
    print(f"Expert progress mean: {results['expert_progress_mean']:.3f}")
    print(f"Random progress mean: {results['random_progress_mean']:.3f}")
    print(f"Corrupt progress mean: {results['corrupt_progress_mean']:.3f}")
    print(f"\nNext step: Retrain progress c-VAE on this balanced dataset!")


if __name__ == '__main__':
    main()