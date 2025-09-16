"""
Training script that integrates INAC (In-Sample Softmax Offline RL) with CVAE progress rewards

This script combines:
1. A trained Progress-Supervised CVAE model that outputs 1D progress latents (0-1)
2. INAC offline RL algorithm that constrains training to actions in expert dataset
3. Uses CVAE progress latents as reward signals for RL training

The CVAE progress latent serves as a reward signal representing task completion progress,
while INAC ensures the policy stays close to expert demonstrations.
"""

import os
import sys
import torch
import numpy as np
import zarr
from pathlib import Path
import argparse
from typing import Dict, Tuple
import pickle
import copy

# Add INAC codebase to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'INAC_MLRC_24'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from INAC_MLRC_24.core.agent.in_sample import InSampleAC
from INAC_MLRC_24.core.utils import torch_utils, logger, run_funcs
from cvae_progress import ProgressConditionalVAE


class CVAERewardEnvironment:
    """
    Custom environment wrapper that uses CVAE progress latents as rewards.
    """

    def __init__(
        self,
        cvae_model_path: str,
        original_dataset_path: str,
        device: str = 'cuda'
    ):
        self.device = device

        # Load trained CVAE model
        print(f"Loading CVAE model from {cvae_model_path}")
        checkpoint = torch.load(cvae_model_path, map_location=device)

        model_config = checkpoint['model_config']
        self.cvae = ProgressConditionalVAE(
            state_dim=model_config['state_dim'],
            action_dim=model_config['action_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers']
        )
        self.cvae.load_state_dict(checkpoint['model_state_dict'])
        self.cvae.to(device)
        self.cvae.eval()

        # Load normalization statistics
        self.data_stats = checkpoint.get('data_stats', None)

        # Load original dataset to get state/action structure
        print(f"Loading dataset structure from {original_dataset_path}")
        dataset = zarr.open(original_dataset_path, mode='r')

        robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])
        sample_state = np.concatenate([robot_states[0], parts_poses[0]])
        sample_action = np.array(dataset['action/pos'][0])

        self.state_dim = len(sample_state)
        self.action_dim = len(sample_action)

        print(f"Environment dimensions: state={self.state_dim}, action={self.action_dim}")

    def normalize_state_action(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize state and action using CVAE training statistics."""
        if self.data_stats is not None:
            state = (state - self.data_stats['state_mean']) / self.data_stats['state_std']
            action = (action - self.data_stats['action_mean']) / self.data_stats['action_std']
        return state, action

    def get_cvae_progress_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Get progress reward from CVAE model.
        Uses the CVAE's predicted progress latent as reward signal.
        """
        # Normalize inputs
        norm_state, norm_action = self.normalize_state_action(state, action)

        # Convert to tensors
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get progress prediction from CVAE
            progress_pred = self.cvae.predict_progress(state_tensor)
            progress_reward = progress_pred.cpu().numpy()[0, 0]  # Extract scalar value

        return float(progress_reward)


class FurnitureAssemblyDataset:
    """
    Dataset class that loads furniture assembly data and provides CVAE-based rewards.
    """

    def __init__(
        self,
        dataset_path: str,
        cvae_model_path: str,
        action_type: str = 'pos',
        max_episodes: int = None
    ):
        self.dataset_path = dataset_path
        self.action_type = action_type

        # Load dataset
        print(f"Loading furniture assembly dataset from {dataset_path}")
        dataset = zarr.open(dataset_path, mode='r')

        # Load state-action data
        robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])
        self.states = np.concatenate([robot_states, parts_poses], axis=-1)
        self.actions = np.array(dataset[f'action/{action_type}'][:])

        # Load episode information
        self.episode_ends = np.array(dataset['episode_ends'][:])
        self.success = np.array(dataset['success'][:])

        # Initialize CVAE reward environment
        self.cvae_env = CVAERewardEnvironment(
            cvae_model_path=cvae_model_path,
            original_dataset_path=dataset_path
        )

        # Limit episodes if specified
        if max_episodes is not None:
            valid_episodes = min(max_episodes, len(self.episode_ends))
            max_idx = self.episode_ends[valid_episodes - 1]
            self.states = self.states[:max_idx]
            self.actions = self.actions[:max_idx]
            self.episode_ends = self.episode_ends[:valid_episodes]
            self.success = self.success[:valid_episodes]

        print(f"Dataset loaded: {len(self.states)} transitions, {len(self.episode_ends)} episodes")
        print(f"State dim: {self.states.shape[1]}, Action dim: {self.actions.shape[1]}")

    def compute_cvae_rewards(self) -> np.ndarray:
        """
        Compute CVAE-based rewards for all state-action pairs in the dataset.
        """
        print("Computing CVAE progress rewards for all transitions...")
        rewards = np.zeros(len(self.states))

        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(self.states), batch_size):
            end_idx = min(i + batch_size, len(self.states))

            for j in range(i, end_idx):
                rewards[j] = self.cvae_env.get_cvae_progress_reward(
                    self.states[j], self.actions[j]
                )

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(self.states)} transitions")

        print(f"CVAE rewards computed - min: {rewards.min():.3f}, max: {rewards.max():.3f}, mean: {rewards.mean():.3f}")
        return rewards

    def to_d4rl_format(self) -> Dict:
        """
        Convert to D4RL-style format for INAC training.
        """
        # Compute CVAE rewards
        rewards = self.compute_cvae_rewards()

        # Create next states (shifted by 1)
        next_states = np.zeros_like(self.states)
        terminals = np.zeros(len(self.states), dtype=bool)

        # Handle episode boundaries
        episode_start = 0
        for episode_end in self.episode_ends:
            # Next states within episode
            if episode_end - episode_start > 1:
                next_states[episode_start:episode_end-1] = self.states[episode_start+1:episode_end]

            # Terminal state (last state of episode points to itself)
            next_states[episode_end-1] = self.states[episode_end-1]
            terminals[episode_end-1] = True

            episode_start = episode_end

        # D4RL format dictionary
        d4rl_data = {
            'observations': self.states.astype(np.float32),
            'actions': self.actions.astype(np.float32),
            'rewards': rewards.astype(np.float32),
            'next_observations': next_states.astype(np.float32),
            'terminals': terminals
        }

        print(f"Converted to D4RL format:")
        print(f"  Observations: {d4rl_data['observations'].shape}")
        print(f"  Actions: {d4rl_data['actions'].shape}")
        print(f"  Rewards: {d4rl_data['rewards'].shape}")
        print(f"  Terminals: {d4rl_data['terminals'].sum()} terminal states")

        return d4rl_data


class FurnitureAssemblyEnvironment:
    """
    Custom environment for furniture assembly task.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = self._create_dummy_space(state_dim)
        self.action_space = self._create_dummy_space(action_dim)

    def _create_dummy_space(self, dim):
        """Create dummy gym space for compatibility."""
        class DummySpace:
            def __init__(self, dim):
                self.shape = (dim,)
                self.low = -np.inf * np.ones(dim)
                self.high = np.inf * np.ones(dim)
        return DummySpace(dim)

    def reset(self):
        return np.zeros(self.state_dim)

    def step(self, action):
        # Dummy implementation - not used in offline RL
        return np.zeros(self.state_dim), 0.0, False, {}


class INACFurnitureConfig:
    """Configuration for INAC training on furniture assembly."""

    def __init__(self, **kwargs):
        # Default values
        self.seed = 0
        self.env_name = 'FurnitureAssembly'
        self.dataset = 'expert'
        self.discrete_control = 0
        self.state_dim = 58  # Will be updated based on data
        self.action_dim = 7   # Will be updated based on data
        self.tau = 0.01
        self.max_steps = 100000
        self.log_interval = 5000
        self.learning_rate = 3e-4
        self.hidden_units = 256
        self.batch_size = 256
        self.timeout = 1000
        self.gamma = 0.99
        self.use_target_network = 1
        self.target_network_update_freq = 1
        self.polyak = 0.995
        self.evaluation_criteria = 'return'
        self.device = 'cuda'
        self.info = 'cvae_rewards'
        self.dataset_method = 'none'
        self.ratio = 0.05
        self.dataset_level = 'medium'
        self.lambdaVal = 'none'

        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def train_inac_with_cvae_rewards(
    dataset_path: str,
    cvae_model_path: str,
    output_dir: str = "./inac_cvae_results",
    max_episodes: int = None,
    **training_args
):
    """
    Train INAC with CVAE progress rewards on furniture assembly task.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare dataset
    furniture_dataset = FurnitureAssemblyDataset(
        dataset_path=dataset_path,
        cvae_model_path=cvae_model_path,
        max_episodes=max_episodes
    )

    offline_data = furniture_dataset.to_d4rl_format()

    # Create config with correct dimensions
    config = INACFurnitureConfig(
        state_dim=furniture_dataset.states.shape[1],
        action_dim=furniture_dataset.actions.shape[1],
        **training_args
    )

    # Set up paths and environment
    config.exp_path = str(output_dir)
    torch_utils.ensure_dir(config.exp_path)

    # Create custom environment
    def env_fn():
        return FurnitureAssemblyEnvironment(config.state_dim, config.action_dim)

    config.env_fn = env_fn
    config.offline_data = offline_data

    # Set up random seed
    torch_utils.set_one_thread()
    torch_utils.random_seed(config.seed)

    # Set up logger
    config.tensorboard_logs = True
    config.logger = logger.Logger(config, config.exp_path)

    print("INAC Training Configuration:")
    print(f"  State dim: {config.state_dim}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Tau: {config.tau}")
    print(f"  Output dir: {config.exp_path}")

    # Initialize INAC agent
    agent = InSampleAC(
        device=config.device,
        discrete_control=config.discrete_control,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_units=config.hidden_units,
        learning_rate=config.learning_rate,
        tau=config.tau,
        polyak=config.polyak,
        exp_path=config.exp_path,
        seed=config.seed,
        env_fn=config.env_fn,
        timeout=config.timeout,
        gamma=config.gamma,
        offline_data=config.offline_data,
        batch_size=config.batch_size,
        use_target_network=config.use_target_network,
        target_network_update_freq=config.target_network_update_freq,
        evaluation_criteria=config.evaluation_criteria,
        logger=config.logger,
        lambdaVal=config.lambdaVal
    )

    # Train the agent
    print("Starting INAC training with CVAE progress rewards...")
    run_funcs.run_steps(agent, config.max_steps, config.log_interval, config.exp_path, config)

    print(f"Training completed! Results saved to {config.exp_path}")
    return agent, config


def main():
    parser = argparse.ArgumentParser(description='Train INAC with CVAE Progress Rewards')

    # Data paths
    parser.add_argument('--dataset_path', type=str,
                        default='./robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/low/success.zarr',
                        help='Path to furniture assembly dataset')
    parser.add_argument('--cvae_model_path', type=str,
                        default='./original_progress_cvae_results/best_original_progress_cvae_model.pt',
                        help='Path to trained CVAE model')
    parser.add_argument('--output_dir', type=str, default='./inac_cvae_results',
                        help='Output directory for results')

    # Training parameters
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum episodes to use from dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='Temperature parameter for in-sample softmax')
    parser.add_argument('--log_interval', type=int, default=5000,
                        help='Logging interval')
    parser.add_argument('--lambdaVal', type=str, default='none',
                        help='BC regularization weight')

    args = parser.parse_args()

    # Extract training arguments
    training_args = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'tau': args.tau,
        'log_interval': args.log_interval,
        'lambdaVal': args.lambdaVal
    }

    # Train the model
    agent, config = train_inac_with_cvae_rewards(
        dataset_path=args.dataset_path,
        cvae_model_path=args.cvae_model_path,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        **training_args
    )

    print("Training completed successfully!")


if __name__ == '__main__':
    main()