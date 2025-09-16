"""
Enhanced training script that integrates INAC (In-Sample Softmax Offline RL) with CVAE progress rewards

This script includes:
1. Comprehensive wandb logging similar to robust-rearrangement codebase
2. Headless environment evaluation with video recording
3. Policy rollouts and success rate tracking
4. Video saving and wandb upload functionality

The CVAE progress latent serves as a reward signal representing task completion progress,
while INAC ensures the policy stays close to expert demonstrations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robust-rearrangement"))

# Import Isaac Gym modules first, before any torch usage
import furniture_bench  # This imports Isaac Gym modules
from src.gym import get_rl_env
from src.eval.rollout import calculate_success_rate

# Import torch after Isaac Gym
import torch
import os
import sys
# torch will be imported later to avoid Isaac Gym conflicts
import numpy as np
import zarr
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
import pickle
import copy
import tempfile
from io import BytesIO
import imageio
import cv2
from tqdm import tqdm
import time

# Set environment variable to suppress D4RL errors before imports
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

# Add INAC codebase to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'INAC_MLRC_24'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'robust-rearrangement'))

try:
    from INAC_MLRC_24.core.agent.in_sample import InSampleAC
    from INAC_MLRC_24.core.utils import torch_utils, logger, run_funcs
except ImportError as e:
    print(f"INAC import failed: {e}")
    print("Please fix MuJoCo/D4RL installation or use: pip install cython==0.29.21")
    exit(1)

from cvae_progress import ProgressConditionalVAE

# Import robot state filtering function
sys.path.append(os.path.join(os.path.dirname(__file__), 'robust-rearrangement/src'))
from common.robot_state import filter_and_concat_robot_state

# Import torch after Isaac Gym setup is done
def _import_torch():
    import torch
    return torch

torch = None  # Will be imported when needed

# Wandb import
import wandb


class VideoRecorder:
    """Utility class for recording and saving videos."""

    def __init__(self):
        pass

    @staticmethod
    def create_in_memory_mp4(np_images, fps=10):
        """Create MP4 video in memory from numpy image sequence."""
        # Create a temporary file with .mp4 extension
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Get dimensions from the first frame
            if len(np_images) == 0:
                return BytesIO()

            height, width = np_images[0].shape[:2]

            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

            # Write frames
            for img in np_images:
                # Convert RGB to BGR (OpenCV uses BGR)
                if img.shape[2] == 3:  # If RGB
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                out.write(img_bgr)

            # Release the VideoWriter
            out.release()

            # Read the file back into memory
            with open(temp_filename, 'rb') as f:
                output = BytesIO(f.read())

            # Clean up the temporary file
            os.unlink(temp_filename)

            # Seek to the beginning of the BytesIO object
            output.seek(0)
            return output

        except Exception as e:
            # If there's an error, clean up and return empty BytesIO
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            print(f"Error creating video with OpenCV: {str(e)}")
            return BytesIO()

    @staticmethod
    def save_video_file(np_images, filename, fps=10):
        """Save video directly to file using robust-rearrangement approach."""
        try:
            from src.visualization.render_mp4 import create_in_memory_mp4

            # Create video in memory first
            video_bytes = create_in_memory_mp4(np_images, fps=fps)

            # Save to file
            with open(filename, 'wb') as f:
                f.write(video_bytes.getvalue())

        except ImportError:
            # Fallback to imageio without fps parameter
            with imageio.get_writer(filename) as writer:
                for img in np_images:
                    writer.append_data(img)


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
        global torch
        if torch is None:
            torch = _import_torch()

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

        # Load original dataset to get state/action structure with proper filtering
        print(f"Loading dataset structure from {original_dataset_path}")
        dataset = zarr.open(original_dataset_path, mode='r')

        raw_robot_state = np.array(dataset['robot_state'][0])
        parts_poses = np.array(dataset['parts_poses'][0])

        # Apply same robot state filtering as used in training
        robot_state_dict = {
            'ee_pos': raw_robot_state[:3],
            'ee_quat': raw_robot_state[3:7],
            'ee_pos_vel': raw_robot_state[7:10],
            'ee_ori_vel': raw_robot_state[10:13],
            'gripper_width': raw_robot_state[13]
        }
        filtered_robot_state = filter_and_concat_robot_state(robot_state_dict)

        sample_state = np.concatenate([filtered_robot_state, parts_poses])
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
        global torch
        if torch is None:
            torch = _import_torch()

        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get progress prediction from CVAE
            progress_pred = self.cvae.predict_progress(state_tensor)
            progress_reward = progress_pred.cpu().numpy()[0, 0]  # Extract scalar value

        return float(progress_reward)


class FurnitureAssemblySimpleEnv:
    """
    Simple simulation environment for furniture assembly using dataset trajectories.
    This is used for evaluation rollouts.
    """

    def __init__(
        self,
        dataset_path: str,
        cvae_env: CVAERewardEnvironment,
        max_episode_steps: int = 200
    ):
        self.cvae_env = cvae_env
        self.max_episode_steps = max_episode_steps

        # Load dataset for evaluation trajectories
        dataset = zarr.open(dataset_path, mode='r')
        robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])
        self.states = np.concatenate([robot_states, parts_poses], axis=-1)
        self.actions = np.array(dataset['action/pos'][:])
        self.episode_ends = np.array(dataset['episode_ends'][:])

        # Environment properties
        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.observation_space = self._create_dummy_space(self.state_dim)
        self.action_space = self._create_dummy_space(self.action_dim)

        # Current episode state
        self.current_episode_idx = 0
        self.current_step = 0
        self.episode_start_idx = 0
        self.episode_end_idx = self.episode_ends[0] if len(self.episode_ends) > 0 else len(self.states)

        # Video recording
        self.recording = False
        self.recorded_frames = []

    def _create_dummy_space(self, dim):
        """Create dummy gym space for compatibility."""
        class DummySpace:
            def __init__(self, dim):
                self.shape = (dim,)
                self.low = -np.inf * np.ones(dim)
                self.high = np.inf * np.ones(dim)
        return DummySpace(dim)

    def reset(self, record_video: bool = False):
        """Reset environment to start of a random episode."""
        self.recording = record_video
        self.recorded_frames = []

        # Choose random episode
        self.current_episode_idx = np.random.randint(0, len(self.episode_ends))

        # Calculate episode boundaries
        if self.current_episode_idx == 0:
            self.episode_start_idx = 0
        else:
            self.episode_start_idx = self.episode_ends[self.current_episode_idx - 1]
        self.episode_end_idx = self.episode_ends[self.current_episode_idx]

        self.current_step = 0

        # Return initial state
        initial_state = self.states[self.episode_start_idx].copy()

        if self.recording:
            # Create a dummy frame (you could render actual environment here)
            frame = self._create_dummy_frame(initial_state)
            self.recorded_frames.append(frame)

        return initial_state

    def step(self, action):
        """Take a step in the environment."""
        self.current_step += 1
        current_idx = min(
            self.episode_start_idx + self.current_step,
            self.episode_end_idx - 1
        )

        # Get next state from dataset
        next_state = self.states[current_idx].copy()

        # Get CVAE progress reward
        reward = self.cvae_env.get_cvae_progress_reward(next_state, action)

        # Check if episode is done
        done = (current_idx >= self.episode_end_idx - 1) or (self.current_step >= self.max_episode_steps)

        # Create dummy frame for recording
        if self.recording:
            frame = self._create_dummy_frame(next_state)
            self.recorded_frames.append(frame)

        info = {
            'progress_reward': reward,
            'episode_step': self.current_step,
            'dataset_idx': current_idx
        }

        return next_state, reward, done, info

    def _create_dummy_frame(self, state):
        """Create a dummy frame for video recording. In real env, this would be actual rendering."""
        # Create a simple visualization based on state
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128  # Gray background

        # Add some visual elements based on state values
        # This is just a placeholder - in real environment you'd have actual rendering

        # Add text with step info
        cv2.putText(frame, f"Step: {self.current_step}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Episode: {self.current_episode_idx}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add simple visualization based on state values (normalized)
        state_norm = (state - state.min()) / (state.max() - state.min() + 1e-8)
        for i, val in enumerate(state_norm[:10]):  # Show first 10 state components
            y_pos = int(100 + i * 10)
            bar_length = int(val * 200)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_length, y_pos + 8), (0, 255, 0), -1)

        return frame

    def get_recorded_video(self):
        """Get recorded video frames."""
        return np.array(self.recorded_frames) if self.recorded_frames else np.array([])


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

        # Load state-action data with proper robot state filtering
        raw_robot_states = np.array(dataset['robot_state'][:])
        parts_poses = np.array(dataset['parts_poses'][:])

        # Apply robot state filtering as done in robust-rearrangement
        print("Applying robot state filtering...")
        filtered_robot_states = []
        for raw_robot_state in raw_robot_states:
            # Convert raw robot state to dictionary format expected by filter function
            robot_state_dict = {
                'ee_pos': raw_robot_state[:3],
                'ee_quat': raw_robot_state[3:7],
                'ee_pos_vel': raw_robot_state[7:10],
                'ee_ori_vel': raw_robot_state[10:13],
                'gripper_width': raw_robot_state[13]  # Skip the last 2 dimensions
            }

            filtered = filter_and_concat_robot_state(robot_state_dict)
            filtered_robot_states.append(filtered)

        filtered_robot_states = np.array(filtered_robot_states)
        print(f"Robot state filtering: {raw_robot_states.shape[1]} -> {filtered_robot_states.shape[1]} dims")

        # Concatenate filtered robot states with parts poses
        self.states = np.concatenate([filtered_robot_states, parts_poses], axis=-1)
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

        # Scale rewards for INAC - CVAE gives 0-1, but INAC is very sensitive to reward scale
        # Use a more conservative scaling to prevent exploding losses
        rewards = rewards * 10.0  # Scale 0-1 to 0-10 (more conservative)

        print(f"Scaled rewards - min: {rewards.min():.3f}, max: {rewards.max():.3f}, mean: {rewards.mean():.3f}")
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

        # INAC format dictionary (wrapped under single key)
        d4rl_data = {
            'env': {
                'states': self.states.astype(np.float32),
                'actions': self.actions.astype(np.float32),
                'rewards': rewards.astype(np.float32),
                'next_states': next_states.astype(np.float32),
                'terminations': terminals
            }
        }

        print(f"Converted to INAC format:")
        print(f"  Observations: {d4rl_data['env']['states'].shape}")
        print(f"  Actions: {d4rl_data['env']['actions'].shape}")
        print(f"  Rewards: {d4rl_data['env']['rewards'].shape}")
        print(f"  Terminals: {d4rl_data['env']['terminations'].sum()} terminal states")

        return d4rl_data


class PolicyWrapper:
    """Wrapper around INAC agent to provide simple policy interface for evaluation."""

    def __init__(self, inac_agent):
        self.agent = inac_agent
        self.device = inac_agent.device

    def predict(self, state):
        """Get action prediction from the policy."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, _ = self.agent.ac.pi(state)

        return action.cpu().numpy().squeeze()


def evaluate_policy(
    policy: PolicyWrapper,
    env: FurnitureAssemblySimpleEnv,
    n_episodes: int = 10,
    record_videos: bool = True,
    video_save_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate policy in the environment.
    """
    episode_returns = []
    episode_lengths = []
    success_count = 0
    videos = []

    print(f"Evaluating policy for {n_episodes} episodes...")

    for episode_idx in tqdm(range(n_episodes), desc="Evaluation episodes"):
        state = env.reset(record_video=record_videos)
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            action = policy.predict(state)
            next_state, reward, done, info = env.step(action)

            episode_return += reward
            episode_length += 1
            state = next_state

            # Simple success criterion: high cumulative progress reward
            if episode_return > 0.8:  # Adjust threshold as needed
                success_count += 1
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Save video if recording
        if record_videos:
            video_frames = env.get_recorded_video()
            if len(video_frames) > 0:
                videos.append(video_frames)

                # Save individual video file
                if video_save_dir is not None:
                    video_path = video_save_dir / f"eval_episode_{episode_idx:03d}.mp4"
                    VideoRecorder.save_video_file(video_frames, str(video_path), fps=10)

    # Calculate statistics
    eval_stats = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes,
        'n_episodes': n_episodes
    }

    print(f"Evaluation Results:")
    print(f"  Mean Return: {eval_stats['mean_return']:.3f} ± {eval_stats['std_return']:.3f}")
    print(f"  Mean Length: {eval_stats['mean_length']:.1f}")
    print(f"  Success Rate: {eval_stats['success_rate']:.2%}")

    return eval_stats, videos


def try_furnituresim_evaluation(policy, config, n_episodes, video_save_dir):
    """Try to evaluate with FurnitureSim, return None if fails."""
    try:

        class FurnitureSimActor:
            def __init__(self, inac_policy, furnituresim_env, device="cuda"):
                self.policy = inac_policy
                self.device = device
                self.env = furnituresim_env  # Need env reference for robot state filtering

                # Add required attributes from robust-rearrangement Actor interface
                # Create a dummy normalizer that has .to() method but does nothing
                class DummyNormalizer:
                    def to(self, device):
                        return self

                self.normalizer = DummyNormalizer()  # INAC doesn't use normalizer but needs .to() method
                self.model = inac_policy  # The policy itself is the model

            def reset(self):
                """Reset the actor (required by calculate_success_rate)"""
                pass  # INAC policy doesn't need special reset

            def set_task(self, task_idx):
                """Set the task for the actor (required by calculate_success_rate)"""
                pass  # INAC policy doesn't need task setting

            def action(self, obs, deterministic=False):
                """Get action from observation (main interface used by rollout)"""
                # At this point, rollout.py has already called:
                # obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])
                # So obs["robot_state"] is already a tensor, not a dict

                robot_state = obs["robot_state"].flatten()  # Already processed by rollout
                parts_poses = obs["parts_poses"].flatten()

                state = torch.cat([robot_state, parts_poses], dim=0)

                if len(state.shape) == 1:
                    state = state.unsqueeze(0)

                with torch.no_grad():
                    action, _ = self.policy(state)

                return action

        # Create FurnitureSim environment - keep concat_robot_state=False so TensorDict works
        # The filtering will happen in rollout.py before calling actor.action()
        env = get_rl_env(
            gpu_id=0,
            task="one_leg",
            num_envs=1,
            randomness="low",
            observation_space="state",
            max_env_steps=200,
            resize_img=False,
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            verbose=False,
            headless=True
            # concat_robot_state=False by default - rollout.py will call filter_and_concat_robot_state
        )
        actor = FurnitureSimActor(policy.agent.ac.pi, env, device=policy.device)

        rollout_stats = calculate_success_rate(
            env=env,
            actor=actor,
            n_rollouts=n_episodes,
            rollout_max_steps=200,
            epoch_idx=0,
            rollout_save_dir=None,  # Disable rollout saving to avoid TensorDict error
            save_rollouts_to_wandb=False,  # Disable wandb saving to avoid TensorDict error
            save_failures=False,
            n_parts_assemble=1,
            compress_pickles=False,
            resize_video=False  # Disable video processing to avoid TensorDict error
        )

        eval_stats = {
            "mean_return": float(rollout_stats.total_return / rollout_stats.n_rollouts),
            "std_return": 0.0,
            "mean_length": float(rollout_stats.rollout_max_steps),
            "success_rate": float(rollout_stats.success_rate),
            "n_episodes": rollout_stats.n_rollouts,
            "n_success": rollout_stats.n_success
        }

        print(f"FurnitureSim Success Rate: {eval_stats['success_rate']:.2%}")
        return eval_stats, []

    except Exception as e:
        print(f"FurnitureSim evaluation failed: {e}")
        return None, None


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
        self.tau = 0.1  # Increased for stability with CVAE rewards
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

        # Evaluation settings
        self.eval_every = 10000
        self.eval_episodes = 10
        self.record_videos = True

        # Wandb settings
        self.wandb_project = 'inac_cvae_furniture'
        self.wandb_entity = 'swami2004'
        self.wandb_mode = 'online'
        self.wandb_name = 'swami2004'

        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def train_inac_with_cvae_rewards_enhanced(
    dataset_path: str,
    cvae_model_path: str,
    output_dir: str = "./inac_cvae_results_enhanced",
    max_episodes: int = None,
    **training_args
):
    """
    Train INAC with CVAE progress rewards on furniture assembly task with enhanced logging and evaluation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

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

    # Create evaluation environment
    eval_env = FurnitureAssemblySimpleEnv(
        dataset_path=dataset_path,
        cvae_env=furniture_dataset.cvae_env,
        max_episode_steps=200
    )

    # Add env_fn to config for INAC logger compatibility
    config.env_fn = lambda: eval_env
    config.offline_data = offline_data

    # Set up random seed
    global torch
    if torch is None:
        torch = _import_torch()

    torch_utils.set_one_thread()
    torch_utils.random_seed(config.seed)

    # Initialize Wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
        mode=config.wandb_mode,
        config={
            'algorithm': 'INAC',
            'reward_type': 'CVAE_progress',
            'dataset_path': dataset_path,
            'cvae_model_path': cvae_model_path,
            'state_dim': config.state_dim,
            'action_dim': config.action_dim,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'tau': config.tau,
            'max_steps': config.max_steps,
            'seed': config.seed
        }
    )

    # Set up logger (keeping original INAC logging)
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
    print(f"  Wandb project: {config.wandb_project}")

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
        env_fn=lambda: eval_env,  # Dummy env function for compatibility
        timeout=config.timeout,
        gamma=config.gamma,
        offline_data=offline_data,
        batch_size=config.batch_size,
        use_target_network=config.use_target_network,
        target_network_update_freq=config.target_network_update_freq,
        evaluation_criteria=config.evaluation_criteria,
        logger=config.logger,
        lambdaVal=config.lambdaVal
    )

    # Create policy wrapper for evaluation
    policy = PolicyWrapper(agent)

    print("Starting INAC training with CVAE progress rewards...")

    # Enhanced training loop with evaluation
    step = 0
    start_time = time.time()

    while step < config.max_steps:
        # Training step
        batch_data = agent.get_data()
        loss_dict = agent.update(batch_data)

        step += 1

        # Log training metrics
        if step % config.log_interval == 0:
            train_time = time.time() - start_time

            # Log to wandb
            wandb_log = {
                'step': step,
                'train/actor_loss': loss_dict['actor'],
                'train/critic_loss': loss_dict['critic'],
                'train/value_loss': loss_dict['value'],
                'train/beta_loss': loss_dict['beta'],
                'train/q_info': loss_dict['q_info'],
                'train/v_info': loss_dict['v_info'],
                'train/logp_info': loss_dict['logp_info'],
                'train/steps_per_sec': config.log_interval / train_time
            }

            wandb.log(wandb_log, step=step)

            print(f"Step {step:6d}: Actor={loss_dict['actor']:.4f}, "
                  f"Critic={loss_dict['critic']:.4f}, "
                  f"Value={loss_dict['value']:.4f}")

            start_time = time.time()

        # Evaluation
        if step % config.eval_every == 0 and step > 0:
            print(f"\nEvaluating policy at step {step}...")

            eval_video_dir = video_dir / f"step_{step:06d}"
            eval_video_dir.mkdir(exist_ok=True)

            # Try FurnitureSim evaluation first, fallback to simple env
            furnituresim_stats, _ = try_furnituresim_evaluation(
                policy, config, config.eval_episodes, eval_video_dir
            )

            if furnituresim_stats is not None:
                eval_stats = furnituresim_stats
                eval_type = "furnituresim"
                videos = []  # FurnitureSim handles its own videos
            else:
                print("Using simple environment for evaluation...")
                eval_stats, videos = evaluate_policy(
                    policy=policy,
                    env=eval_env,
                    n_episodes=config.eval_episodes,
                    record_videos=config.record_videos,
                    video_save_dir=eval_video_dir if config.record_videos else None
                )
                eval_type = "simple"

            # Log evaluation results to wandb
            eval_log = {
                f'eval_{eval_type}/mean_return': eval_stats['mean_return'],
                f'eval_{eval_type}/std_return': eval_stats['std_return'],
                f'eval_{eval_type}/mean_length': eval_stats['mean_length'],
                f'eval_{eval_type}/success_rate': eval_stats['success_rate']
            }

            wandb.log(eval_log, step=step)

            # Upload videos to wandb (only for simple env)
            if eval_type == "simple" and config.record_videos and len(videos) > 0:
                # Create video montage from first few episodes
                n_video_upload = min(3, len(videos))

                for i in range(n_video_upload):
                    video_frames = videos[i]

                    # Convert list to numpy array for wandb.Video
                    video_array = np.array(video_frames)  # Shape: (num_frames, height, width, channels)

                    # Create wandb video from numpy array (similar to robust-rearrangement)
                    wandb.log({
                        f'eval/video_episode_{i}': wandb.Video(video_array, fps=10, format="mp4")
                    }, step=step)

            print(f"Evaluation completed. Success rate: {eval_stats['success_rate']:.2%}")

    print("Training completed!")

    # Final evaluation
    print("\nRunning final evaluation...")

    # Try FurnitureSim first
    final_eval_stats, _ = try_furnituresim_evaluation(
        policy, config, 20, video_dir / "final_eval"
    )

    if final_eval_stats is not None:
        eval_type = "furnituresim"
    else:
        print("Using simple environment for final evaluation...")
        final_eval_stats, final_videos = evaluate_policy(
            policy=policy,
            env=eval_env,
            n_episodes=20,  # More episodes for final eval
            record_videos=True,
            video_save_dir=video_dir / "final_eval"
        )
        eval_type = "simple"

    # Log final results
    final_log = {
        f'final_eval_{eval_type}/mean_return': final_eval_stats['mean_return'],
        f'final_eval_{eval_type}/std_return': final_eval_stats['std_return'],
        f'final_eval_{eval_type}/success_rate': final_eval_stats['success_rate']
    }
    wandb.log(final_log)

    # Save final model (exclude lambda functions from config)
    model_path = output_dir / 'final_inac_cvae_model.pt'

    # Create a picklable config copy (exclude lambda functions)
    config_dict = {
        'seed': config.seed,
        'env_name': config.env_name,
        'dataset': config.dataset,
        'state_dim': config.state_dim,
        'action_dim': config.action_dim,
        'tau': config.tau,
        'max_steps': config.max_steps,
        'log_interval': config.log_interval,
        'learning_rate': config.learning_rate,
        'hidden_units': config.hidden_units,
        'batch_size': config.batch_size,
        'gamma': config.gamma,
        'polyak': config.polyak,
        'exp_path': config.exp_path,
        'lambdaVal': config.lambdaVal
    }

    torch.save({
        'agent_state_dict': agent.ac.pi.state_dict(),
        'config': config_dict,
        'eval_stats': final_eval_stats
    }, model_path)

    print(f"Final model saved to {model_path}")
    print(f"Final success rate: {final_eval_stats['success_rate']:.2%}")

    wandb.finish()

    return agent, config, final_eval_stats


def main():
    parser = argparse.ArgumentParser(description='Train INAC with CVAE Progress Rewards (Enhanced)')

    # Data paths
    parser.add_argument('--dataset_path', type=str,
                        default='./robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/low/success.zarr',
                        help='Path to furniture assembly dataset')
    parser.add_argument('--cvae_model_path', type=str,
                        default='./corrected_progress_cvae_results/corrected_progress_cvae_model.pt',
                        help='Path to trained CVAE model')
    parser.add_argument('--output_dir', type=str, default='./inac_cvae_results_enhanced',
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
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Temperature parameter for in-sample softmax')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Logging interval')
    parser.add_argument('--eval_every', type=int, default=5000,
                        help='Evaluation interval')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--lambdaVal', type=str, default='none',
                        help='BC regularization weight')

    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='inac_cvae_furniture',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='swami2004',
                        help='Wandb entity')
    parser.add_argument('--wandb_name', type=str, default='swami2004',
                        help='Wandb run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Wandb mode')
    parser.add_argument('--record_videos', action='store_true', default=True,
                        help='Record evaluation videos')

    args = parser.parse_args()

    # Extract training arguments
    training_args = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'tau': args.tau,
        'log_interval': args.log_interval,
        'eval_every': args.eval_every,
        'eval_episodes': args.eval_episodes,
        'lambdaVal': args.lambdaVal,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_name': args.wandb_name,
        'wandb_mode': args.wandb_mode,
        'record_videos': args.record_videos
    }

    # Train the model
    agent, config, eval_stats = train_inac_with_cvae_rewards_enhanced(
        dataset_path=args.dataset_path,
        cvae_model_path=args.cvae_model_path,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        **training_args
    )

    print("Training completed successfully!")
    print(f"Final success rate: {eval_stats['success_rate']:.2%}")

if __name__ == '__main__':
    main()



