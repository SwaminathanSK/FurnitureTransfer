"""
Residual INAC Training with CVAE Progress Rewards

This script implements an offline INAC-based residual RL method that uses CVAE encoder outputs
as rewards over a frozen BC policy. Instead of the online sparse reward PPO residual training,
this approach uses the in-sample softmax-based INAC algorithm for more stable offline training.

Key features:
1. Frozen base diffusion policy from behavior cloning
2. INAC-based residual policy training (offline, in-sample softmax)
3. CVAE progress rewards instead of sparse environment rewards
4. Integration with robust-rearrangement codebase structure
"""

import os
import sys
from pathlib import Path
import time
import random
from typing import Dict, Tuple, Optional

# Initialize environment for evaluation
from src.gym import get_rl_env
from src.eval.rollout import calculate_success_rate
from src.common.tasks import task2idx

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import wandb
import zarr

# Add INAC codebase to path
sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

# Robust-rearrangement imports
from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.models.residual import ResidualPolicy
from src.eval.eval_utils import get_model_from_api_or_cached
from src.common.config_util import merge_base_bc_config_with_root_config
from src.dataset.dataset import StateDataset
from src.common.pytorch_util import dict_to_device
from src.common.robot_state import filter_and_concat_robot_state
from src.gym.observation import DEFAULT_STATE_OBS

# INAC imports
try:
    from core.agent.in_sample import InSampleAC
    from core.utils import torch_utils, logger
except ImportError as e:
    print(f"INAC import failed: {e}")
    print("Please ensure INAC_MLRC_24 is properly installed")
    sys.exit(1)

# CVAE imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cvae_progress import ProgressConditionalVAE

# Register eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


class CVAERewardComputer:
    """
    Computes CVAE-based progress rewards for residual RL training.
    """

    def __init__(
        self,
        cvae_model_path: str,
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
        print(f"CVAE model loaded successfully with state_dim={model_config['state_dim']}, action_dim={model_config['action_dim']}")

    def normalize_state_action(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize state and action using CVAE training statistics."""
        if self.data_stats is not None:
            state = (state - self.data_stats['state_mean']) / (self.data_stats['state_std'] + 1e-8)
            action = (action - self.data_stats['action_mean']) / (self.data_stats['action_std'] + 1e-8)
        return state, action

    def compute_progress_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute progress reward from CVAE model.
        Uses the CVAE's predicted progress latent as reward signal.
        """
        # Normalize inputs
        norm_state, norm_action = self.normalize_state_action(state, action)

        # Convert to tensors
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get progress prediction from CVAE (state-only predictor for reward)
            progress_pred = self.cvae.predict_progress(state_tensor)
            progress_reward = progress_pred.cpu().numpy()[0, 0]  # Extract scalar value

        return float(progress_reward)


class FurnitureDatasetForINAC:
    """
    Dataset loader that prepares furniture assembly data for INAC training.
    Loads data and computes CVAE-based rewards.
    """

    def __init__(
        self,
        dataset_path: str,
        cvae_model_path: str,
        action_type: str = 'pos',
        max_episodes: int = None,
        reward_scale: float = 10.0
    ):
        self.dataset_path = dataset_path
        self.action_type = action_type
        self.reward_scale = reward_scale

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
            robot_state_dict = {
                'ee_pos': raw_robot_state[:3],
                'ee_quat': raw_robot_state[3:7],
                'ee_pos_vel': raw_robot_state[7:10],
                'ee_ori_vel': raw_robot_state[10:13],
                'gripper_width': raw_robot_state[13]
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

        # Initialize CVAE reward computer
        self.cvae_reward_computer = CVAERewardComputer(
            cvae_model_path=cvae_model_path
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

    def get_residual_dataset_for_inac(self, base_agent) -> dict:
        """
        Convert dataset to residual format following residual PPO approach:
        1. For each state: get base_action from BC diffusion policy
        2. Create residual_obs = process_obs(state) + base_action
        3. Create residual_actions = target_action - base_action
        """
        import torch
        from tqdm import tqdm
        from src.common.geometry import np_proprioceptive_quat_to_6d_rotation

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_agent = base_agent.to(device)

        print("Computing base actions and residual observations...")

        # Split states back into robot_state and parts_poses
        robot_state_dim = 14  # After filtering: [pos(3), quat(4), pos_vel(3), ori_vel(3), gripper(1)]
        robot_states = self.states[:, :robot_state_dim]
        parts_poses = self.states[:, robot_state_dim:]

        # Process one sample at a time to avoid batch issues with diffusion policy
        residual_observations = []
        base_actions = []

        for i in tqdm(range(len(self.states)), desc="Processing dataset"):
            robot_state = robot_states[i:i+1]  # Keep batch dimension
            parts_pose = parts_poses[i:i+1]

            # Convert to torch tensors
            robot_state_torch = torch.from_numpy(robot_state).float().to(device)
            parts_pose_torch = torch.from_numpy(parts_pose).float().to(device)

            # Create observation dict for base agent
            obs = {
                'robot_state': robot_state_torch,
                'parts_poses': parts_pose_torch
            }

            # Get base actions from BC policy
            with torch.no_grad():
                base_action = base_agent.base_action_normalized(obs)
                # Process observations (converts 14D robot state to 16D)
                processed_obs = base_agent.process_obs(obs)

                # Debug: print shapes on first sample
                if i == 0:
                    print(f"Debug shapes (single sample):")
                    print(f"  processed_obs: {processed_obs.shape}")
                    print(f"  base_action: {base_action.shape}")

                # Create residual observations: processed_obs + base_action
                residual_obs = torch.cat([processed_obs, base_action], dim=-1)

            # Store results
            residual_observations.append(residual_obs.cpu().numpy())
            base_actions.append(base_action.cpu().numpy())

        # Concatenate all batches
        residual_observations = np.concatenate(residual_observations, axis=0)
        base_actions = np.concatenate(base_actions, axis=0)

        # Compute residual actions: target_action - base_action
        residual_actions = self.actions - base_actions

        print(f"Residual dataset created:")
        print(f"  Residual observations: {residual_observations.shape}")
        print(f"  Residual actions: {residual_actions.shape}")
        print(f"  Base actions: {base_actions.shape}")

        return {
            'residual_observations': residual_observations,
            'residual_actions': residual_actions,
            'base_actions': base_actions
        }

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
                rewards[j] = self.cvae_reward_computer.compute_progress_reward(
                    self.states[j], self.actions[j]
                )

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(self.states)} transitions")

        print(f"CVAE rewards computed - min: {rewards.min():.3f}, max: {rewards.max():.3f}, mean: {rewards.mean():.3f}")

        # Scale rewards for INAC
        rewards = rewards * self.reward_scale

        print(f"Scaled rewards - min: {rewards.min():.3f}, max: {rewards.max():.3f}, mean: {rewards.mean():.3f}")
        return rewards

    def to_inac_format(self, base_agent) -> Dict:
        """
        Convert to format expected by INAC training.
        For residual INAC, we need to create (state + base_action) pairs as observations.
        """
        # Compute CVAE rewards
        rewards = self.compute_cvae_rewards()

        print("Creating residual observations (state + base_action)...")

        # Convert states and actions to torch tensors for base policy processing
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For residual policy, we need robot_state and parts_poses separately
        # Based on the error, we know filtered robot state is 14 dims, so:
        robot_state_dim = 14  # After filtering
        robot_states = self.states[:, :robot_state_dim]
        parts_poses = self.states[:, robot_state_dim:]

        print(f"Processing {len(self.states)} states individually...")

        residual_observations_list = []

        # Process each observation individually to avoid deque state issues
        # with the diffusion policy's sequential inference design
        for i in range(len(self.states)):
            # Reset base agent state for each observation
            base_agent.reset()

            # Convert single observation to tensor
            obs = {
                'robot_state': torch.tensor(robot_states[i:i+1], dtype=torch.float32).to(device),
                'parts_poses': torch.tensor(parts_poses[i:i+1], dtype=torch.float32).to(device)
            }

            # Get base action and processed observation
            with torch.no_grad():
                try:
                    base_naction = base_agent.base_action_normalized(obs)
                    processed_obs = base_agent.process_obs(obs)

                    # Combine processed obs + base action for residual policy
                    residual_obs = torch.cat([processed_obs, base_naction], dim=-1)
                    residual_observations_list.append(residual_obs.cpu().numpy())

                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        print(f"cuDNN error at observation {i}, skipping...")
                        # Skip this observation if cuDNN fails
                        continue
                    else:
                        raise e

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(self.states)} states")

        # Concatenate all batches
        residual_observations = np.concatenate(residual_observations_list, axis=0)
        print(f"Created residual observations with shape: {residual_observations.shape}")

        # Create next states (shifted by 1)
        next_states = np.zeros_like(residual_observations)
        terminals = np.zeros(len(residual_observations), dtype=bool)

        # Handle episode boundaries
        episode_start = 0
        for episode_end in self.episode_ends:
            # Next states within episode
            if episode_end - episode_start > 1:
                next_states[episode_start:episode_end-1] = residual_observations[episode_start+1:episode_end]

            # Terminal state (last state of episode points to itself)
            next_states[episode_end-1] = residual_observations[episode_end-1]
            terminals[episode_end-1] = True

            episode_start = episode_end

        # INAC format dictionary
        inac_data = {
            'env': {
                'states': residual_observations.astype(np.float32),
                'actions': self.actions.astype(np.float32),
                'rewards': rewards.astype(np.float32),
                'next_states': next_states.astype(np.float32),
                'terminations': terminals
            }
        }

        print(f"Converted to INAC format:")
        print(f"  Residual States: {inac_data['env']['states'].shape}")
        print(f"  Actions: {inac_data['env']['actions'].shape}")
        print(f"  Rewards: {inac_data['env']['rewards'].shape}")
        print(f"  Terminals: {inac_data['env']['terminations'].sum()} terminal states")

        return inac_data


class ResidualINACActor:
    """
    INAC-based residual actor that combines a frozen base policy with a residual policy.
    """

    def __init__(
        self,
        base_agent: ResidualDiffusionPolicy,
        inac_agent: InSampleAC,
        device: str = 'cuda'
    ):
        self.base_agent = base_agent
        self.inac_agent = inac_agent
        self.device = device

        # Freeze base policy
        for param in self.base_agent.model.parameters():
            param.requires_grad = False
        for param in self.base_agent.normalizer.parameters():
            param.requires_grad = False

        print("Base policy frozen for residual training")

    def get_residual_action(self, residual_obs: torch.Tensor) -> torch.Tensor:
        """Get residual action from INAC policy."""
        with torch.no_grad():
            residual_action, _ = self.inac_agent.ac.pi(residual_obs)
        return residual_action

    def get_full_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get full action (base + residual)."""
        # Get base action
        base_naction = self.base_agent.base_action_normalized(obs)

        # Process observation for residual policy
        processed_obs = self.base_agent.process_obs(obs)
        residual_obs = torch.cat([processed_obs, base_naction], dim=-1)

        # Get residual action
        residual_action = self.get_residual_action(residual_obs)

        # Combine base and residual
        full_naction = base_naction + residual_action * self.base_agent.residual_policy.action_scale

        # Denormalize
        return self.base_agent.normalizer(full_naction, "action", forward=False)


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    Main training function for residual INAC with CVAE rewards.
    """

    OmegaConf.set_struct(cfg, False)

    # Add CVAE-specific config
    if not hasattr(cfg, 'cvae'):
        cfg.cvae = DictConfig({
            'model_path': './corrected_progress_cvae_results/corrected_progress_cvae_model.pt',
            'reward_scale': 10.0
        })

    if not hasattr(cfg, 'inac'):
        cfg.inac = DictConfig({
            'hidden_units': 256,
            'learning_rate': 3e-4,
            'tau': 0.1,
            'batch_size': 256,
            'max_steps': 100000,
            'log_interval': 1000,
            'eval_interval': 5000,
            'gamma': 0.99,
            'polyak': 0.995
        })

    # Ensure exactly one of cfg.base_policy.wandb_id or cfg.base_policy.wt_path is set
    assert (
        sum([
            cfg.base_policy.wandb_id is not None,
            cfg.base_policy.wt_path is not None,
        ]) == 1
    ), "Exactly one of base_policy.wandb_id or base_policy.wt_path must be set"

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    run_name = f"{int(time.time())}_residual_inac_cvae_{cfg.seed}"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device(f"cuda:{cfg.gpu_id}")

    # Load base policy
    if cfg.base_policy.wandb_id is not None:
        base_cfg, base_wts = get_model_from_api_or_cached(
            cfg.base_policy.wandb_id,
            wt_type=cfg.base_policy.wt_type,
            wandb_mode=cfg.wandb.mode,
        )
    elif cfg.base_policy.wt_path is not None:
        base_wts = cfg.base_policy.wt_path
        base_cfg: DictConfig = OmegaConf.create(torch.load(base_wts)["config"])
    else:
        raise ValueError("No base policy provided")

    merge_base_bc_config_with_root_config(cfg, base_cfg)
    cfg.actor_name = f"residual_inac_{cfg.base_policy.actor.name}"

    # Initialize base agent
    if cfg.base_policy.actor.name == "diffusion":
        base_agent = ResidualDiffusionPolicy(device, base_cfg)
    else:
        raise ValueError(f"Unsupported actor type: {cfg.base_policy.actor.name}")

    base_agent.load_base_state_dict(base_wts)
    base_agent.to(device)
    base_agent.eval()

    print(f"Base policy loaded: {cfg.base_policy.actor.name}")

    # Load dataset and compute CVAE rewards
    dataset_path = './robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/low/success.zarr'
    if hasattr(cfg, 'data_path'):
        dataset_path = cfg.data_path

    furniture_dataset = FurnitureDatasetForINAC(
        dataset_path=dataset_path,
        cvae_model_path=cfg.cvae.model_path,
        reward_scale=cfg.cvae.reward_scale,
        max_episodes=getattr(cfg, 'max_episodes', None)
    )

    # Convert dataset to residual format following residual PPO approach
    print("Converting dataset to residual format...")

    # Create cache filename based on dataset path and base policy
    import hashlib
    dataset_hash = hashlib.md5(str(cfg.data_path).encode()).hexdigest()[:8]
    base_policy_path = cfg.base_policy.get('wt_path', cfg.base_policy.get('wandb_id', 'unknown'))
    base_policy_hash = hashlib.md5(str(base_policy_path).encode()).hexdigest()[:8]
    cache_filename = f"residual_data_{dataset_hash}_{base_policy_hash}.npz"
    cache_path = Path("./cache") / cache_filename
    cache_path.parent.mkdir(exist_ok=True)

    # Try to load cached data first
    if cache_path.exists():
        print(f"Loading cached residual data from {cache_path}")
        cached_data = np.load(cache_path)
        residual_data = {
            'residual_observations': cached_data['residual_observations'],
            'residual_actions': cached_data['residual_actions'],
            'base_actions': cached_data['base_actions']
        }
        print(f"Cached residual dataset loaded:")
        print(f"  Residual observations: {residual_data['residual_observations'].shape}")
        print(f"  Residual actions: {residual_data['residual_actions'].shape}")
        print(f"  Base actions: {residual_data['base_actions'].shape}")
    else:
        print("No cached data found, processing dataset...")
        residual_data = furniture_dataset.get_residual_dataset_for_inac(base_agent)

        # Save processed data to cache
        print(f"Saving processed data to cache: {cache_path}")
        np.savez_compressed(
            cache_path,
            residual_observations=residual_data['residual_observations'],
            residual_actions=residual_data['residual_actions'],
            base_actions=residual_data['base_actions']
        )

    # Use CVAE rewards but residual observations and actions
    offline_data = {
        'env': {
            'states': residual_data['residual_observations'],  # processed_obs + base_action
            'actions': residual_data['residual_actions'],      # target_action - base_action
            'rewards': furniture_dataset.compute_cvae_rewards(),
            'next_states': np.roll(residual_data['residual_observations'], -1, axis=0),
            'terminations': np.zeros(len(residual_data['residual_observations']), dtype=bool)
        }
    }

    # Calculate dimensions
    residual_obs_dim = residual_data['residual_observations'].shape[1]  # processed_state + base_action
    action_dim = residual_data['residual_actions'].shape[1]
    original_state_dim = furniture_dataset.states.shape[1]  # For logging purposes

    print(f"Residual policy dimensions: residual_obs_dim={residual_obs_dim}, action_dim={action_dim}")
    print(f"Original state dim: {original_state_dim}")

    # Setup paths
    exp_path = Path("./residual_inac_cvae_results") / run_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Set up torch utils
    torch_utils.set_one_thread()
    torch_utils.random_seed(cfg.seed)

    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.mode,
        config={
            'algorithm': 'Residual_INAC_CVAE',
            'base_policy': cfg.base_policy.actor.name,
            'cvae_model_path': cfg.cvae.model_path,
            'original_state_dim': original_state_dim,
            'action_dim': action_dim,
            'residual_obs_dim': residual_obs_dim,
            'batch_size': cfg.inac.batch_size,
            'learning_rate': cfg.inac.learning_rate,
            'tau': cfg.inac.tau,
            'max_steps': cfg.inac.max_steps,
            'seed': cfg.seed,
            'reward_scale': cfg.cvae.reward_scale
        }
    )

    # Initialize INAC logger
    inac_logger = logger.Logger(cfg.inac, str(exp_path))

    # Initialize INAC agent for residual policy
    inac_agent = InSampleAC(
        device=str(device),
        discrete_control=0,
        state_dim=residual_obs_dim,
        action_dim=action_dim,
        hidden_units=cfg.inac.hidden_units,
        learning_rate=cfg.inac.learning_rate,
        tau=cfg.inac.tau,
        polyak=cfg.inac.polyak,
        exp_path=str(exp_path),
        seed=cfg.seed,
        env_fn=lambda: None,  # Dummy env function
        timeout=1000,
        gamma=cfg.inac.gamma,
        offline_data=offline_data,
        batch_size=cfg.inac.batch_size,
        use_target_network=1,
        target_network_update_freq=1,
        evaluation_criteria='return',
        logger=inac_logger,
        lambdaVal='none'
    )

    print("INAC agent initialized for residual policy training")

    # Create evaluation environment once and reuse it
    eval_env = None
    if cfg.inac.eval_interval > 0:
        print("Creating evaluation environment...")
        # Check if we want to save videos and configure environment accordingly
        save_videos = getattr(cfg.eval, 'save_videos', False)
        show_on_screen = getattr(cfg.eval, 'show_on_screen', False)

        if save_videos or show_on_screen:
            # Include camera observations for video recording (need both cameras)
            obs_keys = DEFAULT_STATE_OBS + ["color_image1", "color_image2", "parts_poses"]
            observation_space = "state_image"
            print("Creating evaluation environment with camera observations")
            if show_on_screen:
                print("  One rollout will be shown on screen (headless=False)")
        else:
            obs_keys = DEFAULT_STATE_OBS + ["parts_poses"]  # Need parts_poses for state observations too
            observation_space = "state"

        eval_env = get_rl_env(
            gpu_id=cfg.gpu_id,
            act_rot_repr=getattr(cfg.control, 'act_rot_repr', 'rot_6d'),
            action_type=getattr(cfg.control, 'control_mode', 'pos'),
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode=getattr(cfg.control, 'controller', 'diffik'),
            obs_keys=obs_keys,
            task=cfg.env.task,
            compute_device_id=cfg.gpu_id,
            graphics_device_id=cfg.gpu_id,
            headless=not show_on_screen,  # Show on screen for first rollout if requested
            num_envs=1,  # Single env for eval
            observation_space=observation_space,
            randomness=cfg.env.randomness,
            max_env_steps=100_000_000,
        )
        print("Evaluation environment created successfully")

    # Training loop
    print("Starting residual INAC training with CVAE rewards...")

    step = 0
    start_time = time.time()

    while step < cfg.inac.max_steps:
        # Training step
        batch_data = inac_agent.get_data()
        loss_dict = inac_agent.update(batch_data)

        step += 1

        # Log training metrics
        if step % cfg.inac.log_interval == 0:
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
                'train/steps_per_sec': cfg.inac.log_interval / train_time
            }

            wandb.log(wandb_log, step=step)

            print(f"Step {step:6d}: Actor={loss_dict['actor']:.4f}, "
                  f"Critic={loss_dict['critic']:.4f}, "
                  f"Value={loss_dict['value']:.4f}")

            start_time = time.time()

        # Evaluation - do direct rollouts like residual PPO
        if step % cfg.inac.eval_interval == 0 and step > 0 and eval_env is not None:
            print(f"\nEvaluation at step {step}")

            # Evaluation parameters
            n_rollouts = getattr(cfg.eval, 'n_rollouts', 10)
            rollout_max_steps = getattr(cfg.rollout, 'max_steps', 200)

            total_success = 0
            total_returns = []

            print(f"Running {n_rollouts} evaluation rollouts...")

            # Video saving setup
            save_videos = getattr(cfg.eval, 'save_videos', False)
            show_on_screen = getattr(cfg.eval, 'show_on_screen', False)
            video_save_dir = "/home/swaminathan/furniture-videos"
            if save_videos:
                import os
                from datetime import datetime
                from src.visualization.render_mp4 import create_in_memory_mp4
                os.makedirs(video_save_dir, exist_ok=True)

            for rollout_idx in range(n_rollouts):
                # Only show the first rollout on screen if requested
                if show_on_screen and rollout_idx == 0:
                    print(f"  Rollout {rollout_idx}: Showing on screen")
                elif show_on_screen and rollout_idx == 1:
                    print("  Subsequent rollouts will run headless")
                # Reset environment and base agent
                obs = eval_env.reset()
                base_agent.reset()

                episode_reward = 0
                done = False
                step_count = 0

                # Video recording setup for this rollout
                if save_videos:
                    video_obs_list = []
                    # Debug: print available observation keys
                    if rollout_idx == 0 and step_count == 0:
                        print(f"  Available observation keys: {list(obs.keys())}")
                        for key, value in obs.items():
                            if hasattr(value, 'shape'):
                                print(f"    {key}: {value.shape}")

                    # Store initial observation - combine both camera views
                    video_obs = obs.copy()
                    if "color_image1" in video_obs and "color_image2" in video_obs:
                        # Concatenate both camera views side by side
                        img1 = video_obs["color_image1"].cpu().numpy()
                        img2 = video_obs["color_image2"].cpu().numpy()
                        # Remove batch dimension if present
                        if len(img1.shape) == 4 and img1.shape[0] == 1:
                            img1 = img1.squeeze(0)
                        if len(img2.shape) == 4 and img2.shape[0] == 1:
                            img2 = img2.squeeze(0)
                        # Concatenate horizontally (side by side)
                        combined_frame = np.concatenate([img1, img2], axis=1)  # (H, W1+W2, C)
                        video_obs_list.append(combined_frame)
                    elif "color_image1" in video_obs:
                        img1 = video_obs["color_image1"].cpu().numpy()
                        if len(img1.shape) == 4 and img1.shape[0] == 1:
                            img1 = img1.squeeze(0)
                        video_obs_list.append(img1)
                    else:
                        print(f"  No camera observations found for video recording")

                while not done and step_count < rollout_max_steps:
                    # Only convert robot state if it's a dict (not already a tensor)
                    if isinstance(obs["robot_state"], dict):
                        obs["robot_state"] = eval_env.filter_and_concat_robot_state(obs["robot_state"])

                    # Move to device
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor):
                            obs[key] = value.to(device)

                    # Get base action
                    with torch.no_grad():
                        base_naction = base_agent.base_action_normalized(obs)

                        # Process observation for residual policy
                        processed_obs = base_agent.process_obs(obs)
                        residual_obs = torch.cat([processed_obs, base_naction], dim=-1)

                        # Get residual action from INAC
                        residual_action, _ = inac_agent.ac.pi(residual_obs)

                        # Combine base + residual
                        full_naction = base_naction + residual_action * base_agent.residual_policy.action_scale

                        # Denormalize action
                        action = base_agent.normalizer(full_naction, "action", forward=False)

                    # Step environment (handle both old and new gym return formats)
                    step_result = eval_env.step(action)
                    if len(step_result) == 5:
                        obs, reward, done, truncated, info = step_result
                    else:
                        obs, reward, done, info = step_result
                        truncated = False

                    episode_reward += reward.item()
                    step_count += 1

                    # Compute and print CVAE reward for this step
                    if step_count % 20 == 0 or step_count <= 5:  # Print every 20 steps or first 5 steps
                        # Get current state for CVAE reward computation
                        robot_state_dim = 14  # After filtering
                        if isinstance(obs["robot_state"], torch.Tensor):
                            current_robot_state = obs["robot_state"].cpu().numpy().flatten()[:robot_state_dim]
                        else:
                            # Convert dict to tensor if needed
                            robot_state_tensor = eval_env.filter_and_concat_robot_state(obs["robot_state"])
                            current_robot_state = robot_state_tensor.cpu().numpy().flatten()[:robot_state_dim]

                        current_parts_poses = obs["parts_poses"].cpu().numpy().flatten()
                        current_state = np.concatenate([current_robot_state, current_parts_poses])

                        # Get the action that was just executed
                        current_action = action.cpu().numpy().flatten()

                        # Compute CVAE reward using the same system as training
                        try:
                            cvae_reward = furniture_dataset.cvae_reward_computer.compute_progress_reward(
                                current_state, current_action
                            ) * furniture_dataset.reward_scale
                            print(f"    Step {step_count}: Env_reward={reward.item():.3f}, CVAE_reward={cvae_reward:.3f}")
                        except Exception as e:
                            print(f"    Step {step_count}: Env_reward={reward.item():.3f}, CVAE_reward=N/A (error: {e})")

                    # Record video frame if needed
                    if save_videos:
                        if "color_image1" in obs and "color_image2" in obs:
                            # Combine both camera views
                            img1 = obs["color_image1"].cpu().numpy()
                            img2 = obs["color_image2"].cpu().numpy()
                            # Remove batch dimension if present
                            if len(img1.shape) == 4 and img1.shape[0] == 1:
                                img1 = img1.squeeze(0)
                            if len(img2.shape) == 4 and img2.shape[0] == 1:
                                img2 = img2.squeeze(0)
                            # Concatenate horizontally (side by side)
                            combined_frame = np.concatenate([img1, img2], axis=1)
                            video_obs_list.append(combined_frame)
                        elif "color_image1" in obs:
                            img1 = obs["color_image1"].cpu().numpy()
                            if len(img1.shape) == 4 and img1.shape[0] == 1:
                                img1 = img1.squeeze(0)
                            video_obs_list.append(img1)

                    # Handle done signal
                    if isinstance(done, torch.Tensor):
                        done = done.item()
                    if isinstance(truncated, torch.Tensor):
                        truncated = truncated.item()

                    done = done or truncated

                # Check success (reward of 1 means success)
                success = episode_reward >= 1.0
                if success:
                    total_success += 1

                total_returns.append(episode_reward)

                # Save video for this rollout if requested
                if save_videos:
                    print(f"  Video frames collected: {len(video_obs_list)}")
                    if len(video_obs_list) > 0:
                        try:
                            # Debug: check frame properties
                            first_frame = video_obs_list[0]
                            print(f"  First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
                            print(f"  Frame value range: min={first_frame.min()}, max={first_frame.max()}")

                            # Convert frames to numpy array and create video
                            video_frames = np.array(video_obs_list)
                            print(f"  Video array shape: {video_frames.shape}")

                            # Remove batch dimension if present (201, 1, 224, 224, 3) -> (201, 224, 224, 3)
                            if len(video_frames.shape) == 5 and video_frames.shape[1] == 1:
                                video_frames = video_frames.squeeze(1)
                                print(f"  Video array shape after squeeze: {video_frames.shape}")

                            # Ensure frames are in correct format (0-255 uint8)
                            if video_frames.dtype != np.uint8:
                                if video_frames.max() <= 1.0:
                                    # Normalize from [0,1] to [0,255]
                                    video_frames = (video_frames * 255).astype(np.uint8)
                                else:
                                    video_frames = video_frames.astype(np.uint8)

                            print(f"  Creating video with {len(video_frames)} frames...")
                            video = create_in_memory_mp4(video_frames, fps=20)

                            if video.getvalue():
                                # Save video file
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                success_str = "success" if success else "failed"
                                video_filename = f"residual_inac_step{step}_rollout{rollout_idx}_{success_str}_{timestamp}.mp4"
                                video_path = os.path.join(video_save_dir, video_filename)

                                with open(video_path, "wb") as f:
                                    f.write(video.getvalue())
                                print(f"  Video saved: {video_path} ({len(video.getvalue())} bytes)")
                            else:
                                print(f"  Error: Video creation returned empty buffer")

                        except Exception as e:
                            print(f"  Failed to save video for rollout {rollout_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  No video frames collected for rollout {rollout_idx}")

                if (rollout_idx + 1) % 5 == 0:
                    print(f"  Completed {rollout_idx + 1}/{n_rollouts} rollouts")

            # Calculate metrics
            success_rate = total_success / n_rollouts
            mean_return = np.mean(total_returns) if total_returns else 0.0

            print(f"Evaluation results:")
            print(f"  Success rate: {success_rate:.3f} ({total_success}/{n_rollouts})")
            print(f"  Mean return: {mean_return:.3f}")

            # Log evaluation metrics
            wandb.log({
                'eval/success_rate': success_rate,
                'eval/mean_return': mean_return,
                'eval/n_success': total_success,
                'eval/n_rollouts': n_rollouts,
                'eval/step': step
            }, step=step)

    print("Training completed!")

    # Save final model
    model_path = exp_path / 'final_residual_inac_model.pt'
    torch.save({
        'inac_actor_state_dict': inac_agent.ac.pi.state_dict(),
        'inac_critic_state_dict': inac_agent.ac.v.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'state_dim': original_state_dim,
        'action_dim': action_dim,
        'residual_obs_dim': residual_obs_dim
    }, model_path)

    print(f"Final model saved to {model_path}")

    # Clean up evaluation environment
    if eval_env is not None:
        try:
            eval_env.close()
            print("Evaluation environment cleaned up")
        except:
            pass

    wandb.finish()

    return inac_agent, base_agent


if __name__ == "__main__":
    main()