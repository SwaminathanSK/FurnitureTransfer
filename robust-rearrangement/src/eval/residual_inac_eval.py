"""
Evaluation script for Residual INAC with CVAE rewards.

This script evaluates the trained residual INAC policy by:
1. Loading the frozen base policy and trained residual INAC policy
2. Running rollouts in the environment
3. Computing success rates and other metrics
4. Recording videos and logging to wandb
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

# Add INAC codebase to path
sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

# Robust-rearrangement imports
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.eval.eval_utils import get_model_from_api_or_cached
from src.common.config_util import merge_base_bc_config_with_root_config
from src.gym import get_rl_env
from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.gym.observation import DEFAULT_STATE_OBS
from src.eval.rollout import calculate_success_rate

# INAC imports
try:
    from core.agent.in_sample import InSampleAC
    from core.utils import torch_utils
except ImportError as e:
    print(f"INAC import failed: {e}")
    print("Please ensure INAC_MLRC_24 is properly installed")
    sys.exit(1)

# Register eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


class ResidualINACEvaluator:
    """
    Evaluator for residual INAC policies.
    """

    def __init__(
        self,
        base_agent: ResidualDiffusionPolicy,
        inac_model_path: str,
        inac_config: Dict,
        device: str = 'cuda'
    ):
        self.base_agent = base_agent
        self.device = device

        # Load INAC model
        print(f"Loading INAC model from {inac_model_path}")
        checkpoint = torch.load(inac_model_path, map_location=device)

        # Reconstruct INAC agent
        residual_obs_dim = checkpoint['residual_obs_dim']
        action_dim = checkpoint['action_dim']

        # Create dummy INAC agent for loading weights
        self.inac_agent = InSampleAC(
            device=str(device),
            discrete_control=0,
            state_dim=residual_obs_dim,
            action_dim=action_dim,
            hidden_units=inac_config.get('hidden_units', 256),
            learning_rate=inac_config.get('learning_rate', 3e-4),
            tau=inac_config.get('tau', 0.1),
            polyak=inac_config.get('polyak', 0.995),
            exp_path='./temp',
            seed=0,
            env_fn=lambda: None,
            timeout=1000,
            gamma=inac_config.get('gamma', 0.99),
            offline_data={'env': {'states': np.zeros((1, residual_obs_dim)),
                                 'actions': np.zeros((1, action_dim)),
                                 'rewards': np.zeros(1),
                                 'next_states': np.zeros((1, residual_obs_dim)),
                                 'terminations': np.zeros(1, dtype=bool)}},
            batch_size=inac_config.get('batch_size', 256),
            use_target_network=1,
            target_network_update_freq=1,
            evaluation_criteria='return',
            logger=None,
            lambdaVal='none'
        )

        # Load weights
        self.inac_agent.ac.pi.load_state_dict(checkpoint['inac_actor_state_dict'])
        self.inac_agent.ac.v.load_state_dict(checkpoint['inac_critic_state_dict'])

        print(f"INAC model loaded successfully")
        print(f"Residual obs dim: {residual_obs_dim}, Action dim: {action_dim}")

        # Freeze base policy
        for param in self.base_agent.model.parameters():
            param.requires_grad = False
        for param in self.base_agent.normalizer.parameters():
            param.requires_grad = False

    def predict_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Predict action using base + residual policy.

        Args:
            obs: Environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Combined action (base + residual)
        """
        with torch.no_grad():
            # Get base action
            base_naction = self.base_agent.base_action_normalized(obs)

            # Process observation for residual policy
            processed_obs = self.base_agent.process_obs(obs)
            residual_obs = torch.cat([processed_obs, base_naction], dim=-1)

            # Get residual action from INAC
            if deterministic:
                residual_action = self.inac_agent.ac.pi(residual_obs)[0]  # Mean action
            else:
                residual_action, _ = self.inac_agent.ac.pi(residual_obs)  # Sample action

            # Combine base and residual
            full_naction = base_naction + residual_action * self.base_agent.residual_policy.action_scale

            # Denormalize
            return self.base_agent.normalizer(full_naction, "action", forward=False)

    def reset(self):
        """Reset the agent state."""
        self.base_agent.reset()


class ResidualINACActor:
    """
    Actor wrapper compatible with robust-rearrangement evaluation functions.
    """

    def __init__(self, evaluator: ResidualINACEvaluator):
        self.evaluator = evaluator
        self.normalizer = evaluator.base_agent.normalizer  # Required by evaluation
        self.model = self  # Self-reference for compatibility

    def action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Get action from observation (main interface used by rollout)."""
        return self.evaluator.predict_action(obs, deterministic=deterministic)

    def reset(self):
        """Reset the actor."""
        self.evaluator.reset()

    def to(self, device):
        """Move to device (compatibility)."""
        return self

    def set_task(self, task_idx):
        """Set task (compatibility)."""
        pass


def evaluate_residual_inac(
    base_policy_path: str,
    inac_model_path: str,
    n_rollouts: int = 10,
    task: str = "one_leg",
    gpu_id: int = 0,
    save_videos: bool = True,
    wandb_logging: bool = True
) -> Dict:
    """
    Evaluate residual INAC policy in environment.

    Args:
        base_policy_path: Path to base policy weights or wandb run id
        inac_model_path: Path to trained INAC model
        n_rollouts: Number of evaluation rollouts
        task: Task name
        gpu_id: GPU ID to use
        save_videos: Whether to save evaluation videos
        wandb_logging: Whether to log to wandb

    Returns:
        Evaluation metrics
    """

    device = torch.device(f"cuda:{gpu_id}")

    # Load base policy config and weights
    if base_policy_path.startswith('swami2004/'):
        # Wandb run ID
        base_cfg, base_wts = get_model_from_api_or_cached(
            base_policy_path,
            wt_type='best',
            wandb_mode='online'
        )
    else:
        # Local file path
        base_wts = base_policy_path
        base_cfg = OmegaConf.create(torch.load(base_wts)["config"])

    # Initialize base agent
    base_agent = ResidualDiffusionPolicy(device, base_cfg)
    base_agent.load_base_state_dict(base_wts)
    base_agent.to(device)
    base_agent.eval()

    print(f"Base policy loaded from {base_policy_path}")

    # Load INAC model config
    inac_checkpoint = torch.load(inac_model_path, map_location='cpu')
    inac_config = inac_checkpoint.get('config', {}).get('inac', {})

    # Initialize evaluator
    evaluator = ResidualINACEvaluator(
        base_agent=base_agent,
        inac_model_path=inac_model_path,
        inac_config=inac_config,
        device=str(device)
    )

    # Create actor wrapper
    actor = ResidualINACActor(evaluator)

    # Create environment
    env = get_rl_env(
        gpu_id=gpu_id,
        task=task,
        num_envs=1,
        randomness="low",
        observation_space="state_vision" if save_videos else "state",
        max_env_steps=200,
        resize_img=save_videos,
        act_rot_repr="rot_6d",
        action_type="pos",
        april_tags=False,
        verbose=False,
        headless=not save_videos
    )

    print(f"Environment created for task: {task}")

    # Set up evaluation directory
    eval_dir = Path("./residual_inac_eval_results") / f"eval_{int(time.time())}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if wandb_logging:
        wandb.init(
            project="residual_inac_evaluation",
            name=f"eval_residual_inac_{task}_{int(time.time())}",
            config={
                'base_policy_path': base_policy_path,
                'inac_model_path': inac_model_path,
                'task': task,
                'n_rollouts': n_rollouts,
                'gpu_id': gpu_id
            }
        )

    # Run evaluation
    print(f"Running {n_rollouts} evaluation rollouts...")

    rollout_stats = calculate_success_rate(
        env=env,
        actor=actor,
        n_rollouts=n_rollouts,
        rollout_max_steps=200,
        epoch_idx=0,
        rollout_save_dir=eval_dir if save_videos else None,
        save_rollouts_to_wandb=wandb_logging,
        save_failures=True,
        n_parts_assemble=1,
        compress_pickles=False,
        resize_video=save_videos
    )

    # Compute metrics
    eval_metrics = {
        "success_rate": rollout_stats.success_rate,
        "mean_return": rollout_stats.total_return / rollout_stats.n_rollouts,
        "n_rollouts": rollout_stats.n_rollouts,
        "n_success": rollout_stats.n_success,
        "rollout_max_steps": rollout_stats.rollout_max_steps
    }

    print(f"Evaluation Results:")
    print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
    print(f"  Mean Return: {eval_metrics['mean_return']:.3f}")
    print(f"  Successful Rollouts: {eval_metrics['n_success']}/{eval_metrics['n_rollouts']}")

    # Log to wandb
    if wandb_logging:
        wandb.log(eval_metrics)
        wandb.finish()

    # Save results
    results_path = eval_dir / "eval_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Results saved to {results_path}")

    return eval_metrics


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    Main evaluation function.
    """

    # Default paths - modify as needed
    base_policy_path = getattr(cfg, 'base_policy_path', 'swami2004/your_base_policy_run_id')
    inac_model_path = getattr(cfg, 'inac_model_path', './residual_inac_cvae_results/final_residual_inac_model.pt')

    # Evaluation parameters
    n_rollouts = getattr(cfg, 'n_rollouts', 10)
    task = getattr(cfg.env, 'task', 'one_leg')
    gpu_id = getattr(cfg, 'gpu_id', 0)
    save_videos = getattr(cfg, 'save_videos', True)
    wandb_logging = getattr(cfg, 'wandb_logging', True)

    # Run evaluation
    eval_metrics = evaluate_residual_inac(
        base_policy_path=base_policy_path,
        inac_model_path=inac_model_path,
        n_rollouts=n_rollouts,
        task=task,
        gpu_id=gpu_id,
        save_videos=save_videos,
        wandb_logging=wandb_logging
    )

    return eval_metrics


if __name__ == "__main__":
    main()