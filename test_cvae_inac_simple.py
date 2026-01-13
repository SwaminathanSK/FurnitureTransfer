#!/usr/bin/env python3
"""
Quick test script for CVAE + INAC pipeline on MuJoCo environments.
This is a simplified version for rapid prototyping and testing.

Usage:
    python test_cvae_inac_simple.py --env halfcheetah --steps 10000
"""

# IMPORTANT: Set MuJoCo rendering BEFORE any imports
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use CPU rendering to avoid GPU segfault
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import argparse
import numpy as np
import torch
import gym  # Use old gym for D4RL compatibility
import d4rl
from pathlib import Path
import sys
import wandb

# Add INAC to path
sys.path.append('/home/swaminathan/git/FurnitureTransfer/INAC_MLRC_24')

# Import the main components
from simple_cvae_mujoco import SimpleMuJoCoDataset, train_cvae, train_inac_with_cvae_rewards, evaluate_policy, load_cvae_from_checkpoint


def quick_test(env_name: str, max_steps: int = 10000, eval_episodes: int = 5, use_wandb: bool = False, record_video: bool = False, cvae_checkpoint: str = None,
               beta: float = 0.01, gamma: float = 1.0, no_recon_loss: bool = False, no_progress_loss: bool = False, run_id: str = None):
    """
    Quick test of CVAE + INAC pipeline with minimal training for fast iteration.

    Args:
        cvae_checkpoint: Path to pre-trained CVAE checkpoint. If provided, skips CVAE training.
        beta: Weight for KL divergence loss
        gamma: Weight for progress losses
        no_recon_loss: If True, removes reconstruction loss (ablation study)
        no_progress_loss: If True, removes progress supervision (tests latent-only rewards)
        run_id: Optional identifier for this run (for multiple runs of same ablation)
    """
    print(f"=== Quick Test: {env_name} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate run ID if not provided
    if run_id is None:
        import time
        run_id = f"run_{int(time.time())}"

    if record_video:
        print("Video recording enabled - will save evaluation videos")

    # Initialize wandb if requested
    if use_wandb:
        import time
        run_name = f"{int(time.time())}_cvae_inac_{env_name}"
        wandb.init(
            project="mujoco-cvae-inac",
            name=run_name,
            config={
                'env_name': env_name,
                'max_steps': max_steps,
                'eval_episodes': eval_episodes,
                'device': device,
                'hidden_dim': 128,
                'num_epochs': 20,
                'batch_size': 128
            }
        )

    # Load dataset (using smaller subset for speed)
    print("Loading dataset...")
    dataset = SimpleMuJoCoDataset(env_name, "medium-expert")

    # Check if we should load from checkpoint or train new
    if cvae_checkpoint and os.path.exists(cvae_checkpoint):
        print(f"\nLoading CVAE from checkpoint: {cvae_checkpoint}")
        cvae_model = load_cvae_from_checkpoint(cvae_checkpoint, device=device)
    else:
        if cvae_checkpoint:
            print(f"WARNING: Checkpoint {cvae_checkpoint} not found, training from scratch...")

        # Use smaller subset for quick testing
        subset_data = dataset.get_train_data(subset_fraction=0.1)  # Use only 10% of data
        print(f"Using {len(subset_data['states'])} samples for quick test")

        # Train CVAE (fewer epochs for speed)
        print("\nTraining CVAE (quick)...")
        cvae_model = train_cvae(
            dataset,
            hidden_dim=128,  # Smaller network
            num_epochs=20,   # Fewer epochs
            batch_size=128,  # Smaller batch
            device=device,
            use_wandb=use_wandb,
            beta=beta,
            gamma=gamma,
            use_recon_loss=not no_recon_loss,
            use_progress_loss=not no_progress_loss
        )

        # Save CVAE model immediately after training with ablation-specific name
        os.makedirs(f"./cvae_checkpoints/{env_name}", exist_ok=True)

        # Create filename suffix based on ablation settings
        ablation_suffix = []
        if no_recon_loss:
            ablation_suffix.append("norecon")
        if no_progress_loss:
            ablation_suffix.append("noprog")
        if gamma != 1.0:
            ablation_suffix.append(f"gamma{gamma}")
        if beta != 0.01:
            ablation_suffix.append(f"beta{beta}")

        suffix_str = "_" + "_".join(ablation_suffix) if ablation_suffix else ""
        cvae_save_path = f"./cvae_checkpoints/{env_name}/cvae_model{suffix_str}_{run_id}.pt"
        save_dict = {
            'model_state_dict': cvae_model.state_dict(),
            'state_dim': dataset.state_dim,
            'action_dim': dataset.action_dim,
            'hidden_dim': 128
        }
        # Save normalization stats if they exist
        if hasattr(cvae_model, 'state_mean'):
            save_dict['state_mean'] = cvae_model.state_mean.cpu()
            save_dict['state_std'] = cvae_model.state_std.cpu()
            save_dict['action_mean'] = cvae_model.action_mean.cpu()
            save_dict['action_std'] = cvae_model.action_std.cpu()
        torch.save(save_dict, cvae_save_path)
        print(f"✓ CVAE model saved to {cvae_save_path}")

    # Train INAC (fewer steps for speed)
    print("\nTraining INAC (quick)...")

    # Set up rewards cache path with ablation-specific name
    ablation_suffix = []
    if no_recon_loss:
        ablation_suffix.append("norecon")
    if no_progress_loss:
        ablation_suffix.append("noprog")
    if gamma != 1.0:
        ablation_suffix.append(f"gamma{gamma}")
    if beta != 0.01:
        ablation_suffix.append(f"beta{beta}")

    suffix_str = "_" + "_".join(ablation_suffix) if ablation_suffix else ""
    rewards_cache_path = f"./cvae_checkpoints/{env_name}/cvae_rewards_cache{suffix_str}_{run_id}.npz"

    inac_agent = train_inac_with_cvae_rewards(
        dataset,
        cvae_model,
        max_steps=max_steps,
        batch_size=128,  # Smaller batch
        device=device,
        use_wandb=use_wandb,
        log_interval=100,  # More frequent logging
        eval_interval=2000,  # Evaluate every 2000 steps
        eval_episodes=eval_episodes,
        env_name=env_name,
        rewards_cache_path=rewards_cache_path
    )

    # Quick evaluation
    print("\nEvaluating...")
    eval_results = evaluate_policy(inac_agent, env_name, n_episodes=eval_episodes,
                                   record_video=record_video, video_folder=f"./videos/{env_name}")

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'eval/final_mean_return': eval_results['mean_return'],
            'eval/final_std_return': eval_results['std_return'],
            'eval/final_mean_length': eval_results['mean_length']
        })
        wandb.finish()

    return eval_results


def compare_environments():
    """Compare performance across multiple MuJoCo environments."""
    environments = ["halfcheetah", "hopper", "walker2d"]
    results = {}

    for env_name in environments:
        print(f"\n{'='*50}")
        print(f"Testing {env_name.upper()}")
        print(f"{'='*50}")

        try:
            result = quick_test(env_name, max_steps=5000, eval_episodes=3)
            results[env_name] = result
            print(f"{env_name}: Mean Return = {result['mean_return']:.2f}")
        except Exception as e:
            print(f"Failed to test {env_name}: {e}")
            results[env_name] = None

    print(f"\n{'='*50}")
    print("SUMMARY RESULTS")
    print(f"{'='*50}")
    for env_name, result in results.items():
        if result is not None:
            print(f"{env_name:12}: {result['mean_return']:8.2f} ± {result['std_return']:6.2f}")
        else:
            print(f"{env_name:12}: FAILED")


def test_cvae_reward_quality(env_name: str = "halfcheetah"):
    """
    Test the quality of CVAE progress predictions before using for RL.
    This helps debug if the CVAE is learning meaningful progress signals.
    """
    print(f"=== Testing CVAE Reward Quality: {env_name} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = SimpleMuJoCoDataset(env_name, "medium-expert")

    # Train CVAE
    print("Training CVAE...")
    cvae_model = train_cvae(dataset, num_epochs=50, device=device)

    # Test progress predictions on some episodes
    print("\nTesting progress predictions on sample episodes...")

    from simple_cvae_mujoco import CVAERewardComputer
    reward_computer = CVAERewardComputer(cvae_model, device)

    # Test on first few episodes
    for ep_idx in range(min(3, dataset.n_episodes)):
        start = dataset.episode_starts[ep_idx]
        end = dataset.episode_ends[ep_idx]

        episode_states = dataset.states[start:end]
        true_progress = dataset.progress_labels[start:end]

        predicted_rewards = []
        for state in episode_states:
            reward = reward_computer.compute_progress_reward(state)
            predicted_rewards.append(reward)

        predicted_rewards = np.array(predicted_rewards)

        print(f"\nEpisode {ep_idx + 1} (length: {len(episode_states)}):")
        print(f"  True progress:      {true_progress[0]:.3f} -> {true_progress[-1]:.3f}")
        print(f"  Predicted rewards:  {predicted_rewards[0]:.3f} -> {predicted_rewards[-1]:.3f}")
        print(f"  Correlation:        {np.corrcoef(true_progress, predicted_rewards)[0,1]:.3f}")

        # Check if rewards are increasing (good sign)
        increasing = predicted_rewards[-1] > predicted_rewards[0]
        print(f"  Rewards increasing: {increasing}")

    return cvae_model


def main():
    parser = argparse.ArgumentParser(description='Test CVAE + INAC on MuJoCo')
    parser.add_argument('--env', type=str, default='halfcheetah',
                       choices=['halfcheetah', 'hopper', 'walker2d', 'ant'],
                       help='MuJoCo environment to test')
    parser.add_argument('--steps', type=int, default=10000,
                       help='INAC training steps')
    parser.add_argument('--eval_episodes', type=int, default=5,
                       help='Episodes for evaluation')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare', 'cvae_test'],
                       help='Test mode')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--record_video', action='store_true',
                       help='Record evaluation videos')
    parser.add_argument('--cvae_checkpoint', type=str, default=None,
                       help='Path to pre-trained CVAE checkpoint (skips CVAE training)')

    # Ablation study parameters
    parser.add_argument('--beta', type=float, default=0.01,
                       help='Weight for KL divergence loss (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Weight for progress losses (default: 1.0). Try: 0.1, 0.5, 1.0, 2.0, 5.0')
    parser.add_argument('--no_recon_loss', action='store_true',
                       help='Remove reconstruction loss (tests if action reconstruction is needed)')
    parser.add_argument('--no_progress_loss', action='store_true',
                       help='Remove progress supervision (tests if latent encoding alone works)')

    args = parser.parse_args()

    if args.mode == 'single':
        result = quick_test(args.env, args.steps, args.eval_episodes,
                          use_wandb=args.wandb, record_video=args.record_video,
                          cvae_checkpoint=args.cvae_checkpoint,
                          beta=args.beta, gamma=args.gamma,
                          no_recon_loss=args.no_recon_loss,
                          no_progress_loss=args.no_progress_loss)
        print(f"\nFinal Result for {args.env}:")
        print(f"Mean Return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")

    elif args.mode == 'compare':
        compare_environments()

    elif args.mode == 'cvae_test':
        cvae_model = test_cvae_reward_quality(args.env)
        print("CVAE quality test completed.")


if __name__ == "__main__":
    main()