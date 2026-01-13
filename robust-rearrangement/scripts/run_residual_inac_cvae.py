#!/usr/bin/env python3
"""
Convenience script for running residual INAC training and evaluation with CVAE rewards.

This script provides an easy interface to:
1. Train a residual INAC policy using CVAE progress rewards
2. Evaluate the trained policy
3. Handle different configuration options

Usage:
    # Train with default config
    python scripts/run_residual_inac_cvae.py train

    # Train with custom base policy
    python scripts/run_residual_inac_cvae.py train --base_policy_id swami2004/your_bc_run_id

    # Evaluate trained model
    python scripts/run_residual_inac_cvae.py eval --model_path ./path/to/model.pt

    # Train and then evaluate
    python scripts/run_residual_inac_cvae.py train_and_eval
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

# Add robust-rearrangement to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def train_residual_inac(
    base_policy_id: str = None,
    base_policy_path: str = None,
    cvae_model_path: str = None,
    max_steps: int = None,
    config_overrides: str = None
):
    """
    Train residual INAC with CVAE rewards.

    Args:
        base_policy_id: Wandb run ID of base policy
        base_policy_path: Local path to base policy weights
        cvae_model_path: Path to trained CVAE model
        max_steps: Maximum training steps
        config_overrides: Additional hydra config overrides
    """

    cmd = [
        "python", "src/train/residual_inac_cvae.py",
        "--config-name", "residual_inac_cvae"
    ]

    # Add config overrides
    if base_policy_id:
        cmd.extend([f"base_policy.wandb_id={base_policy_id}"])

    if base_policy_path:
        cmd.extend([f"base_policy.wt_path={base_policy_path}"])

    if cvae_model_path:
        cmd.extend([f"cvae.model_path={cvae_model_path}"])

    if max_steps:
        cmd.extend([f"inac.max_steps={max_steps}"])

    if config_overrides:
        cmd.extend(config_overrides.split())

    print(f"Running command: {' '.join(cmd)}")

    # Change to robust-rearrangement directory
    os.chdir(Path(__file__).parent.parent)

    # Run training
    result = subprocess.run(cmd)
    return result.returncode == 0


def evaluate_residual_inac(
    model_path: str,
    base_policy_id: str = None,
    base_policy_path: str = None,
    n_rollouts: int = 10,
    save_videos: bool = True,
    config_overrides: str = None
):
    """
    Evaluate trained residual INAC model.

    Args:
        model_path: Path to trained INAC model
        base_policy_id: Wandb run ID of base policy
        base_policy_path: Local path to base policy weights
        n_rollouts: Number of evaluation rollouts
        save_videos: Whether to save videos
        config_overrides: Additional hydra config overrides
    """

    cmd = [
        "python", "src/eval/residual_inac_eval.py",
        "--config-name", "residual_inac_cvae",
        f"inac_model_path={model_path}",
        f"n_rollouts={n_rollouts}",
        f"save_videos={save_videos}"
    ]

    # Add config overrides
    if base_policy_id:
        cmd.extend([f"base_policy_path={base_policy_id}"])

    if base_policy_path:
        cmd.extend([f"base_policy_path={base_policy_path}"])

    if config_overrides:
        cmd.extend(config_overrides.split())

    print(f"Running command: {' '.join(cmd)}")

    # Change to robust-rearrangement directory
    os.chdir(Path(__file__).parent.parent)

    # Run evaluation
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run residual INAC with CVAE rewards')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train residual INAC policy')
    train_parser.add_argument('--base_policy_id', type=str,
                             help='Wandb run ID of base policy (e.g., swami2004/your_run_id)')
    train_parser.add_argument('--base_policy_path', type=str,
                             help='Local path to base policy weights')
    train_parser.add_argument('--cvae_model_path', type=str,
                             default='./corrected_progress_cvae_results/corrected_progress_cvae_model.pt',
                             help='Path to trained CVAE model')
    train_parser.add_argument('--max_steps', type=int, default=100000,
                             help='Maximum training steps')
    train_parser.add_argument('--config_overrides', type=str,
                             help='Additional hydra config overrides (space-separated)')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained INAC model')
    eval_parser.add_argument('--base_policy_id', type=str,
                            help='Wandb run ID of base policy')
    eval_parser.add_argument('--base_policy_path', type=str,
                            help='Local path to base policy weights')
    eval_parser.add_argument('--n_rollouts', type=int, default=10,
                            help='Number of evaluation rollouts')
    eval_parser.add_argument('--save_videos', action='store_true', default=True,
                            help='Save evaluation videos')
    eval_parser.add_argument('--config_overrides', type=str,
                            help='Additional hydra config overrides (space-separated)')

    # Train and eval command
    train_eval_parser = subparsers.add_parser('train_and_eval',
                                             help='Train and then evaluate')
    train_eval_parser.add_argument('--base_policy_id', type=str,
                                  help='Wandb run ID of base policy')
    train_eval_parser.add_argument('--base_policy_path', type=str,
                                  help='Local path to base policy weights')
    train_eval_parser.add_argument('--cvae_model_path', type=str,
                                  default='./corrected_progress_cvae_results/corrected_progress_cvae_model.pt',
                                  help='Path to trained CVAE model')
    train_eval_parser.add_argument('--max_steps', type=int, default=100000,
                                  help='Maximum training steps')
    train_eval_parser.add_argument('--n_rollouts', type=int, default=10,
                                  help='Number of evaluation rollouts')
    train_eval_parser.add_argument('--config_overrides', type=str,
                                  help='Additional hydra config overrides (space-separated)')

    args = parser.parse_args()

    if args.command == 'train':
        print("Starting residual INAC training with CVAE rewards...")
        success = train_residual_inac(
            base_policy_id=args.base_policy_id,
            base_policy_path=args.base_policy_path,
            cvae_model_path=args.cvae_model_path,
            max_steps=args.max_steps,
            config_overrides=args.config_overrides
        )

        if success:
            print("Training completed successfully!")
        else:
            print("Training failed!")
            sys.exit(1)

    elif args.command == 'eval':
        print("Starting evaluation...")
        success = evaluate_residual_inac(
            model_path=args.model_path,
            base_policy_id=args.base_policy_id,
            base_policy_path=args.base_policy_path,
            n_rollouts=args.n_rollouts,
            save_videos=args.save_videos,
            config_overrides=args.config_overrides
        )

        if success:
            print("Evaluation completed successfully!")
        else:
            print("Evaluation failed!")
            sys.exit(1)

    elif args.command == 'train_and_eval':
        print("Starting training and evaluation...")

        # First, train
        train_success = train_residual_inac(
            base_policy_id=args.base_policy_id,
            base_policy_path=args.base_policy_path,
            cvae_model_path=args.cvae_model_path,
            max_steps=args.max_steps,
            config_overrides=args.config_overrides
        )

        if not train_success:
            print("Training failed!")
            sys.exit(1)

        print("Training completed! Starting evaluation...")

        # Find the most recent model
        results_dir = Path("./residual_inac_cvae_results")
        if results_dir.exists():
            model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                model_path = latest_dir / "final_residual_inac_model.pt"

                if model_path.exists():
                    eval_success = evaluate_residual_inac(
                        model_path=str(model_path),
                        base_policy_id=args.base_policy_id,
                        base_policy_path=args.base_policy_path,
                        n_rollouts=args.n_rollouts,
                        save_videos=True,
                        config_overrides=args.config_overrides
                    )

                    if eval_success:
                        print("Training and evaluation completed successfully!")
                    else:
                        print("Evaluation failed!")
                        sys.exit(1)
                else:
                    print(f"Model file not found: {model_path}")
                    sys.exit(1)
            else:
                print("No training results found!")
                sys.exit(1)
        else:
            print("Results directory not found!")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()