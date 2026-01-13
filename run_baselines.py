#!/usr/bin/env python3
"""
Convenient wrapper to run baseline experiments (BC and IQL) on multiple environments.
"""

import subprocess
import argparse
from pathlib import Path
import sys

# Environment configurations
ENV_CONFIGS = {
    'halfcheetah': 'halfcheetah-medium-expert-v2',
    'walker2d': 'walker2d-medium-expert-v2',
    'ant': 'ant-medium-expert-v2',
    'hopper': 'hopper-medium-expert-v2',
}

# Hyperparameters from IQL paper
IQL_HYPERPARAMS = {
    'halfcheetah-medium-expert-v2': {'tau': 0.7, 'beta': 3.0},
    'walker2d-medium-expert-v2': {'tau': 0.7, 'beta': 3.0},
    'ant-medium-expert-v2': {'tau': 0.7, 'beta': 3.0},
    'hopper-medium-expert-v2': {'tau': 0.7, 'beta': 3.0},
}


def run_bc(env_name, steps=50000, seed=0, log_dir='./baseline_results'):
    """Run BC baseline."""
    print(f"\n{'='*60}")
    print(f"Running BC on {env_name}")
    print(f"{'='*60}\n")

    d4rl_env = ENV_CONFIGS[env_name]

    cmd = [
        'python', 'baselines/bc_baseline.py',
        '--env-name', d4rl_env,
        '--log-dir', log_dir,
        '--seed', str(seed),
        '--n-steps', str(steps),
        '--eval-period', str(max(steps // 10, 1000)),
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    return result.returncode == 0


def run_iql(env_name, steps=1000000, seed=0, log_dir='./baseline_results'):
    """Run IQL baseline using the PyTorch implementation."""
    print(f"\n{'='*60}")
    print(f"Running IQL on {env_name}")
    print(f"{'='*60}\n")

    d4rl_env = ENV_CONFIGS[env_name]
    hyperparams = IQL_HYPERPARAMS[d4rl_env]

    cmd = [
        'python', 'implicit_q_learning/IQL-PyTorch/main.py',
        '--env-name', d4rl_env,
        '--log-dir', log_dir,
        '--seed', str(seed),
        '--n-steps', str(steps),
        '--tau', str(hyperparams['tau']),
        '--beta', str(hyperparams['beta']),
        '--eval-period', str(max(steps // 20, 1000)),
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run baseline experiments')
    parser.add_argument('--method', type=str, required=True, choices=['bc', 'iql', 'both'],
                       help='Which baseline to run')
    parser.add_argument('--env', type=str, nargs='+', required=True,
                       choices=list(ENV_CONFIGS.keys()) + ['all'],
                       help='Environment(s) to run on')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of training steps (default: 50k for BC, 1M for IQL)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--log-dir', type=str, default='./baseline_results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Determine environments to run
    if 'all' in args.env:
        envs = list(ENV_CONFIGS.keys())
    else:
        envs = args.env

    # Determine steps
    bc_steps = args.steps if args.steps else 50000
    iql_steps = args.steps if args.steps else 1000000

    # Run baselines
    results = {}

    for env in envs:
        results[env] = {}

        if args.method in ['bc', 'both']:
            print(f"\n{'#'*60}")
            print(f"# BC - {env.upper()}")
            print(f"{'#'*60}")
            success = run_bc(env, steps=bc_steps, seed=args.seed, log_dir=args.log_dir)
            results[env]['bc'] = 'SUCCESS' if success else 'FAILED'

        if args.method in ['iql', 'both']:
            print(f"\n{'#'*60}")
            print(f"# IQL - {env.upper()}")
            print(f"{'#'*60}")
            success = run_iql(env, steps=iql_steps, seed=args.seed, log_dir=args.log_dir)
            results[env]['iql'] = 'SUCCESS' if success else 'FAILED'

    # Print summary
    print(f"\n\n{'='*60}")
    print("BASELINE EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    for env, methods in results.items():
        print(f"\n{env.upper()}:")
        for method, status in methods.items():
            print(f"  {method.upper()}: {status}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
