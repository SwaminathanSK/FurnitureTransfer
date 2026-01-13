# 2-Week Experimental Plan - Ready-to-Run Commands

## Setup Complete!
- ✅ CVAE + INAC implementation ready
- ✅ BC baseline (simple, proven implementation)
- ✅ IQL baseline (official PyTorch port from IQL paper authors)
- ✅ Reward visualization tools
- ✅ Multi-environment support: HalfCheetah, Walker2d, Ant, Hopper

---

## Week 1: Core Experiments (Days 1-7)

### Priority: Get multi-environment results

### **Days 1-3: Run HalfCheetah, Walker2d, Ant with CVAE+INAC**

#### Quick tests (10k steps, ~15 min each):
```bash
# All environments - quick test
for env in halfcheetah walker2d ant hopper; do
  python test_cvae_inac_simple.py --env $env --steps 10000 --wandb --record_video
done
```

#### Full training (50k steps, ~1-2 hours each):
```bash
# HalfCheetah
python test_cvae_inac_simple.py --env halfcheetah --steps 50000 --wandb --record_video

# Walker2d
python test_cvae_inac_simple.py --env walker2d --steps 50000 --wandb --record_video

# Ant
python test_cvae_inac_simple.py --env ant --steps 50000 --wandb --record_video

# Hopper
python test_cvae_inac_simple.py --env hopper --steps 50000 --wandb --record_video
```

#### Reuse pre-trained CVAEs (saves ~10-15 min per run):
```bash
python test_cvae_inac_simple.py --env halfcheetah --steps 50000 --wandb \
  --cvae_checkpoint ./cvae_checkpoints/halfcheetah/cvae_model.pt
```

---

### **Days 4-5: Implement/run key baselines**

#### BC Baseline (50k steps, ~30-45 min each):
```bash
# Single environment
python run_baselines.py --method bc --env halfcheetah --steps 50000

# All environments
python run_baselines.py --method bc --env all --steps 50000

# Specific environments
python run_baselines.py --method bc --env halfcheetah walker2d ant --steps 50000
```

#### IQL Baseline (1M steps recommended, ~3-4 hours each):
```bash
# Single environment (full training)
python run_baselines.py --method iql --env halfcheetah --steps 1000000

# Quick test (100k steps, ~30 min)
python run_baselines.py --method iql --env halfcheetah --steps 100000

# All environments (full training)
python run_baselines.py --method iql --env all --steps 1000000
```

#### Run both BC and IQL together:
```bash
# This will run BC (50k) then IQL (1M) on each environment
python run_baselines.py --method both --env all
```

---

### **Days 6-7: Buffer for failed runs, debugging**

Check results and re-run any failed experiments:
```bash
# Check what results exist
ls -R baseline_results/
ls -R videos/
ls -R cvae_checkpoints/

# Re-run specific failed runs with different seeds
python test_cvae_inac_simple.py --env halfcheetah --steps 50000 --wandb
python run_baselines.py --method bc --env walker2d --seed 1
```

---

## Week 2: Ablations & Analysis (Days 8-14)

### **Days 8-10: Ablation studies**

Create ablation variants and run them. You'll need to modify `simple_cvae_mujoco.py` to create ablation versions:

#### 1. Remove progress loss (set gamma=0 for progress terms):
Edit line 199-202 in `simple_cvae_mujoco.py`:
```python
total_loss = (recon_loss +
             beta * kl_loss +
             0.0 * progress_supervision_loss +      # ABLATION: no progress loss
             0.0 * progress_prediction_loss)        # ABLATION: no progress loss
```

#### 2. Remove reconstruction loss (set recon weight=0):
```python
total_loss = (0.0 * recon_loss +                   # ABLATION: no recon loss
             beta * kl_loss +
             gamma * progress_supervision_loss +
             gamma * progress_prediction_loss)
```

#### 3. Vary progress loss weight:
Test with gamma values: [0.1, 0.5, 1.0, 5.0, 10.0]

Run ablations on 1-2 environments (HalfCheetah + Walker2d recommended):
```bash
# After making ablation changes, run:
python test_cvae_inac_simple.py --env halfcheetah --steps 50000 --wandb
python test_cvae_inac_simple.py --env walker2d --steps 50000 --wandb
```

---

### **Days 11-13: Reward visualization/analysis**

#### Visualize learned rewards vs true rewards:
```bash
# Generate reward comparison plots
python visualize_rewards.py --env halfcheetah --episodes 5
python visualize_rewards.py --env walker2d --episodes 5
python visualize_rewards.py --env ant --episodes 5
python visualize_rewards.py --env hopper --episodes 5

# Use pre-trained CVAE
python visualize_rewards.py --env halfcheetah --episodes 10 \
  --cvae_checkpoint ./cvae_checkpoints/halfcheetah/cvae_model.pt
```

Outputs:
- Plots saved to: `./reward_visualizations/{env}_reward_comparison.png`
- Shows: actual env rewards, CVAE rewards, cumulative returns, progress signals
- Prints: correlation statistics, reward ranges

---

### **Day 14: Compile results, identify gaps**

Check all experimental results:
```bash
# View baseline results
python -c "
import numpy as np
from pathlib import Path

for env in ['halfcheetah', 'walker2d', 'ant', 'hopper']:
    bc_path = Path(f'baseline_results/{env}-medium-expert-v2/bc_seed0/results.npz')
    if bc_path.exists():
        data = np.load(bc_path)
        print(f'{env} BC: {data[\"normalized_return_mean\"][-1]:.2f}')
"

# Compare with CVAE+INAC results in wandb dashboard
```

Create comparison table (manual or script):
| Environment | BC | IQL | CVAE+INAC |
|-------------|----|----|-----------|
| HalfCheetah | ?? | ?? | ??        |
| Walker2d    | ?? | ?? | ??        |
| Ant         | ?? | ?? | ??        |
| Hopper      | ?? | ?? | ??        |

---

## Visualization & Monitoring

### WandB Dashboard:
- CVAE training: loss curves, progress correlation
- INAC training: actor/critic losses, returns
- Evaluation: episode returns over time

### Videos:
```bash
# Videos saved to:
ls videos/halfcheetah/
ls videos/walker2d/
ls videos/ant/
ls videos/hopper/
```

### Reward Visualizations:
```bash
# Reward comparison plots saved to:
ls reward_visualizations/
```

---

## Quick Reference

### File Structure
```
├── test_cvae_inac_simple.py          # Main CVAE+INAC training
├── simple_cvae_mujoco.py             # CVAE implementation
├── visualize_rewards.py              # Reward visualization
├── run_baselines.py                  # Baseline runner wrapper
├── baselines/
│   └── bc_baseline.py                # BC implementation
├── implicit_q_learning/IQL-PyTorch/  # IQL implementation
├── baseline_results/                 # Baseline outputs
├── cvae_checkpoints/                 # Saved CVAE models
├── videos/                           # Evaluation videos
└── reward_visualizations/            # Reward plots
```

### Environments Available
- `halfcheetah` → halfcheetah-medium-expert-v2
- `walker2d` → walker2d-medium-expert-v2
- `ant` → ant-medium-expert-v2
- `hopper` → hopper-medium-expert-v2

### Common Flags
- `--wandb` : Enable WandB logging
- `--record_video` : Record evaluation videos
- `--cvae_checkpoint PATH` : Use pre-trained CVAE
- `--steps N` : Number of training steps
- `--seed N` : Random seed

---

## Parallel Execution (if you have multiple GPUs/terminals)

```bash
# Terminal 1
python test_cvae_inac_simple.py --env halfcheetah --steps 50000 --wandb

# Terminal 2
python test_cvae_inac_simple.py --env walker2d --steps 50000 --wandb

# Terminal 3
python run_baselines.py --method bc --env ant --steps 50000

# Terminal 4
python run_baselines.py --method iql --env hopper --steps 100000
```

---

## Expected Results (from papers)

### IQL (from official paper):
- HalfCheetah-medium-expert: ~86-88
- Walker2d-medium-expert: ~109-110
- Hopper-medium-expert: ~91
- Ant-medium-expert: Not in main paper

### BC (typical baseline):
- Generally lower than IQL
- HalfCheetah: ~40-50
- Walker2d: ~80-90

### CVAE+INAC (your method):
- Goal: Competitive with or better than IQL
- Key metric: Can we learn good rewards from progress?

---

## Troubleshooting

### Out of memory:
- Reduce `--batch-size` (default 256 → try 128)
- Reduce `--hidden-dim` (default 256 → try 128)

### Slow training:
- Use `--cvae_checkpoint` to skip CVAE training
- Reduce `--steps` for quick tests
- Cached CVAE rewards will speed up INAC training

### WandB issues:
- Run without `--wandb` flag
- Results still saved locally

---

**You're ready to start experiments! All code uses established implementations.**
