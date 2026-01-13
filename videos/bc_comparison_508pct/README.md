# Walker2d-Medium-Replay: Latent CVAE BC vs Standard BC

## Experiment Details

**Run ID:** `walker2d_full_20251022_022935`

**Environment:** walker2d-medium-replay-v2

**Date Generated:** November 4, 2025

## Model Checkpoints

The trained models used for these videos are located at:
```
outputs/latent_cvae_comparison/walker2d_full_20251022_022935/
├── cvae_walker2d.pt           # CVAE with latent_dim = 7 (action_dim=6 + 1)
├── bc_latent_walker2d.pt      # Latent BC policy (state -> latent_action)
└── bc_standard_walker2d.pt    # Standard BC policy (state -> action)
```

## Performance Results

### Training Results (from results_walker2d.txt)

- **CVAE Latent Dim:** 7 (action_dim=6 + 1)
- **Effective Dimensionality:** 6.01/7 (no collapse!)
- **Evaluation Episodes:** 100

| Method | Mean Return | Std Dev |
|--------|-------------|---------|
| Standard BC | 337.35 | ± 373.95 |
| Latent BC | 2049.98 | ± 1357.63 |
| **Improvement** | **+507.67%** | |

### Video Recording Results (10 episodes)

| Method | Mean Return | Std Dev |
|--------|-------------|---------|
| Standard BC | 305.80 | ± 94.23 |
| Latent BC | 1854.35 | ± 1357.92 |
| **Improvement** | **+506.39%** | |

The video recording results closely match the original training evaluation results, confirming the massive performance improvement!

## Video Files

### Latent BC Videos (10 episodes)
Located in: `walker2d_full_20251022_022935/latent_bc/`

Episodes sorted by performance:
1. `latent_bc_ep02_return3830.mp4` - **Best performance** (3830)
2. `latent_bc_ep06_return3528.mp4` - 3528
3. `latent_bc_ep07_return3390.mp4` - 3390
4. `latent_bc_ep04_return3129.mp4` - 3129
5. `latent_bc_ep01_return1525.mp4` - 1525
6. `latent_bc_ep00_return904.mp4` - 904
7. `latent_bc_ep03_return678.mp4` - 678
8. `latent_bc_ep05_return538.mp4` - 538
9. `latent_bc_ep08_return524.mp4` - 524
10. `latent_bc_ep09_return492.mp4` - 492

### Standard BC Videos (10 episodes)
Located in: `walker2d_full_20251022_022935/standard_bc/`

Episodes sorted by performance:
1. `standard_bc_ep06_return522.mp4` - **Best performance** (522)
2. `standard_bc_ep08_return427.mp4` - 427
3. `standard_bc_ep00_return367.mp4` - 367
4. `standard_bc_ep05_return252.mp4` - 252
5. `standard_bc_ep09_return252.mp4` - 252
6. `standard_bc_ep03_return249.mp4` - 249
7. `standard_bc_ep02_return247.mp4` - 247
8. `standard_bc_ep07_return247.mp4` - 247
9. `standard_bc_ep04_return245.mp4` - 245
10. `standard_bc_ep01_return243.mp4` - 243

## Key Observations

1. **Latent BC significantly outperforms Standard BC**: The latent BC achieves returns of up to 3830, while standard BC maxes out at 522.

2. **Consistent improvement**: Even the worst latent BC episode (492) outperforms most standard BC episodes.

3. **Higher variance in latent BC**: The latent BC shows higher variance (±1357.92) compared to standard BC (±94.23), suggesting that while it can achieve very high performance, it can also have some lower-performing episodes.

4. **No dimensional collapse**: The CVAE maintained an effective dimensionality of 6.01/7, meaning all latent dimensions were utilized effectively.

## Method Summary

**Latent CVAE BC** works as follows:
1. Train a CVAE to encode (state, action) pairs into a 7-dimensional latent space
2. Train a BC policy to predict the latent action from the state
3. At test time: state → BC policy → latent action → CVAE decoder → real action

**Standard BC** directly maps state to action:
- state → BC policy → action

The extra latent dimension (beyond the action dimension) appears to capture task-relevant information that helps the BC policy learn more effectively.

## Reproducing the Videos

To regenerate these videos, run:
```bash
python record_bc_comparison_videos.py \
  --run-dir outputs/latent_cvae_comparison/walker2d_full_20251022_022935 \
  --env walker2d \
  --dataset medium-replay \
  --episodes 10 \
  --output videos/bc_comparison_508pct
```
