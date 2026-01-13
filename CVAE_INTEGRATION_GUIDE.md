# CVAE Integration with Offline RL Methods

This guide provides step-by-step instructions for integrating your pre-trained CVAE with different offline RL methods.

## Setup: Clone Repositories

```bash
cd ~/git/FurnitureTransfer
mkdir -p baselines
cd baselines

# 1. CQL (Conservative Q-Learning)
git clone https://github.com/young-geng/CQL.git
# Alternative (official): git clone https://github.com/aviralkumar2907/CQL.git

# 2. IQL (Implicit Q-Learning)
git clone https://github.com/ikostrikov/implicit_q_learning.git
# PyTorch version: git clone https://github.com/gwthomas/IQL-PyTorch.git

# 3. Action Chunking Transformer (ACT)
git clone https://github.com/tonyzhaozh/act.git
```

## Priority Order

### 🥇 #1: CVAE + CQL (Start Here!)

**Repo:** `young-geng/CQL` (cleaner than official)
**Estimated time:** 3-4 hours
**Expected improvement:** +20-40% over CQL baseline

**Key files to modify:**
- `SimpleSAC/conservative_sac.py` - Add latent action space
- `SimpleSAC/replay_buffer.py` - Store latent actions
- `SimpleSAC/model.py` - Integrate CVAE decoder

**Integration strategy:**
1. Load your pre-trained CVAE
2. Pre-compute latent actions for entire dataset
3. Modify Q-network: `Q(s, latent_action)` instead of `Q(s, action)`
4. Modify actor: outputs `latent_action`, then CVAE decodes to `action`
5. Keep CVAE frozen during training

### 🥈 #2: CVAE + IQL (Most Reliable)

**Repo:** `gwthomas/IQL-PyTorch` (cleaner PyTorch implementation)
**Estimated time:** 2-3 hours
**Expected improvement:** +10-25% over IQL baseline

**Key files to modify:**
- `iql/agent.py` - Modify Q-network and policy
- `iql/policy.py` - Output latent actions

**Integration strategy:**
1. Same as CQL but simpler (no conservative penalty complications)
2. IQL uses expectile regression, easier to adapt
3. Policy network outputs latent, CVAE decodes

### 🥉 #3: CVAE + Action Chunking BC (Most Interesting)

**Repo:** `tonyzhaozh/act`
**Estimated time:** 4-5 hours (robot sim environment)
**Expected improvement:** +15-30% on complex tasks

**Note:** ACT is designed for robot manipulation tasks, not D4RL. You'd need to:
1. Adapt ACT to D4RL environments
2. Modify transformer to output latent action sequences
3. CVAE decodes each latent in sequence

**This is more experimental** but could be very impactful for real robotics!

---

## Recommended Repos Summary

| Method | Best Repo | Language | D4RL Support | Difficulty |
|--------|-----------|----------|--------------|------------|
| CQL | `young-geng/CQL` | PyTorch | ✅ Native | Medium |
| IQL | `gwthomas/IQL-PyTorch` | PyTorch | ✅ Native | Easy |
| ACT | `tonyzhaozh/act` | PyTorch | ❌ (robot only) | Hard |

---

## Alternative: CORL Framework

If you want a **unified framework** for all methods:

```bash
git clone https://github.com/corl-team/CORL.git
```

**CORL (Clean Offline RL)** provides:
- ✅ CQL, IQL, TD3+BC, AWAC all in one codebase
- ✅ Clean PyTorch implementations
- ✅ D4RL integration
- ✅ Standardized configs

**Trade-off:**
- Pros: Easier to compare methods, consistent codebase
- Cons: Might be harder to deeply modify for CVAE integration

---

## Next Steps

I'll create detailed integration scripts for each method. Which one do you want to start with?

1. **CQL** - Highest expected impact
2. **IQL** - Easiest to implement
3. **CORL** - All methods in one framework
