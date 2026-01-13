#!/bin/bash
# CQL vs CQL+CVAE Comparison Script

cd /home/swaminathan/git/FurnitureTransfer/baselines/CORL

ENV="walker2d-medium-expert-v2"
TIMESTEPS=1000000
EVAL_FREQ=5000
SEED=0
PROJECT="cql-cvae-comparison"

# Path to pre-trained CVAE checkpoint for Walker2d
CVAE_CHECKPOINT="/home/swaminathan/git/FurnitureTransfer/outputs/latent_cvae_comparison/walker2d_full_20251021_080210/cvae_walker2d.pt"

echo "============================================================"
echo "CQL vs CQL+CVAE Comparison on $ENV"
echo "============================================================"
echo ""

# Run Standard CQL (Baseline)
echo "1. Running Standard CQL (Baseline)..."
echo "------------------------------------------------------------"
python algorithms/offline/cql.py \
    --env $ENV \
    --seed $SEED \
    --eval_freq $EVAL_FREQ \
    --max_timesteps $TIMESTEPS \
    --project $PROJECT \
    --group standard-cql \
    --name standard-cql

echo ""
echo "============================================================"
echo "Standard CQL training complete!"
echo "============================================================"
echo ""

# Run CQL + CVAE
echo "2. Running CQL + CVAE (Latent Action Space)..."
echo "------------------------------------------------------------"
python algorithms/offline/cql_cvae.py \
    --env $ENV \
    --seed $SEED \
    --eval_freq $EVAL_FREQ \
    --max_timesteps $TIMESTEPS \
    --project $PROJECT \
    --group cql-cvae \
    --name cql-cvae \
    --cvae_checkpoint $CVAE_CHECKPOINT

echo ""
echo "============================================================"
echo "CQL + CVAE training complete!"
echo "============================================================"
echo ""
echo "Both experiments finished! Check W&B for comparison:"
echo "Project: $PROJECT"
echo "Groups: standard-cql vs cql-cvae"
echo "============================================================"
