#!/bin/bash
# Efficient Latent Dimension Ablation Study
# Standard BC is saved/loaded per seed - only trained once!

cd /home/swaminathan/git/FurnitureTransfer

ENV="walker2d"
DATASET="medium-replay"
ACTION_DIM=6

# Latent dimensions to test
LATENT_EXTRAS=(-2 -1 0 1 2 4)  # [4, 5, 6, 7, 8, 10]

# Multiple seeds for reliability (ML papers convention)
SEEDS=(0 1 2 42 123)

echo "============================================================"
echo "Efficient Latent Dimension Ablation Study"
echo "Environment: ${ENV}-${DATASET}"
echo "Action dim: ${ACTION_DIM}"
echo "Testing latent dims: [4, 5, 6, 7, 8, 10]"
echo "Seeds: ${SEEDS[@]}"
echo "Total runs: $((${#LATENT_EXTRAS[@]} * ${#SEEDS[@]}))"
echo "============================================================"
echo ""
echo "NOTE: Standard BC will be saved per seed and reused!"
echo "      This avoids redundant training across latent dims."
echo "============================================================"
echo ""

TOTAL_RUNS=$((${#LATENT_EXTRAS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

for EXTRA in "${LATENT_EXTRAS[@]}"; do
    LATENT_DIM=$((ACTION_DIM + EXTRA))

    echo ""
    echo "========================================================"
    echo "Latent Dim = ${LATENT_DIM} (action_dim ${EXTRA:+$EXTRA})"
    echo "========================================================"

    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo ""
        echo "--------------------------------------------------------"
        echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}: Latent=${LATENT_DIM}, Seed=${SEED}"
        echo "--------------------------------------------------------"

        python test_higher_dim_latent_cvae.py \
            --env ${ENV} \
            --dataset ${DATASET} \
            --mode full \
            --latent-extra ${EXTRA} \
            --cvae-epochs 50 \
            --bc-epochs 100 \
            --eval-episodes 100 \
            --seed ${SEED} \
            --no-wandb

        echo "Completed: Latent=${LATENT_DIM}, Seed=${SEED}"
    done
done

echo ""
echo "============================================================"
echo "Ablation study complete!"
echo "============================================================"
echo ""
echo "Total runs: ${TOTAL_RUNS}"
echo "Standard BC trained: ${#SEEDS[@]} times (once per seed)"
echo "CVAE+BC variants trained: ${TOTAL_RUNS} times"
echo "Results saved in: ./outputs/latent_cvae_comparison/"
echo ""
echo "Time saved by checkpoint reuse:"
echo "  Without checkpoints: ~$((TOTAL_RUNS * 30)) min"
echo "  With checkpoints: ~$(((TOTAL_RUNS - ${#SEEDS[@]}) * 25 + ${#SEEDS[@]} * 30)) min"
echo "  Savings: ~$(((TOTAL_RUNS - ${#SEEDS[@]}) * 5)) min (~$((((TOTAL_RUNS - ${#SEEDS[@]}) * 5) / 60)) hours)"
echo "============================================================"
