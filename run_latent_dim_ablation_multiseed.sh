#!/bin/bash
# Latent Dimension Ablation Study with Multiple Seeds
# Proper scientific methodology: average over multiple random seeds

cd /home/swaminathan/git/FurnitureTransfer

ENV="walker2d"
DATASET="medium-replay"
ACTION_DIM=6

# Latent dimensions to test
LATENT_EXTRAS=(-2 -1 0 1 2 4)  # [4, 5, 6, 7, 8, 10]

# Multiple seeds for reliability
SEEDS=(0 1 2 3 4)

echo "============================================================"
echo "Latent Dimension Ablation Study (Multi-Seed)"
echo "Environment: ${ENV}-${DATASET}"
echo "Action dim: ${ACTION_DIM}"
echo "Testing latent dims: [4, 5, 6, 7, 8, 10]"
echo "Seeds: ${SEEDS[@]}"
echo "Total runs: $((${#LATENT_EXTRAS[@]} * ${#SEEDS[@]})) = 30 runs"
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
echo "Total runs completed: ${TOTAL_RUNS}"
echo "Results saved in: ./outputs/latent_cvae_comparison/"
echo ""
echo "Next step: Aggregate results across seeds for each latent dim"
echo "============================================================"
