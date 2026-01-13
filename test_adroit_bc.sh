#!/bin/bash
# Test BC on Adroit environments

cd /home/swaminathan/git/FurnitureTransfer

echo "============================================================"
echo "Testing BC + CVAE on Adroit Environments"
echo "============================================================"
echo ""

# Test on pen-human
echo "1. Testing on pen-human..."
echo "------------------------------------------------------------"
python test_higher_dim_latent_cvae.py \
    --env pen \
    --dataset human \
    --mode full \
    --cvae-epochs 50 \
    --bc-epochs 100 \
    --eval-episodes 10

echo ""
echo "============================================================"
echo "pen-human test complete!"
echo "============================================================"
