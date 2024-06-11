#!/bin/bash
set -e

# Step 1.
# CUDA_VISIBLE_DEVICES=0 python run_net.py --mode train --cfg configs/resnet/r-56_c4.yaml --output_dir work_dirs_resnet
# Step 2.
# We could test that train the local branch with the original dataset, and train the classifer with augmented dataset.
# CUDA_VISIBLE_DEVICES=0 python run_net.py --mode train --cfg configs/pvtv2/pvtv2-b0_c4_ours.yaml --output_dir work_dirs

# Test.
# CUDA_VISIBLE_DEVICES=0 python run_net.py --mode test --cfg configs/pvtv2/pvtv2-b0_c4_ours.yaml TEST.WEIGHTS work_dirs/pvtv2-b0_c4_ours/model.pyth


# Train the ResNet-18 model with the original dataset.
CUDA_VISIBLE_DEVICES=0 python run_net.py --mode train --cfg configs/resnet/r-18_c4_x2.yaml --output_dir work_dirs_resnet