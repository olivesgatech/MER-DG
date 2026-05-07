#!/bin/bash

# Define paths and domains
DATAPATH="/home/yavuz/data/EPIC/frames_rgb_flow/"
SOURCE_DOMAINS="D1 D2"
TARGET_DOMAIN="D3"

echo "=========================================================="
echo "1. Baseline Fusion (Standard Cross-Entropy)"
echo "=========================================================="
python train_mer_dg.py \
  --source_domain $SOURCE_DOMAINS \
  --target_domain $TARGET_DOMAIN \
  --datapath $DATAPATH \
  --use_video --use_audio --use_flow \
  --lr 1e-4 --bsz 16 --nepochs 20 \
  --alpha_contrast 0.0 \
  --lambda_mer 0.0 \
  --use_wandb --wandb_project "MER-DG-EPIC" --appen "_Baseline_Fusion"

echo "=========================================================="
echo "2. Baseline Fusion + MER-DG (Ours)"
echo "=========================================================="
python train_mer_dg.py \
  --source_domain $SOURCE_DOMAINS \
  --target_domain $TARGET_DOMAIN \
  --datapath $DATAPATH \
  --use_video --use_audio --use_flow \
  --lr 1e-4 --bsz 16 --nepochs 20 \
  --alpha_contrast 0.0 \
  --lambda_mer 3.0 --alpha_marg 1.0 --alpha_spec 1.0 \
  --use_wandb --wandb_project "MER-DG-EPIC" --appen "_Baseline_MERDG"

echo "=========================================================="
echo "3. SimMMDG Baseline"
echo "=========================================================="
python train_video_flow_audio_EPIC_MERDG.py \
  --source_domain $SOURCE_DOMAINS \
  --target_domain $TARGET_DOMAIN \
  --datapath $DATAPATH \
  --use_video --use_audio --use_flow \
  --lr 1e-4 --bsz 16 --nepochs 20 \
  --alpha_contrast 3.0 --alpha_trans 0.1 \
  --lambda_mer 0.0 \
  --use_wandb --wandb_project "MER-DG-EPIC" --appen "_SimMMDG"

echo "=========================================================="
echo "4. SimMMDG + MER-DG"
echo "=========================================================="
python train_video_flow_audio_EPIC_MERDG.py \
  --source_domain $SOURCE_DOMAINS \
  --target_domain $TARGET_DOMAIN \
  --datapath $DATAPATH \
  --use_video --use_audio --use_flow \
  --lr 1e-4 --bsz 16 --nepochs 20 \
  --alpha_contrast 3.0 --alpha_trans 0.1 \
  --lambda_mer 3.0 --alpha_marg 1.0 --alpha_spec 1.0 \
  --use_wandb --wandb_project "MER-DG-EPIC" --appen "_SimMMDG_MERDG"

