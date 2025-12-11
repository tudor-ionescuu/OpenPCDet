#!/bin/bash
#SBATCH --job-name=casa_kd_medium
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/medium/train_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/medium/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

# CasA-V Knowledge Distillation Training Script: Medium Variant
# Follows SparseKD paper - PFE 1.0x, BFE 0.75x, Head 0.75x

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Medium.yaml \
    --batch_size 2 \
    --workers 16 \
    --use_wandb \
    --wandb_project CasA-V-kitti \
    --wandb_name CasA-V-KD-Medium \
    --extra_tag kd_medium
