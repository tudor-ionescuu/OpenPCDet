#!/bin/bash
#SBATCH --job-name=casa_kd_micro
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/micro/train_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/micro/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

# CasA-V Knowledge Distillation Training Script: Micro Variant
# Follows SparseKD paper - PFE 0.375x, BFE 0.25x, Head 0.125x
# More aggressive than Extreme

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Micro.yaml \
    --batch_size 2 \
    --workers 16 \
    --use_wandb \
    --wandb_project CasA-V-kitti \
    --wandb_name CasA-V-KD-Micro \
    --extra_tag kd_micro
