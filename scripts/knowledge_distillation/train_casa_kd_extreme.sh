#!/bin/bash
#SBATCH --job-name=casa_kd_extreme
#SBATCH --output=logs/casa/Knowledge-Distillation/ultralight/casa_kd_extreme_%j.out
#SBATCH --error=logs/casa/Knowledge-Distillation/ultralight/casa_kd_extreme_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Extreme.yaml \
    --batch_size 2 \
    --workers 16 \
    --use_wandb \
    --wandb_project CasA-V-kitti \
    --wandb_name CasA-V-KD-Extreme \
    --extra_tag kd_extreme
