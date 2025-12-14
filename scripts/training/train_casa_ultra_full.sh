#!/bin/bash
#SBATCH --job-name=casa_ultra_full
#SBATCH --output=logs/casa/casa_ultra_full_%j.out
#SBATCH --error=logs/casa/casa_ultra_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/CasA-V_Ultralight.yaml \
    --batch_size 2 \
    --workers 16 \
    --use_wandb \
    --wandb_project CasA-V-kitti \
    --wandb_name CasA-V-Ultralight-Full \
    --extra_tag ultralight_full
