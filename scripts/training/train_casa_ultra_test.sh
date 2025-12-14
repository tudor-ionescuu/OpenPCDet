#!/bin/bash
#SBATCH --job-name=casa_ultra_test
#SBATCH --output=logs/casa/casa_ultra_test_%j.out
#SBATCH --error=logs/casa/casa_ultra_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/train.py \
    --cfg_file tools/cfgs/kitti_models/CasA-V_Ultralight.yaml \
    --epochs 3 \
    --batch_size 2 \
    --workers 4 \
    --use_wandb \
    --wandb_project CasA-V-kitti
