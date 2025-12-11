#!/bin/bash
#SBATCH --job-name=eval_kd_small
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/small/eval_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/small/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

# CasA-V Knowledge Distillation Evaluation Script: Small Variant
# Evaluates checkpoint from epoch 80 with batch_size 1 for inference timing

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Small.yaml \
    --ckpt output/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Small/kd_small/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 \
    --workers 16 \
    --infer_time
