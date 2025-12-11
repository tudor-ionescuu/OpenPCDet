#!/bin/bash
#SBATCH --job-name=eval_kd_micro
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/micro/eval_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/micro/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

# CasA-V Knowledge Distillation Evaluation Script: Micro Variant
# Evaluates checkpoint from epoch 80 with batch_size 1 for inference timing

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Micro.yaml \
    --ckpt output/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Micro/kd_micro/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 \
    --workers 16 \
    --infer_time
