#!/bin/bash
#SBATCH --job-name=eval_casa_singleclass
#SBATCH --output=logs/casa/eval_casa_singleclass_%j.out
#SBATCH --error=logs/casa/eval_casa_singleclass_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

# CasA-V Single Class Evaluation
# Evaluates the trained single-class model on KITTI validation set

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/CasA-V_SingleClass.yaml \
    --ckpt output/cfgs/kitti_models/CasA-V_SingleClass/single_class/ckpt/checkpoint_epoch_80.pth \
    --batch_size 1 \
    --workers 8 \
    --extra_tag eval_single_class \
    --infer_time
