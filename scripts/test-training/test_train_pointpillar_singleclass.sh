#!/bin/bash
#SBATCH --job-name=test_pp_single
#SBATCH --output=logs/pointpillar/test_pp_single_%j.out
#SBATCH --error=logs/pointpillar/test_pp_single_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet/tools

python train.py \
    --cfg_file cfgs/kitti_models/pointpillar_singleclass.yaml \
    --batch_size 2 \
    --workers 8 \
    --epochs 2 \
    --extra_tag test_singleclass
