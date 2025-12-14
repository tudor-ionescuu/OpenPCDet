#!/bin/bash
#SBATCH --job-name=eval_pp_full
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/pointpillar/eval_full_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/pointpillar/eval_full_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet/tools

python test.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --ckpt ../pretrained_models/kitti/pointpillar_full.pth \
    --batch_size 1 \
    --workers 8 \
    --infer_time
