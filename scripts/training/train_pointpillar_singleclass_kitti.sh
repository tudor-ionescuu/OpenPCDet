#!/bin/bash
#SBATCH --job-name=pp_singleclass
#SBATCH --output=logs/pointpillar/pp_singleclass_%j.out
#SBATCH --error=logs/pointpillar/pp_singleclass_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet/tools

python train.py \
    --cfg_file cfgs/kitti_models/pointpillar_singleclass.yaml \
    --batch_size 4 \
    --workers 16 \
    --use_wandb \
    --wandb_project pointpillars-kitti \
    --wandb_name pp-full-object \
    --extra_tag singleclass_object
