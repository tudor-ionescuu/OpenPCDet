#!/bin/bash
#SBATCH --job-name=pp_full_fresh_p20
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --output=dcgm/pp_waymo/train_fresh_p20_%j.out
#SBATCH --error=dcgm/pp_waymo/train_fresh_p20_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge
module load cuda/12.1.1

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Set Waymo dataset path
export WAYMO_DATA_PATH=/ibex/project/c2337/datasets/waymo

# Train PointPillar Full on Waymo
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_1x.yaml \
    --epochs 60 \
    --workers 16 \
    --extra_tag waymo_full_p20 \
    --use_wandb \
    --wandb_project pointpillars-waymo\
    --early_stopping \
    --early_stopping_patience 20 \

echo "Training completed!"
