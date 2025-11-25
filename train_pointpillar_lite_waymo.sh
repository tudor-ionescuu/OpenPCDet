#!/bin/bash
#SBATCH --job-name=pp_lite_waymo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --output=dcgm/pp_lite_waymo/train_%j.out
#SBATCH --error=dcgm/pp_lite_waymo/train_%j.err
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

# Create output directory
mkdir -p ../dcgm/pp_lite_waymo

# Train PointPillar Lite on Waymo with wandb and early stopping
echo "Starting training of PointPillar Lite on Waymo..."
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_lite.yaml \
    --batch_size 2 \
    --epochs 30 \
    --workers 16 \
    --extra_tag waymo_lite \
    --use_wandb \
    --wandb_project pointpillars-waymo \
    --early_stopping \
    --early_stopping_patience 7 \
    2>&1 | tee ../dcgm/pp_lite_waymo/train_lite.log

echo "Training completed! Log saved to dcgm/pp_lite_waymo/train_lite.log"
