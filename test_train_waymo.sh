#!/bin/bash
#SBATCH --job-name=pp_waymo_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_waymo_train_%j.out
#SBATCH --error=logs/test_waymo_train_%j.err
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

# Quick test training with PointPillar Lite (2 epochs only)
echo "Starting quick test training of PointPillar Lite on Waymo (2 epochs)..."
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_lite.yaml \
    --batch_size 2 \
    --epochs 2 \
    --workers 8 \
    --extra_tag waymo_lite_test

echo "Test training completed!"
