#!/bin/bash
#SBATCH --job-name=pp_lite_nuscenes
#SBATCH --output=logs/pointpillar_lite_nuscenes_%j.out
#SBATCH --error=logs/pointpillar_lite_nuscenes_%j.err
#SBATCH --time=96:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --partition=batch
#SBATCH --constraint=a100

# Load modules
module purge
module load cuda/12.1.1

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Set nuScenes dataset path
export NUSCENES_DATA_PATH=/ibex/project/c2337/datasets/nuscenes

# Train PointPillar Lite on nuScenes
python train.py \
    --cfg_file cfgs/nuscenes_models/pointpillar_lite.yaml \
    --batch_size 8 \
    --epochs 20 \
    --workers 16

echo "Training completed!"
