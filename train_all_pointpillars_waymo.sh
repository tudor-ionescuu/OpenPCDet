#!/bin/bash
#SBATCH --job-name=pp_all_waymo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=48
#SBATCH --mem=200GB
#SBATCH --time=72:00:00
#SBATCH --output=dcgm/pp_all_waymo/train_%j.out
#SBATCH --error=dcgm/pp_all_waymo/train_%j.err
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

# Create output directories
mkdir -p ../dcgm/pp_all_waymo

# Train all 3 PointPillar variants in parallel on Waymo
echo "Starting parallel training of all 3 PointPillar variants on Waymo..."

# Full PointPillar on GPU 0 (Best accuracy, slowest)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_1x.yaml \
    --batch_size 2 \
    --epochs 30 \
    --workers 16 \
    --extra_tag waymo_parallel_full \
    2>&1 | tee ../dcgm/pp_all_waymo/train_full.log &

# Lite PointPillar on GPU 1 (Good accuracy, faster)
CUDA_VISIBLE_DEVICES=1 python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_lite.yaml \
    --batch_size 4 \
    --epochs 30 \
    --workers 16 \
    --extra_tag waymo_parallel_lite \
    2>&1 | tee ../dcgm/pp_all_waymo/train_lite.log &

# Ultralight PointPillar on GPU 2 (Fastest, lower accuracy)
CUDA_VISIBLE_DEVICES=2 python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_ultralight.yaml \
    --batch_size 6 \
    --epochs 30 \
    --workers 16 \
    --extra_tag waymo_parallel_ultralight \
    2>&1 | tee ../dcgm/pp_all_waymo/train_ultralight.log &

# Wait for all background jobs to complete
wait

echo "All trainings completed!"
