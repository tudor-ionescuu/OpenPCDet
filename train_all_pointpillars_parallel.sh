#!/bin/bash
#SBATCH --job-name=pp_all_parallel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=96:00:00
#SBATCH --output=logs/pointpillar_all_parallel_%j.out
#SBATCH --error=logs/pointpillar_all_parallel_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge
module load cuda/12.1.1

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Set nuScenes dataset path
export NUSCENES_DATA_PATH=/ibex/project/c2337/datasets/nuscenes

# Train all 4 models in parallel, each on its own GPU
echo "Starting parallel training of all 4 PointPillar variants..."

# Full PointPillar on GPU 0 (Best accuracy, slowest)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg_file cfgs/nuscenes_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 20 \
    --workers 16 \
    --extra_tag parallel_full &

# Medium PointPillar on GPU 1 (Balanced accuracy/speed)
CUDA_VISIBLE_DEVICES=1 python train.py \
    --cfg_file cfgs/nuscenes_models/pointpillar_medium.yaml \
    --batch_size 6 \
    --epochs 20 \
    --workers 16 \
    --extra_tag parallel_medium &

# Lite PointPillar on GPU 2 (Good accuracy, faster)
CUDA_VISIBLE_DEVICES=2 python train.py \
    --cfg_file cfgs/nuscenes_models/pointpillar_lite.yaml \
    --batch_size 8 \
    --epochs 20 \
    --workers 16 \
    --extra_tag parallel_lite &

# Ultralight PointPillar on GPU 3 (Fastest, lower accuracy)
CUDA_VISIBLE_DEVICES=3 python train.py \
    --cfg_file cfgs/nuscenes_models/pointpillar_ultralight.yaml \
    --batch_size 12 \
    --epochs 20 \
    --workers 16 \
    --extra_tag parallel_ultralight &

# Wait for all background jobs to complete
wait

echo "All trainings completed!"
