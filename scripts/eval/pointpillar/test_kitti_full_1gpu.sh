#!/bin/bash
#SBATCH --job-name=kitti_full_1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=logs/pointpillar/full/kitti/1gpu_%j.out
#SBATCH --error=logs/pointpillar/full/kitti/1gpu_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Set CUDA library paths for numba
export LD_LIBRARY_PATH=~/Code/openpcdet_project/openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Test PointPillar Full on KITTI
echo "Testing PointPillar Full on KITTI with 1 GPU..."
python test.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size 1 \
    --ckpt ../pretrained_models/kitti/pointpillar_full.pth \
    --infer_time

echo "Test completed!"
