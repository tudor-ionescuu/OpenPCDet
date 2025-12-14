#!/bin/bash
#SBATCH --job-name=test_pp_lite_kitti
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_pointpillar_lite_kitti_%j.out
#SBATCH --error=logs/test_pointpillar_lite_kitti_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge
module avail 2>&1 | grep -i cuda
find /usr/local/cuda* -name "libnvvm.so" 2>/dev/null | head -1

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Set CUDA library paths for numba
export LD_LIBRARY_PATH=~/Code/openpcdet_project/openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Test PointPillar Lite on KITTI
echo "Testing PointPillar Lite on KITTI..."
# Run inference with GPU
python test.py \
    --cfg_file cfgs/kitti_models/pointpillar_lite.yaml \
    --batch_size 1 \
    --ckpt ../pretrained_models/kitti/pointpillar_lite.pth \
    --infer_time

echo "Test completed!"
