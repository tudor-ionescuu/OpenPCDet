#!/bin/bash
#SBATCH --job-name=test_ted_s
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=logs/ted_s_test_%j.out
#SBATCH --error=logs/ted_s_test_%j.err
#SBATCH --account=pi-shokera

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Set CUDA library paths
export LD_LIBRARY_PATH=~/Code/openpcdet_project/openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Link to KITTI data on project storage
rm -f ../data/kitti
ln -s /ibex/project/c2337/openpcdet_data/kitti ../data/kitti

# Test TED-S on KITTI
python test.py \
    --cfg_file cfgs/kitti_models/TED-S.yaml \
    --batch_size 1 \
    --ckpt ../pretrained_models/kitti/TED-S.pth \
    --infer_time
