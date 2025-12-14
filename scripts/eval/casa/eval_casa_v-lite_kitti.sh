#!/bin/bash
#SBATCH --job-name=test_casa_v-lite
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=logs/casa/lite/%j.out
#SBATCH --error=logs/casa/lite/%j.err
#SBATCH --account=pi-shokera

module purge

source ~/Code/openpcdet_project/openpcdet-env/bin/activate

export LD_LIBRARY_PATH=~/Code/openpcdet_project/openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0

cd ~/Code/openpcdet_project/OpenPCDet

python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/CasA-V_Lite.yaml \
    --batch_size 1 \
    --ckpt output/cfgs/kitti_models/CasA-V_Lite/lite_full/ckpt/checkpoint_epoch_80.pth \
    --infer_time
