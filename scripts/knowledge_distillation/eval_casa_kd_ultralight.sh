#!/bin/bash
#SBATCH --job-name=eval_casa_kd_ultra
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/ultralight/eval_kd_ultra_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/casa/Knowledge-Distillation/ultralight/eval_kd_ultra_%j.err
#SBATCH --account=pi-shokera

module purge

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

export LD_LIBRARY_PATH=/home/ionesctn/Code/openpcdet_project/openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet

# Evaluate KD-Ultralight Model (SparseKD width compression)
python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Ultralight.yaml \
    --batch_size 1 \
    --ckpt output/cfgs/kitti_models/knowledge_distillation/CasA-V_KD-Ultralight/kd_ultralight/ckpt/checkpoint_epoch_80.pth \
    --infer_time
