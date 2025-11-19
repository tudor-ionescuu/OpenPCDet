#!/bin/bash
#SBATCH --job-name=test_pp_ultralight
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100|a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_pointpillar_ultralight_%j.out
#SBATCH --error=logs/test_pointpillar_ultralight_%j.err
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

# Test PointPillar Ultralight
echo "Testing PointPillar Ultralight..."
python test.py \
    --cfg_file cfgs/nuscenes_models/pointpillar_ultralight.yaml \
    --batch_size 1 \
    --ckpt ../output/nuscenes_models/pointpillar_ultralight/default/ckpt/checkpoint_epoch_20.pth \
    --infer_time

echo "Test completed!"
