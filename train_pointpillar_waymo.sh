#!/bin/bash
#SBATCH --job-name=pp_waymo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=dcgm/pp_waymo/train_%j.out
#SBATCH --error=dcgm/pp_waymo/train_%j.err
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

# Create output directory
mkdir -p ../dcgm/pp_waymo

# Train PointPillar Full on Waymo
echo "Starting training of PointPillar Full on Waymo..."
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_1x.yaml \
    --batch_size 2 \
    --epochs 30 \
    --workers 16 \
    --extra_tag waymo_full \
    2>&1 | tee ../dcgm/pp_waymo/train_full.log

echo "Training completed! Log saved to dcgm/pp_waymo/train_full.log"
