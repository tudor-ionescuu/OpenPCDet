#!/bin/bash
#SBATCH --job-name=pp_waymo_wandb_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --output=logs/test_waymo_wandb_%j.out
#SBATCH --error=logs/test_waymo_wandb_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge
module load cuda/12.1.1

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Install wandb if not already installed
pip install wandb --quiet

# Change to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Set Waymo dataset path
export WAYMO_DATA_PATH=/ibex/project/c2337/datasets/waymo

# Set wandb to offline mode for testing (no login required)
export WANDB_MODE=offline

# Test training with wandb logging (5 epochs to test early stopping with patience=3)
echo "Starting test training with wandb logging (5 epochs)..."
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_lite.yaml \
    --batch_size 2 \
    --epochs 5 \
    --workers 8 \
    --extra_tag waymo_lite_wandb_test \
    --use_wandb \
    --wandb_project pointpillars-waymo-test \
    --early_stopping \
    --early_stopping_patience 3

echo "Test training with wandb completed!"
