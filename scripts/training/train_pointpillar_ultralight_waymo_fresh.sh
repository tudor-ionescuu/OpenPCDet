#!/bin/bash
#SBATCH --job-name=pp_ultra_fresh_1VFE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --output=dcgm/pp_ultra_waymo/train_fresh_1VFE_%j.out
#SBATCH --error=dcgm/pp_ultra_waymo/train_fresh_1VFE_%j.err
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

# Resume training of PointPillar Ultralight on Waymo
echo "Resuming training of PointPillar Ultralight on Waymo from epoch 30..."
python train.py \
    --cfg_file cfgs/waymo_models/pointpillar_ultralight.yaml \
    --epochs 60 \
    --workers 16 \
    --extra_tag waymo_ultralight_1vfe \
    --use_wandb \
    --wandb_project pointpillars-waymo-1vfe \
    --early_stopping \
    --early_stopping_patience 6 \
    2>&1 | tee ../dcgm/pp_ultra_waymo/train_ultralight_resume.log

echo "Training completed! Log saved to dcgm/pp_ultra_waymo/train_ultralight_resume.log"
