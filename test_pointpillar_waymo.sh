#!/bin/bash
#SBATCH --job-name=test_pp_waymo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100|a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=dcgm/test_pp_waymo/test_%j.out
#SBATCH --error=dcgm/test_pp_waymo/test_%j.err
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
mkdir -p ../dcgm/test_pp_waymo

# Test PointPillar Full on Waymo
echo "Testing PointPillar Full on Waymo..."
python test.py \
    --cfg_file cfgs/waymo_models/pointpillar_1x.yaml \
    --batch_size 1 \
    --ckpt ../output/waymo_models/pointpillar_1x/waymo_full/ckpt/checkpoint_epoch_30.pth \
    --infer_time \
    2>&1 | tee ../dcgm/test_pp_waymo/test_full.log

echo "Test completed! Log saved to dcgm/test_pp_waymo/test_full.log"
