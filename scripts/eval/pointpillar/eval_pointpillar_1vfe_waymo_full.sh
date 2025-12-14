#!/bin/bash
#SBATCH --job-name=eval_pp_1vfe_full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=dcgm/eval_pp_1vfe_waymo/eval_full_%j.out
#SBATCH --error=dcgm/eval_pp_1vfe_waymo/eval_full_%j.err
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
mkdir -p ../dcgm/eval_pp_1vfe_waymo

# Evaluate PointPillar Full 1VFE on Waymo
echo "Evaluating PointPillar Full 1VFE on Waymo..."
echo "Model: pointpillar_1x with 1 VFE layer"
echo "Checkpoint: checkpoint_epoch_60.pth"
echo ""

python test.py \
    --cfg_file cfgs/waymo_models/pointpillar_1x.yaml \
    --batch_size 1 \
    --ckpt ../output/waymo_models/pointpillar_1x/waymo_full_1vfe_p20/ckpt/checkpoint_epoch_60.pth \
    --infer_time \
    2>&1 | tee ../dcgm/eval_pp_1vfe_waymo/eval_full.log

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: dcgm/eval_pp_1vfe_waymo/eval_full.log"
echo "=========================================="
