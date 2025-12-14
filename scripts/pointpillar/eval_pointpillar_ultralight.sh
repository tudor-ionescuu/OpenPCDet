#!/bin/bash
#SBATCH --job-name=eval_pp_ultra
#SBATCH --output=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/pointpillar/eval_ultralight_%j.out
#SBATCH --error=/home/ionesctn/Code/openpcdet_project/OpenPCDet/logs/pointpillar/eval_ultralight_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet/tools

python test.py \
    --cfg_file cfgs/kitti_models/pointpillar_ultralight.yaml \
    --ckpt ../pretrained_models/kitti/pointpillar_ultralight.pth \
    --batch_size 1 \
    --workers 8 \
    --infer_time

# jobs for reevaluating pointpillars on ibex because of time diff between ibex and tower
# [ionesctn@vsc509-03-l OpenPCDet]$ cd /home/ionesctn/Code/openpcdet_project/OpenPCDet && rm -f logs/pointpillar/eval_*.err logs/pointpillar/eval_*.out 2>/dev/null; sbatch scripts/pointpillar/eval_pointpillar_full.sh && sbatch scripts/pointpillar/eval_pointpillar_lite.sh && sbatch scripts/pointpillar/eval_pointpillar_ultralight.sh
# Submitted batch job 43266139
# Submitted batch job 43266140
# Submitted batch job 43266141