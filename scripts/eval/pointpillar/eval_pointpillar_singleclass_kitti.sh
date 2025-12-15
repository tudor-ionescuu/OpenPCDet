#!/bin/bash
#SBATCH --job-name=test_pp_singleclass
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/pointpillar/test_pp_singleclass_%j.out
#SBATCH --error=logs/pointpillar/test_pp_singleclass_%j.err
#SBATCH --partition=batch

source /home/ionesctn/Code/openpcdet_project/openpcdet-env/bin/activate

cd /home/ionesctn/Code/openpcdet_project/OpenPCDet/tools

python test.py \
    --cfg_file cfgs/kitti_models/pointpillar_singleclass.yaml \
    --batch_size 1 \
    --ckpt ../output/kitti_models/pointpillar_singleclass/singleclass_object/ckpt/checkpoint_epoch_80.pth \
    --infer_time

echo "Test completed!"
