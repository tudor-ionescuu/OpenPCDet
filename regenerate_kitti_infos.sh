#!/bin/bash
#SBATCH --job-name=regen_kitti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=logs/regenerate_kitti_%j.out
#SBATCH --error=logs/regenerate_kitti_%j.err
#SBATCH --account=pi-shokera

# Activate environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Go to tools directory
cd ~/Code/openpcdet_project/OpenPCDet/tools

# Regenerate KITTI info files with current numpy version
echo "Regenerating KITTI info files..."
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos cfgs/dataset_configs/kitti_dataset.yaml

echo "Done! Pickle files regenerated."
