#!/bin/bash
#SBATCH --job-name=prep_waymo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/prep_waymo_%j.out
#SBATCH --error=logs/prep_waymo_%j.err
#SBATCH --account=pi-shokera

# Load modules
module purge
module load cuda/11.8.0

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

# Change to OpenPCDet directory
cd ~/Code/openpcdet_project/OpenPCDet

echo "Setting up Waymo dataset symlinks..."

# Remove old waymo directory if it exists
if [ -d "data/waymo/raw_data" ]; then
    rm -rf data/waymo/raw_data
fi

# Create raw_data symlink to individual_files
ln -sf /ibex/project/c2337/datasets/waymo/individual_files data/waymo/raw_data

echo "Symlink created: data/waymo/raw_data -> /ibex/project/c2337/datasets/waymo/individual_files"

# Change to tools directory
cd tools

echo "Starting Waymo dataset processing (this may take several hours)..."
echo "Processing with single frame only..."

python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml

echo "Waymo dataset preparation completed!"
