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

# Install required packages for Waymo dataset processing
echo "Installing TensorFlow and Waymo Open Dataset with compatible versions..."
pip install numpy==1.23.5
pip install protobuf==3.20.3
pip install tensorflow==2.13.0
pip install waymo-open-dataset-tf-2-12-0
pip install typing-extensions==4.12.2

# Change to OpenPCDet directory
cd ~/Code/openpcdet_project/OpenPCDet

echo "Setting up Waymo dataset symlinks..."

# Remove old raw_data if it exists
if [ -L "data/waymo/raw_data" ] || [ -d "data/waymo/raw_data" ]; then
    rm -rf data/waymo/raw_data
fi

# Remove any existing processed data symlinks/dirs
rm -rf data/waymo/waymo_processed_data_v0_5_0*

# Create raw_data directory
mkdir -p data/waymo/raw_data

# Create symlinks to training and validation tfrecords directly in raw_data
echo "Linking training tfrecords..."
ln -sf /ibex/project/c2337/datasets/waymo/individual_files/training/segment-*.tfrecord data/waymo/raw_data/

echo "Linking validation tfrecords..."
ln -sf /ibex/project/c2337/datasets/waymo/individual_files/validation/segment-*.tfrecord data/waymo/raw_data/

echo "Symlinks created in data/waymo/raw_data/"
ls data/waymo/raw_data/ | wc -l
echo "tfrecord files linked"

# Create processed data directory in ibex project storage (30TB available) and symlink it
echo "Setting up processed data directory in ibex project storage..."
mkdir -p /ibex/project/c2337/openpcdet_data/waymo_processed/waymo_processed_data_v0_5_0
ln -sf /ibex/project/c2337/openpcdet_data/waymo_processed/waymo_processed_data_v0_5_0 data/waymo/waymo_processed_data_v0_5_0

# Also symlink GT database directories to ibex storage
mkdir -p /ibex/project/c2337/openpcdet_data/waymo_processed/gt_database_train
mkdir -p /ibex/project/c2337/openpcdet_data/waymo_processed/gt_database_train_multiframe
ln -sf /ibex/project/c2337/openpcdet_data/waymo_processed/gt_database_train data/waymo/waymo_processed_data_v0_5_0_gt_database_train_sampled_1
ln -sf /ibex/project/c2337/openpcdet_data/waymo_processed/gt_database_train_multiframe data/waymo/waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0

echo "Processed data will be saved to: /ibex/project/c2337/openpcdet_data/waymo_processed/"

# Change to tools directory
cd tools

echo "Starting Waymo dataset processing (this may take several hours)..."
echo "Processing with single frame only..."

python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml

echo "Waymo dataset preparation completed!"
