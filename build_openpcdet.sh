#!/bin/bash
#SBATCH --job-name=build_openpcdet
#SBATCH --output=logs/build_openpcdet_%j.out
#SBATCH --error=logs/build_openpcdet_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

# Activate virtual environment
source ../openpcdet-env/bin/activate

# Build and install OpenPCDet
echo "Building OpenPCDet with CUDA support..."
python setup.py develop

echo "Build completed!"
echo "Verifying installation..."
python -c "import pcdet; print('OpenPCDet version:', pcdet.__version__)"
