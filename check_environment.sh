#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH --output=logs/check_environment_%j.out
#SBATCH --error=logs/check_environment_%j.err
#SBATCH --account=pi-shokera

echo "========================================="
echo "ENVIRONMENT CHECK ON GPU NODE"
echo "========================================="
echo ""

echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Activate virtual environment
source ~/Code/openpcdet_project/openpcdet-env/bin/activate

echo "========================================="
echo "PYTHON ENVIRONMENT"
echo "========================================="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "========================================="
echo "CUDA INFORMATION"
echo "========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not found or error"
echo ""

echo "========================================="
echo "INSTALLED PACKAGES (pip list)"
echo "========================================="
pip list
echo ""

echo "========================================="
echo "KEY DEPENDENCIES"
echo "========================================="
echo "PyTorch and CUDA:"
pip list | grep -i torch || echo "torch not found"
echo ""
echo "spconv:"
pip list | grep -i spconv || echo "spconv not found"
echo ""
echo "numpy:"
pip list | grep -i numpy || echo "numpy not found"
echo ""
echo "numba:"
pip list | grep -i numba || echo "numba not found"
echo ""
echo "llvmlite:"
pip list | grep -i llvmlite || echo "llvmlite not found"
echo ""

echo "========================================="
echo "PCDET IMPORT TEST"
echo "========================================="
cd ~/Code/openpcdet_project/OpenPCDet
python -c "
import sys
try:
    import pcdet
    print(f'✓ pcdet imported successfully')
    print(f'  Location: {pcdet.__file__}')
except Exception as e:
    print(f'✗ Failed to import pcdet: {e}')
    
try:
    import torch
    print(f'✓ torch imported successfully')
    print(f'  Version: {torch.__version__}')
    print(f'  CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'✗ Failed to import torch: {e}')

try:
    import spconv.pytorch as spconv
    print(f'✓ spconv imported successfully')
    print(f'  Version: {spconv.__version__}')
except Exception as e:
    print(f'✗ Failed to import spconv: {e}')

try:
    import numba
    print(f'✓ numba imported successfully')
    print(f'  Version: {numba.__version__}')
except Exception as e:
    print(f'✗ Failed to import numba: {e}')
"
echo ""

echo "========================================="
echo "ENVIRONMENT VARIABLES"
echo "========================================="
echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"
echo ""
echo "CUDA_HOME:"
echo "$CUDA_HOME"
echo ""
echo "PYTHONPATH:"
echo "$PYTHONPATH"
echo ""

echo "========================================="
echo "CUDA LIBRARIES CHECK"
echo "========================================="
echo "Looking for libnvvm.so:"
find ~/Code/openpcdet_project/openpcdet-env -name "libnvvm.so" 2>/dev/null || echo "Not found in venv"
echo ""

echo "========================================="
echo "COMPLETE - Check output above for details"
echo "========================================="
