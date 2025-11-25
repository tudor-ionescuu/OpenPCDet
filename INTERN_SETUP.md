# PointPillar Adaptive Routing Setup Guide

This branch (`feature/adaptive-routing`) contains three PointPillar variants for KITTI dataset:
- **Full**: 19MB model with full accuracy
- **Lite**: 9.7MB model with reduced channels
- **Ultralight**: 2.0MB model with minimal channels

## Prerequisites

- Access to IBEX cluster with SLURM
- Python 3.9 environment
- CUDA-capable GPU (A100 recommended)

## Project Structure

After setup, your project directory should look like this:

```
openpcdet_project/
├── OpenPCDet/                          # Main repository
│   ├── data/
│   │   └── kitti/                      # Symlink to /ibex/project/c2337/openpcdet_data/kitti
│   ├── pretrained_models/
│   │   └── kitti/
│   │       ├── pointpillar_full.pth
│   │       ├── pointpillar_lite.pth
│   │       └── pointpillar_ultralight.pth
│   ├── tools/
│   │   └── cfgs/
│   │       └── kitti_models/
│   │           ├── pointpillar.yaml
│   │           ├── pointpillar_lite.yaml
│   │           └── pointpillar_ultralight.yaml
│   ├── logs/                           # SLURM job logs
│   ├── output/                         # Test results
│   ├── test_pointpillar_full_kitti.sh
│   ├── test_pointpillar_lite_kitti.sh
│   ├── test_pointpillar_ultralight_kitti.sh
│   ├── check_environment.sh
│   ├── requirements.txt
│   └── setup.py
└── openpcdet-env/                      # Python virtual environment
```

## Installation Steps

### 1. Clone the Repository
```bash
mkdir -p openpcdet_project
cd openpcdet_project
git clone -b feature/adaptive-routing https://github.com/tudor-ionescuu/OpenPCDet.git
cd OpenPCDet
```

### 2. Create Virtual Environment
```bash
cd ../
python3.9 -m venv openpcdet-env
source openpcdet-env/bin/activate
cd OpenPCDet
```

### 3. Install Dependencies
```bash
# Install all dependencies from requirements.txt
# This includes PyTorch 2.8.0, spconv-cu121, and all other packages
pip install -r requirements.txt
```

### 4. Build and Install OpenPCDet
The build process requires GPU access to compile CUDA extensions. Submit a build job:
```bash
# Create logs directory
mkdir -p logs

# Submit build job to GPU node
sbatch build_openpcdet.sh

# Monitor the job
squeue -u $USER

# Check build output (once job completes)
tail logs/build_openpcdet_<JOB_ID>.out
```

**Note**: The build script automatically loads CUDA 12.1 module on the GPU node and compiles 7 CUDA extensions needed for 3D object detection.

### 5. Verify Installation
Once the build job completes successfully, verify the installation:
```bash
# Check that pcdet can be imported
python -c "import pcdet; print('OpenPCDet installed successfully!')"
```

**Important Notes**:
- The environment uses **Python 3.9.18**
- **PyTorch 2.8.0** with **CUDA 12.8**
- **numpy 1.23.5** - CRITICAL: Do not upgrade numpy, as the KITTI pickle files are generated with this version
- **spconv-cu121 2.3.8** for sparse 3D convolutions
- **numba 0.60.0** with llvmlite 0.43.0 for CUDA kernels
- **Building requires GPU access** - CUDA extensions must be compiled on a GPU node
- All versions in requirements.txt are verified working on IBEX A100 nodes

### 6. Download Pretrained Models
Download the pretrained models from Google Drive and organize them:
```bash
# Create directory for pretrained models
mkdir -p pretrained_models/kitti

# Download models from:
# https://drive.google.com/drive/folders/1iq-gO2qJHB7DKV5et6Wy0GNjzOosdka2?usp=sharing
# 
# Place the following files in pretrained_models/kitti/:
# - pointpillar_full.pth (19MB)
# - pointpillar_lite.pth (9.7MB)
# - pointpillar_ultralight.pth (2.0MB)
```

### 7. Set Up KITTI Data
The KITTI data is stored in shared IBEX storage. Create a symlink:
```bash
cd data
ln -s /ibex/project/c2337/openpcdet_data/kitti kitti
cd ..
```

Verify the symlink:
```bash
ls -la data/kitti
```

You should see the following structure:
```
kitti/
├── ImageSets/
├── training/
├── testing/
├── gt_database/
├── kitti_dbinfos_train.pkl
├── kitti_infos_train.pkl
├── kitti_infos_val.pkl
├── kitti_infos_test.pkl
└── kitti_infos_trainval.pkl
```

## Testing the Models

All three PointPillar variants are ready to test with pretrained models.

### Test PointPillar Full
```bash
sbatch test_pointpillar_full_kitti.sh
```

### Test PointPillar Lite
```bash
sbatch test_pointpillar_lite_kitti.sh
```

### Test PointPillar Ultralight
```bash
sbatch test_pointpillar_ultralight_kitti.sh
```

### Monitor Job Status
```bash
squeue -u $USER
```

### View Test Results
```bash
# Check output logs
tail -f logs/test_pointpillar_full_kitti_<JOB_ID>.out

# View results directory
ls -l output/kitti_models/pointpillar/default/eval/
```

## Pretrained Models

All pretrained models are located in `pretrained_models/kitti/`:
- `pointpillar_full.pth` - Full model (19MB)
- `pointpillar_lite.pth` - Lite model (9.7MB)  
- `pointpillar_ultralight.pth` - Ultralight model (2.0MB)

## Model Configurations

Model configs are in `tools/cfgs/kitti_models/`:
- `pointpillar.yaml` - Full model config
- `pointpillar_lite.yaml` - Lite model config
- `pointpillar_ultralight.yaml` - Ultralight model config

## Expected Performance

KITTI validation set results (Car 3D AP at IoU=0.7):

| Model | Size | Easy | Moderate | Hard | Inference Time |
|-------|------|------|----------|------|----------------|
| Full | 19MB | 87.09% | 77.49% | 74.55% | ~37ms |
| Lite | 9.7MB | 77.92% | 68.80% | 66.46% | ~28ms |
| Ultralight | 2.0MB | 68.75% | 58.66% | 57.46% | ~27ms |


## Troubleshooting

### Issue: CUDA library not found
```bash
export LD_LIBRARY_PATH=../openpcdet-env/lib/python3.9/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=0
```

### Issue: Data not found
Verify the symlink exists:
```bash
readlink -f data/kitti
```
Should output: `/ibex/project/c2337/openpcdet_data/kitti`

### Issue: spconv version mismatch
Make sure you installed spconv-cu118:
```bash
pip list | grep spconv
```

