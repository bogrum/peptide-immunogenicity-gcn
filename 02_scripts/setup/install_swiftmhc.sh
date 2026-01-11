#!/bin/bash
# SwiftMHC Installation Script
set -e

echo "=========================================="
echo "SwiftMHC Installation"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate modeller_env

echo ""
echo "[1/5] Installing PyTorch with CUDA..."
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo ""
echo "[2/5] Installing PyMOL..."
conda install -c conda-forge pymol-open-source -y

echo ""
echo "[3/5] Cloning and installing OpenFold..."
cd /home/emre/workspace/seminar_study/analysis

# Clone OpenFold if not exists
if [ ! -d "openfold" ]; then
    git clone https://github.com/aqlaboratory/openfold.git
fi

cd openfold

# Install OpenFold dependencies
echo "Installing OpenFold third-party dependencies..."
bash scripts/install_third_party_dependencies.sh

# Install OpenFold
pip install -e .

echo ""
echo "[4/5] Installing SwiftMHC..."
cd /home/emre/workspace/seminar_study/analysis/swiftmhc-inference-main
pip install -e .

echo ""
echo "[5/5] Testing installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pymol; print('PyMOL: OK')"
python -c "import openfold; print('OpenFold: OK')"
swiftmhc_predict --help | head -5

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next step: Run prediction on your peptides"
echo "Command:"
echo "  cd swiftmhc-inference-main"
echo "  swiftmhc_predict --num-builders 1 --batch-size 8 \\"
echo "      trained-models/8k-trained-model.pth \\"
echo "      our_peptides.csv \\"
echo "      data/HLA-A0201-from-3MRD.hdf5 \\"
echo "      results_our_peptides"
