#!/bin/bash
# Wrapper script to run MODELLER with conda environment

# Activate conda
eval "$(conda shell.bash hook)"
conda activate modeller_env

# Run the script
python model_peptides_simple.py

# Deactivate
conda deactivate
