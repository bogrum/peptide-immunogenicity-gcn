#!/usr/bin/env python3
"""
Simple SwiftMHC runner that loads CUDA stub and forces CPU mode
"""

import sys
import os

# CRITICAL: Force CPU mode (RTX 5090 sm_120 not supported)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add paths
sys.path.insert(0, '/home/emre/workspace/seminar_study/analysis')
sys.path.insert(0, '/home/emre/workspace/seminar_study/analysis/openfold')

# Load CUDA stub before any imports
import openfold_cuda_stub

# Now import and run swiftmhc_predict
if __name__ == '__main__':
    # Import the main function
    sys.path.insert(0, '/home/emre/workspace/seminar_study/analysis/swiftmhc-inference-main')

    # Set up arguments
    sys.argv = [
        'swiftmhc_predict',
        '--num-builders', '0',
        '--batch-size', '8',
        'trained-models/8k-trained-model.pth',
        'our_peptides.csv',
        'data/HLA-A0201-from-3MRD.hdf5',
        'results_our_peptides'
    ]

    # Change to swiftmhc directory
    os.chdir('/home/emre/workspace/seminar_study/analysis/swiftmhc-inference-main')

    # Import and run main
    from scripts import swiftmhc_predict as predict_module
    # The module will execute when imported since it has if __name__ == '__main__'
