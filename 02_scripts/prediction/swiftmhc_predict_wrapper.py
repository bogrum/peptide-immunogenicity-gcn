#!/usr/bin/env python3
"""
Wrapper for swiftmhc_predict that loads CUDA stubs before OpenFold imports.
"""

import sys
import os

# CRITICAL: Force CPU mode because RTX 5090 (sm_120) is not supported by PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add OpenFold to path
sys.path.insert(0, '/home/emre/workspace/seminar_study/analysis/openfold')
sys.path.insert(0, '/home/emre/workspace/seminar_study/analysis')

# Load CUDA stub BEFORE any OpenFold imports
import openfold_cuda_stub

# Now run the actual swiftmhc_predict script
if __name__ == '__main__':
    # Get the actual script path
    script_path = '/home/emre/anaconda3/envs/swiftmhc_env/bin/swiftmhc_predict'

    # Read and execute the script
    with open(script_path) as f:
        script_content = f.read()

    # Execute in the current namespace
    exec(compile(script_content, script_path, 'exec'))
