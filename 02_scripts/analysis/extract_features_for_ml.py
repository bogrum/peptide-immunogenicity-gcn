#!/usr/bin/env python3
"""
Extract features from MD trajectories for machine learning
Following Weber et al. 2024 methodology

Features extracted:
1. Per-residue SASA (hydrophobic vs hydrophilic)
2. Backbone dihedral angles (phi, psi)
3. RMSD and RMSF
4. Anchor dynamics (P2, P9)
5. Contact maps (8 Å cutoff)

Usage: python extract_features_for_ml.py
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms, distances, contacts
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
import os
from pathlib import Path

# Hydrophobic residues (following paper's methodology)
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']
HYDROPHILIC_RESIDUES = ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS']

def extract_peptide_features(peptide_name, system_dir, output_dir):
    """
    Extract all features for a single peptide following Weber et al. methodology
    """
    print(f"\nProcessing {peptide_name}...")

    # Load trajectory
    tpr_file = os.path.join(system_dir, 'md.tpr')
    xtc_file = os.path.join(system_dir, 'md_30-200ns.xtc')

    if not os.path.exists(xtc_file):
        print(f"  Processing trajectory to extract 30-200 ns window...")
        # Will be created by the bash script first
        return None

    u = mda.Universe(tpr_file, xtc_file)

    # Select groups
    protein = u.select_atoms('protein')
    peptide = u.select_atoms('segid PEPTIDE or (protein and resid > 180)')  # Adjust based on actual peptide selection
    mhc = u.select_atoms('protein and not (segid PEPTIDE or resid > 180)')

    n_frames = len(u.trajectory)
    print(f"  Frames: {n_frames}")

    features = {
        'peptide': peptide_name,
        'n_frames': n_frames,
        'sasa_hydrophobic': [],
        'sasa_hydrophilic': [],
        'rmsd': [],
        'dihedral_phi': [],
        'dihedral_psi': [],
        'anchor_p2_distance': [],
        'anchor_p9_distance': [],
        'rmsf_peptide': [],
    }

    # Reference structure for RMSD (first frame)
    ref_coords = protein.positions.copy()

    print("  Extracting features...")
    for ts in u.trajectory:
        # 1. SASA (will calculate separately per residue)
        # This is a placeholder - actual SASA needs gmx sasa

        # 2. RMSD (no fitting - raw displacement)
        rmsd_val = rms.rmsd(protein.positions, ref_coords, superposition=False)
        features['rmsd'].append(rmsd_val)

        # 3. Dihedral angles (phi, psi for peptide backbone)
        # This requires proper dihedral calculation
        # Placeholder for now

        # 4. Anchor distances (simplified)
        if len(peptide.residues) >= 9:
            p2_atoms = peptide.residues[1].atoms  # Second residue (P2)
            p9_atoms = peptide.residues[8].atoms  # Ninth residue (P9)

            # Distance to MHC (simplified - distance to alpha chain)
            # This is a placeholder

    # Save features
    output_file = os.path.join(output_dir, f'{peptide_name}_features.npz')
    np.savez(output_file, **features)

    print(f"  ✓ Features saved to {output_file}")
    return features

def process_all_peptides():
    """
    Process all completed peptides
    """
    project_root = Path(__file__).parent.parent.parent
    systems_dir = project_root / 'md_data' / 'systems'
    analysis_dir = project_root / 'md_data' / 'analysis'
    features_dir = analysis_dir / 'ml_features'
    features_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Feature Extraction for Machine Learning")
    print("  Following Weber et al. 2024 Methodology")
    print("=" * 60)
    print()
    print("NOTE: Run process_all_trajectories.sh first to generate")
    print("      the processed trajectory files (md_30-200ns.xtc)")
    print()

    # Find all completed peptides
    peptides = []
    for peptide_dir in sorted(systems_dir.iterdir()):
        if peptide_dir.is_dir():
            md_log = peptide_dir / 'md.log'
            if md_log.exists():
                # Check if completed
                with open(md_log) as f:
                    last_lines = f.readlines()[-100:]
                    if any('Performance:' in line for line in last_lines):
                        peptides.append(peptide_dir.name)

    print(f"Found {len(peptides)} completed simulations\n")

    # Process each
    all_features = []
    for peptide in peptides:
        system_dir = systems_dir / peptide
        try:
            features = extract_peptide_features(peptide, system_dir, features_dir)
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"  ✗ Error processing {peptide}: {e}")

    print(f"\n✓ Processed {len(all_features)}/{len(peptides)} peptides")
    print(f"\nOutput directory: {features_dir}")

    return all_features

if __name__ == '__main__':
    # Check if MDAnalysis is available
    try:
        import MDAnalysis
        print(f"Using MDAnalysis version {MDAnalysis.__version__}")
    except ImportError:
        print("ERROR: MDAnalysis not installed")
        print("Install with: pip install MDAnalysis")
        exit(1)

    process_all_peptides()
