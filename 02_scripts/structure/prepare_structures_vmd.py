#!/usr/bin/env python3
"""
Automated structure preparation using VMD Mutator
Prepares all 30 peptide-MHC structures from template 5NMH.pdb
"""

import pandas as pd
import subprocess
import os
from pathlib import Path

# Paths
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load peptide list
df = pd.read_csv(PEPTIDE_LIST)
print(f"Loaded {len(df)} peptides from {PEPTIDE_LIST}")

# Template sequence (from 5NMH.pdb, chain P)
TEMPLATE_SEQ = "NLVPMVATV"

def get_mutations(template, target):
    """Get list of mutations from template to target sequence"""
    mutations = []
    for i, (t, p) in enumerate(zip(template, target)):
        if t != p:
            # VMD uses 1-based indexing
            mutations.append((i+1, t, p))
    return mutations

# Generate VMD script for each peptide
vmd_script_path = OUTPUT_DIR / "mutate_all.tcl"

with open(vmd_script_path, 'w') as f:
    f.write("# VMD script to mutate all 30 peptides\n")
    f.write("# Generated automatically\n\n")

    for idx, row in df.iterrows():
        peptide_id = row['peptide_id']
        sequence = row['sequence']

        mutations = get_mutations(TEMPLATE_SEQ, sequence)

        print(f"\n{peptide_id}: {sequence}")
        print(f"  Mutations: {len(mutations)}")

        output_pdb = OUTPUT_DIR / f"{peptide_id}_initial.pdb"

        f.write(f"\n# {peptide_id}: {sequence}\n")
        f.write(f"mol new {TEMPLATE_PDB}\n")
        f.write(f"set sel [atomselect top \"chain P\"]\n")

        # Apply mutations
        for pos, old_aa, new_aa in mutations:
            f.write(f"package require mutator\n")
            f.write(f"::Mutator::mutate {pos} {new_aa} P\n")

        # Save structure
        f.write(f"set all [atomselect top \"all\"]\n")
        f.write(f"$all writepdb {output_pdb}\n")
        f.write(f"mol delete top\n")

        print(f"  Will save to: {output_pdb}")

print(f"\n✅ VMD script created: {vmd_script_path}")
print(f"\nTo run VMD mutation:")
print(f"  vmd -dispdev text -e {vmd_script_path}")
print("\nOr run this script with VMD automation:")
print(f"  python3 {__file__} --run-vmd")

# Option to run VMD automatically
import sys
if '--run-vmd' in sys.argv:
    print("\n🚀 Running VMD to generate structures...")
    result = subprocess.run(
        ['vmd', '-dispdev', 'text', '-e', str(vmd_script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ VMD execution completed!")

        # Count generated files
        pdb_files = list(OUTPUT_DIR.glob("*_initial.pdb"))
        print(f"\n📁 Generated {len(pdb_files)} PDB files:")
        for pdb in sorted(pdb_files)[:5]:
            print(f"  - {pdb.name}")
        if len(pdb_files) > 5:
            print(f"  ... and {len(pdb_files) - 5} more")
    else:
        print("❌ VMD execution failed:")
        print(result.stderr)
