#!/usr/bin/env python3
"""
Proper peptide-MHC structure preparation using MODELLER
This creates scientifically rigorous homology models for all 30 peptides
"""

import pandas as pd
from pathlib import Path
import sys

# Import MODELLER
try:
    from modeller import *
    from modeller.automodel import *
except ImportError:
    print("ERROR: MODELLER not found!")
    print("Please activate the conda environment: conda activate modeller_env")
    sys.exit(1)

# Configuration
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures/modeller")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Template info
TEMPLATE_ID = "5nmh"
TEMPLATE_CHAIN_MHC_HEAVY = "A"
TEMPLATE_CHAIN_B2M = "B"
TEMPLATE_CHAIN_PEPTIDE = "C"
TEMPLATE_PEPTIDE_SEQ = "SLYNTIATL"

print("=" * 80)
print("MODELLER STRUCTURE PREPARATION")
print("=" * 80)
print(f"\nTemplate: {TEMPLATE_PDB}")
print(f"Output: {OUTPUT_DIR}")
print()

# Load peptide list
df = pd.read_csv(PEPTIDE_LIST)
print(f"Loaded {len(df)} peptides to model\n")

def prepare_alignment_file(peptide_id, peptide_seq, template_seq, output_file):
    """
    Create PIR alignment file for MODELLER
    Format: PIR (Protein Information Resource)
    """
    with open(output_file, 'w') as f:
        # Template sequence
        f.write(f">P1;{TEMPLATE_ID}\n")
        f.write(f"structureX:{TEMPLATE_ID}:1:{TEMPLATE_CHAIN_PEPTIDE}:9:{TEMPLATE_CHAIN_PEPTIDE}::::\n")
        f.write(f"{template_seq}*\n\n")

        # Target sequence
        f.write(f">P1;{peptide_id}\n")
        f.write(f"sequence:{peptide_id}:1:C:9:C::::\n")
        f.write(f"{peptide_seq}*\n")

def model_peptide(peptide_id, peptide_seq, work_dir):
    """
    Use MODELLER to create homology model
    """
    # Create alignment file
    ali_file = work_dir / f"{peptide_id}.ali"
    prepare_alignment_file(peptide_id, peptide_seq, TEMPLATE_PEPTIDE_SEQ, ali_file)

    # Initialize MODELLER environment
    env = Environ()
    env.io.atom_files_directory = ['.', str(Path(TEMPLATE_PDB).parent)]

    # Create automodel class
    class MyModel(AutoModel):
        def select_atoms(self):
            # Only model the peptide chain (C), keep template for other chains
            return Selection(self.chains['C'])

    # Run modeling
    a = MyModel(
        env,
        alnfile=str(ali_file),
        knowns=TEMPLATE_ID,
        sequence=peptide_id,
        assess_methods=(assess.DOPE, assess.GA341)
    )

    # Set modeling parameters
    a.starting_model = 1
    a.ending_model = 1  # Generate 1 model (increase for better sampling)
    a.md_level = refine.slow  # Slow refinement for better quality

    # Change to working directory
    import os
    original_dir = os.getcwd()
    os.chdir(work_dir)

    try:
        a.make()

        # Get best model
        best_model = f"{peptide_id}.B99990001.pdb"
        output_pdb = work_dir / f"{peptide_id}_modeller.pdb"

        if Path(best_model).exists():
            Path(best_model).rename(output_pdb)
            return output_pdb
        else:
            return None

    finally:
        os.chdir(original_dir)

# Process each peptide
success_count = 0
failed = []

for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']
    label = row['label']

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence} ({label})")

    # Create work directory for this peptide
    work_dir = OUTPUT_DIR / peptide_id
    work_dir.mkdir(exist_ok=True)

    try:
        output_pdb = model_peptide(peptide_id, sequence, work_dir)

        if output_pdb and output_pdb.exists():
            print(f"       ✅ Model created: {output_pdb.name}")
            print(f"          Size: {output_pdb.stat().st_size / 1024:.1f} KB\n")
            success_count += 1
        else:
            print(f"       ❌ Model file not found\n")
            failed.append(peptide_id)

    except Exception as e:
        print(f"       ❌ ERROR: {e}\n")
        failed.append(peptide_id)

print("=" * 80)
print("MODELING COMPLETE")
print("=" * 80)
print(f"✅ Success: {success_count}/{len(df)} structures")

if failed:
    print(f"❌ Failed: {len(failed)} structures")
    for pid in failed:
        print(f"   - {pid}")
else:
    print("🎉 All structures modeled successfully!")

print(f"\n📁 Output files: {OUTPUT_DIR}/*/{{peptide_id}}_modeller.pdb")
print("\n🔜 Next: Clean and prepare for GROMACS")
