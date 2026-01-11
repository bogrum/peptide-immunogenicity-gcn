#!/usr/bin/env python3
"""
Simple structure preparation using BioPython
Prepares all 30 peptide-MHC structures by mutating template 5NMH.pdb
"""

import pandas as pd
import os
from pathlib import Path

# Check if BioPython is available
try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️  BioPython not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'biopython'])
    from Bio import PDB
    from Bio.PDB import PDBIO, Select

# Paths
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Template sequence (from 5NMH.pdb, chain C - actual peptide: SLYNTIATL)
# We'll use chain C (9-mer peptide) and mutate it to our target sequences
TEMPLATE_SEQ = "SLYNTIATL"  # Actual peptide in 5NMH

# Three-letter amino acid codes
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

def mutate_residue(residue, new_aa_1letter):
    """
    Mutate a residue to a new amino acid by changing its resname
    and removing side chain atoms (keep only backbone N, CA, C, O)
    """
    new_aa_3letter = AA_1TO3[new_aa_1letter]

    # Change residue name
    residue.resname = new_aa_3letter

    # Keep only backbone atoms (remove side chain)
    backbone_atoms = ['N', 'CA', 'C', 'O', 'H', 'HA']
    atoms_to_remove = []

    for atom in residue:
        if atom.name not in backbone_atoms:
            atoms_to_remove.append(atom.id)

    for atom_id in atoms_to_remove:
        residue.detach_child(atom_id)

    return residue

def prepare_structure(peptide_id, sequence, template_path, output_path):
    """
    Prepare a mutated structure for a given peptide sequence
    """
    # Load template structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('template', template_path)

    # Find peptide chain (chain C in 5NMH - the 9-mer peptide)
    peptide_chain = None
    for model in structure:
        for chain in model:
            if chain.id == 'C':
                peptide_chain = chain
                break

    if peptide_chain is None:
        raise ValueError("Peptide chain 'C' not found in template")

    # Get peptide residues (should be 9 residues)
    peptide_residues = list(peptide_chain.get_residues())

    if len(peptide_residues) != 9:
        print(f"  ⚠️  Warning: Expected 9 residues, found {len(peptide_residues)}")

    # Apply mutations
    mutations_applied = 0
    for i, (residue, target_aa) in enumerate(zip(peptide_residues, sequence)):
        current_aa = AA_3TO1.get(residue.resname, '?')

        if current_aa != target_aa:
            mutate_residue(residue, target_aa)
            mutations_applied += 1

    # Save mutated structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))

    return mutations_applied

# Main execution
print("=" * 80)
print("AUTOMATED STRUCTURE PREPARATION")
print("=" * 80)

# Load peptide list
df = pd.read_csv(PEPTIDE_LIST)
print(f"\n📄 Loaded {len(df)} peptides from {PEPTIDE_LIST}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"🧬 Template: {TEMPLATE_PDB} (sequence: {TEMPLATE_SEQ})\n")

# Check template exists
if not os.path.exists(TEMPLATE_PDB):
    print(f"❌ Template PDB not found: {TEMPLATE_PDB}")
    exit(1)

# Process each peptide
success_count = 0
failed = []

for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']
    label = row['label']

    output_path = OUTPUT_DIR / f"{peptide_id}_initial.pdb"

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence} ({label})")

    try:
        mutations = prepare_structure(peptide_id, sequence, TEMPLATE_PDB, output_path)
        print(f"       ✅ Created {output_path.name} ({mutations} mutations)\n")
        success_count += 1
    except Exception as e:
        print(f"       ❌ FAILED: {e}\n")
        failed.append(peptide_id)

print("=" * 80)
print(f"STRUCTURE PREPARATION COMPLETE")
print("=" * 80)
print(f"✅ Success: {success_count}/{len(df)} structures")

if failed:
    print(f"❌ Failed: {len(failed)} structures")
    for pid in failed:
        print(f"   - {pid}")
else:
    print("🎉 All structures prepared successfully!")

print(f"\n📁 Output files: {OUTPUT_DIR}/*_initial.pdb")
print("\n⚠️  NOTE: These structures contain only backbone atoms.")
print("   GROMACS pdb2gmx will add missing side chain atoms and hydrogens.")
print("\n🔜 Next step: Run GROMACS pdb2gmx on each structure")
print("   See: md_data/mdp_files/README_MDP.md for instructions")
