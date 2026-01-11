#!/usr/bin/env python3
"""
Clean PDB files - keep only protein chains (A, B, C), remove water and ligands
"""

from Bio import PDB
from pathlib import Path

class ProteinSelect(PDB.Select):
    """Select only protein chains A, B, C"""
    def accept_residue(self, residue):
        # Keep only standard amino acids in chains A, B, C
        return residue.get_parent().id in ['A', 'B', 'C'] and \
               residue.id[0] == ' '  # Standard residue (not HETATM)

# Input/output directories
INPUT_DIR = Path("peptides/structures")
OUTPUT_DIR = Path("peptides/structures/cleaned")
OUTPUT_DIR.mkdir(exist_ok=True)

# Get all initial PDB files
pdb_files = list(INPUT_DIR.glob("*_initial.pdb"))

print(f"Found {len(pdb_files)} PDB files to clean")
print()

parser = PDB.PDBParser(QUIET=True)
io = PDB.PDBIO()

for pdb_file in sorted(pdb_files):
    peptide_id = pdb_file.stem.replace('_initial', '')
    output_file = OUTPUT_DIR / f"{peptide_id}_cleaned.pdb"

    print(f"Cleaning {pdb_file.name}...")

    # Load structure
    structure = parser.get_structure('complex', pdb_file)

    # Save only protein chains
    io.set_structure(structure)
    io.save(str(output_file), ProteinSelect())

    print(f"  ✅ Saved to {output_file.name}")

print(f"\n✅ All {len(pdb_files)} structures cleaned!")
print(f"📁 Output directory: {OUTPUT_DIR}")
