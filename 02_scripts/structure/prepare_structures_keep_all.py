#!/usr/bin/env python3
"""
Simplest approach: Just change residue names, keep all atoms
GROMACS will handle mismatches by rebuilding what it needs
"""

import pandas as pd
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBIO

# Paths
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures/simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_SEQ = "SLYNTIATL"

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

class ProteinOnlySelect(PDB.Select):
    """Keep only standard protein residues"""
    def accept_residue(self, residue):
        return residue.id[0] == ' '

def mutate_simple(residue, new_aa):
    """Just change the residue name, keep ALL atoms"""
    residue.resname = AA_1TO3[new_aa]

def prepare_structure(peptide_id, sequence, template_path, output_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('template', template_path)

    # Find peptide chain C
    peptide_chain = None
    for model in structure:
        for chain in model:
            if chain.id == 'C':
                peptide_chain = chain
                break

    if not peptide_chain:
        raise ValueError("Chain C not found")

    # Get first 9 residues
    peptide_residues = [r for r in peptide_chain.get_residues() if r.id[0] == ' '][:9]

    # Just change residue names
    mutations = 0
    for residue, target_aa in zip(peptide_residues, sequence):
        current_aa = AA_3TO1.get(residue.resname, '?')
        if current_aa != target_aa:
            mutate_simple(residue, target_aa)
            mutations += 1

    # Remove extra residues
    all_residues = list(peptide_chain.get_residues())
    for res in all_residues[9:]:
        peptide_chain.detach_child(res.id)

    # Save
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ProteinOnlySelect())

    return mutations

# Main
print("=" * 80)
print("SIMPLE STRUCTURE PREPARATION (Keep all atoms, change names only)")
print("=" * 80)

df = pd.read_csv(PEPTIDE_LIST)
print(f"\n📄 {len(df)} peptides\n")

success = 0
for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']

    output_path = OUTPUT_DIR / f"{peptide_id}.pdb"

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence}")

    try:
        mutations = prepare_structure(peptide_id, sequence, TEMPLATE_PDB, output_path)
        print(f"       ✅ {mutations} mutations\n")
        success += 1
    except Exception as e:
        print(f"       ❌ {e}\n")

print("=" * 80)
print(f"✅ {success}/{len(df)} structures ready")
print(f"\n📁 {OUTPUT_DIR}/*.pdb")
print("\n⚠️  NOTE: These have mismatched atoms. GROMACS pdb2gmx -missing")
print("   flag may be needed, or it might just work with warnings.")
