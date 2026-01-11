#!/usr/bin/env python3
"""
SIMPLE PRAGMATIC APPROACH:
- Use BioPython to mutate residue names
- Save structures
- Let GROMACS pdb2gmx with -missing flag handle the rest
"""

import pandas as pd
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBIO

# Config
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures/final_simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_SEQ = "SLYNTIATL"

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

class ProteinOnly(PDB.Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '  # Standard residues only

print("=" * 80)
print("SIMPLE STRUCTURE PREPARATION")
print("Just mutate residue names, GROMACS will fix the rest")
print("=" * 80)

df = pd.read_csv(PEPTIDE_LIST)
print(f"\nPeptides: {len(df)}\n")

parser = PDB.PDBParser(QUIET=True)
io = PDBIO()

success = 0

for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence}")

    try:
        # Load template
        structure = parser.get_structure('template', TEMPLATE_PDB)

        # Find peptide chain C
        peptide_chain = None
        for model in structure:
            for chain in model:
                if chain.id == 'C':
                    peptide_chain = chain
                    break

        # Get first 9 residues
        residues = [r for r in peptide_chain.get_residues() if r.id[0] == ' '][:9]

        # Just change residue names
        for residue, target_aa in zip(residues, sequence):
            residue.resname = AA_1TO3[target_aa]

        # Remove extra residues in chain C
        all_res = list(peptide_chain.get_residues())
        for res in all_res[9:]:
            peptide_chain.detach_child(res.id)

        # Save
        output_pdb = OUTPUT_DIR / f"{peptide_id}.pdb"
        io.set_structure(structure)
        io.save(str(output_pdb), ProteinOnly())

        print(f"       ✅ {output_pdb.name}\n")
        success += 1

    except Exception as e:
        print(f"       ❌ {e}\n")

print("=" * 80)
print(f"✅ {success}/{len(df)} structures ready")
print("=" * 80)
print(f"\n📁 {OUTPUT_DIR}/*.pdb")
print("\n🔜 Use with GROMACS:")
print("   gmx pdb2gmx -f <file>.pdb -missing")
print("\nThe -missing flag will add all missing atoms automatically!")
