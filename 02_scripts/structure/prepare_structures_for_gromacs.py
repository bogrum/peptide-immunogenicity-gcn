#!/usr/bin/env python3
"""
Improved structure preparation - keeps CB atoms for GROMACS compatibility
GROMACS pdb2gmx needs at least N, CA, C, O, CB to rebuild side chains
"""

import pandas as pd
from pathlib import Path

try:
    from Bio import PDB
    from Bio.PDB import PDBIO
except ImportError:
    print("Installing BioPython...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'biopython'])
    from Bio import PDB
    from Bio.PDB import PDBIO

# Paths
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures/for_gromacs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Template sequence
TEMPLATE_SEQ = "SLYNTIATL"

# Amino acid conversions
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

class ProteinOnlySelect(PDB.Select):
    """Select only standard protein residues (no water, ligands)"""
    def accept_residue(self, residue):
        return residue.id[0] == ' '  # Standard residue (not HETATM)

def mutate_residue_minimal(residue, new_aa_1letter):
    """
    Minimal mutation: change residue name and keep backbone + CB
    GROMACS pdb2gmx will rebuild the rest of the side chain
    """
    new_aa_3letter = AA_1TO3[new_aa_1letter]

    # Change residue name
    residue.resname = new_aa_3letter

    # For GLY, remove CB if present (GLY has no CB)
    if new_aa_3letter == 'GLY':
        if 'CB' in residue:
            residue.detach_child('CB')
        # Keep only backbone for GLY
        backbone_atoms = ['N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3']
        atoms_to_remove = [atom.id for atom in residue if atom.id not in backbone_atoms]
        for atom_id in atoms_to_remove:
            if atom_id in residue:
                residue.detach_child(atom_id)
    else:
        # For non-GLY, keep backbone + CB, remove other side chain atoms
        keep_atoms = ['N', 'CA', 'C', 'O', 'H', 'HA', 'CB', 'HB', 'HB1', 'HB2', 'HB3']
        atoms_to_remove = [atom.id for atom in residue if atom.id not in keep_atoms]
        for atom_id in atoms_to_remove:
            if atom_id in residue:
                residue.detach_child(atom_id)

    return residue

def prepare_structure(peptide_id, sequence, template_path, output_path):
    """Prepare structure with minimal mutations"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('template', template_path)

    # Find peptide chain C
    peptide_chain = None
    for model in structure:
        for chain in model:
            if chain.id == 'C':
                peptide_chain = chain
                break

    if peptide_chain is None:
        raise ValueError("Peptide chain 'C' not found")

    # Get first 9 residues (the actual peptide)
    peptide_residues = [res for res in peptide_chain.get_residues() if res.id[0] == ' '][:9]

    if len(peptide_residues) != 9:
        print(f"  ⚠️  Warning: Expected 9 residues, found {len(peptide_residues)}")

    # Apply mutations
    mutations_count = 0
    for residue, target_aa in zip(peptide_residues, sequence):
        current_aa = AA_3TO1.get(residue.resname, '?')
        if current_aa != target_aa:
            mutate_residue_minimal(residue, target_aa)
            mutations_count += 1

    # Remove extra residues in chain C (keep only first 9)
    all_residues = list(peptide_chain.get_residues())
    for res in all_residues[9:]:
        peptide_chain.detach_child(res.id)

    # Save with only protein chains
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ProteinOnlySelect())

    return mutations_count

# Main execution
print("=" * 80)
print("GROMACS-COMPATIBLE STRUCTURE PREPARATION")
print("=" * 80)

df = pd.read_csv(PEPTIDE_LIST)
print(f"\n📄 Loaded {len(df)} peptides")
print(f"📁 Output: {OUTPUT_DIR}")
print(f"🧬 Template: {TEMPLATE_PDB} (peptide: {TEMPLATE_SEQ})\n")

success = 0
failed = []

for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']
    label = row['label']

    output_path = OUTPUT_DIR / f"{peptide_id}.pdb"

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence} ({label})")

    try:
        mutations = prepare_structure(peptide_id, sequence, TEMPLATE_PDB, output_path)
        print(f"       ✅ {output_path.name} ({mutations} mutations)\n")
        success += 1
    except Exception as e:
        print(f"       ❌ FAILED: {e}\n")
        failed.append(peptide_id)

print("=" * 80)
print(f"✅ Success: {success}/{len(df)}")
if failed:
    print(f"❌ Failed: {', '.join(failed)}")
else:
    print("🎉 All structures ready for GROMACS!")

print(f"\n📁 Files: {OUTPUT_DIR}/*.pdb")
print("\n🔜 Next: Run MD simulation with:")
print(f"   ./run_md_single_peptide.sh immuno_01 ../../peptides/structures/for_gromacs/immuno_01.pdb")
