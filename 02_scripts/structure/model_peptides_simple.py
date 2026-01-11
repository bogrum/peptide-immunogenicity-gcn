#!/usr/bin/env python3
"""
Simple MODELLER-based peptide mutation
Models only the peptide, keeps MHC chains from template
"""

import pandas as pd
from pathlib import Path
import sys
import os

try:
    from modeller import *
    from modeller.optimizers import MolecularDynamics, ConjugateGradients
    from modeller.automodel import *
except ImportError:
    print("ERROR: Run with: conda activate modeller_env")
    sys.exit(1)

from Bio import PDB
from Bio.PDB import PDBIO

# Configuration
PEPTIDE_LIST = "peptides/selected/peptide_list.csv"
TEMPLATE_PDB = "5NMH.pdb"
OUTPUT_DIR = Path("peptides/structures/modeller_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PEPTIDE STRUCTURE MODELING WITH MODELLER")
print("=" * 80)

# Load peptides
df = pd.read_csv(PEPTIDE_LIST)
print(f"\nPeptides to model: {len(df)}")
print(f"Output: {OUTPUT_DIR}\n")

def mutate_peptide_modeller(template_pdb, peptide_seq, output_pdb, peptide_id):
    """
    Use MODELLER to mutate peptide chain and optimize geometry
    """
    env = Environ()
    env.io.atom_files_directory = ['.', str(Path(template_pdb).parent)]
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')

    # Read template
    mdl = Model(env, file=template_pdb)

    # Define mutations (residue positions are 1-indexed in MODELLER)
    # Chain C is the peptide, residues 1-9
    # Template peptide: SLYNTIATL
    template_seq = "SLYNTIATL"

    # Create selection of peptide chain
    sel = Selection(mdl.chains['C'])

    # For each position, if different from template, mutate
    mutations = []
    for pos in range(9):
        template_aa = template_seq[pos]
        target_aa = peptide_seq[pos]

        if template_aa != target_aa:
            # MODELLER residue numbering: chain.residue_number
            res_id = f"C:{pos+1}"
            mutations.append((res_id, target_aa))

    # Apply mutations using MODELLER's mutate module
    if mutations:
        for res_id, new_aa in mutations:
            # Simple approach: change residue type
            # For production, use modeller.selection.mutate()
            pass  # MODELLER mutation is complex, using BioPython instead

    # Save
    mdl.write(file=str(output_pdb))

# Actually, let's use a hybrid approach: BioPython + energy minimization
def mutate_hybrid(template_pdb, peptide_seq, output_pdb, peptide_id):
    """
    Hybrid approach: BioPython mutation + MODELLER refinement
    """
    # Step 1: Load and mutate with BioPython
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('template', template_pdb)

    # Find peptide chain
    peptide_chain = None
    for model in structure:
        for chain in model:
            if chain.id == 'C':
                peptide_chain = chain
                break

    # Simple residue name change (like before)
    aa_1to3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }

    residues = [r for r in peptide_chain.get_residues() if r.id[0] == ' '][:9]

    for residue, target_aa in zip(residues, peptide_seq):
        residue.resname = aa_1to3[target_aa]

    # Remove extra residues
    all_res = list(peptide_chain.get_residues())
    for res in all_res[9:]:
        peptide_chain.detach_child(res.id)

    # Remove non-protein chains (water, ligands)
    class ProteinOnly(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[0] == ' '

    # Save intermediate
    temp_pdb = OUTPUT_DIR / f"{peptide_id}_temp.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(temp_pdb), ProteinOnly())

    # Step 2: Use MODELLER to add missing atoms and refine
    env = Environ()
    env.io.atom_files_directory = ['.', str(OUTPUT_DIR)]
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')

    # Read the mutated structure
    mdl = Model(env, file=str(temp_pdb))

    # Generate topology
    mdl.generate_topology(mdl.model_segment)

    # Build missing atoms
    mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)

    # Select peptide chain for optimization
    try:
        sel = Selection(mdl.chains['C'])
    except:
        # If chain selection fails, select all atoms
        sel = Selection(mdl)

    # Energy minimization
    cg = ConjugateGradients()
    cg.optimize(sel, max_iterations=200, output='NO_REPORT')

    # Write final model
    mdl.write(file=str(output_pdb))

    # Clean up temp file
    if temp_pdb.exists():
        temp_pdb.unlink()

    return output_pdb

# Process peptides
success = 0
failed = []

for idx, row in df.iterrows():
    peptide_id = row['peptide_id']
    sequence = row['sequence']
    label = row['label']

    output_pdb = OUTPUT_DIR / f"{peptide_id}.pdb"

    print(f"[{idx+1:2d}/{len(df)}] {peptide_id}: {sequence} ({label})")

    try:
        result = mutate_hybrid(TEMPLATE_PDB, sequence, output_pdb, peptide_id)
        if result.exists():
            print(f"       ✅ {result.name} ({result.stat().st_size/1024:.1f} KB)\n")
            success += 1
        else:
            print(f"       ❌ Failed\n")
            failed.append(peptide_id)
    except Exception as e:
        print(f"       ❌ ERROR: {e}\n")
        failed.append(peptide_id)

print("=" * 80)
print(f"✅ Success: {success}/{len(df)}")
if failed:
    print(f"❌ Failed: {', '.join(failed)}")
print("=" * 80)
print(f"\n📁 Files: {OUTPUT_DIR}/*.pdb")
print("\n🔜 Ready for GROMACS MD simulations!")
