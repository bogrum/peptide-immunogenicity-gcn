#!/usr/bin/env python3
"""
Build the 2 missing structures using MODELLER
"""

from modeller import *
from modeller.automodel import *
import os

# MODELLER setup
env = Environ()
env.io.atom_files_directory = ['.', './']

# Template structure (5NMH from SwiftMHC HLA data)
TEMPLATE_PDB = "5NMH.pdb"
TEMPLATE_SEQ = "SLYNTIATL"  # Original peptide in 5NMH chain C

# Peptides to model
peptides = {
    'VIFCHPGQL': 'HLA-Ax02_01-VIFCHPGQL',
    'QAMEDLVRA': 'HLA-Ax02_01-QAMEDLVRA'
}

print("=" * 60)
print("Building Missing Peptide-MHC Structures with MODELLER")
print("=" * 60)
print()

# Check if template exists
if not os.path.exists(TEMPLATE_PDB):
    print(f"ERROR: Template {TEMPLATE_PDB} not found!")
    print("Downloading from PDB...")
    import urllib.request
    url = "https://files.rcsb.org/download/5NMH.pdb"
    urllib.request.urlretrieve(url, TEMPLATE_PDB)
    print(f"Downloaded {TEMPLATE_PDB}")
    print()

for peptide_seq, output_name in peptides.items():
    print(f"Building: {peptide_seq}")

    # Create alignment file
    with open('alignment.ali', 'w') as f:
        f.write(f""">P1;5NMH
structureX:5NMH:1:C:9:C:::-1.00:-1.00
{TEMPLATE_SEQ}*

>P1;{output_name}
sequence:{output_name}:1:C:9:C:::-1.00:-1.00
{peptide_seq}*
""")

    # Build model
    a = AutoModel(env,
                  alnfile='alignment.ali',
                  knowns='5NMH',
                  sequence=output_name)

    # Only model the peptide (chain C)
    a.starting_model = 1
    a.ending_model = 1

    a.make()

    # Rename output
    output_pdb = f"results_our_peptides/{output_name}.pdb"

    # Find the best model
    ok_models = [x for x in a.outputs if x['failure'] is None]
    if ok_models:
        best_model = ok_models[0]['name']
        os.rename(best_model, output_pdb)
        print(f"  ✅ Created: {output_pdb}")
    else:
        print(f"  ❌ Failed to build {peptide_seq}")

    print()

print("=" * 60)
print("Done! Check results_our_peptides/ for new structures")
print("=" * 60)
