#!/usr/bin/env python3
"""
Comprehensive analysis of selected peptides dataset
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('peptides/selected/peptide_list_from_paper.csv')

print("=" * 80)
print("SELECTED PEPTIDES DATASET ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. BASIC DATASET STRUCTURE
# ============================================================================
print("\n" + "=" * 80)
print("1. BASIC DATASET STRUCTURE")
print("=" * 80)

print(f"\nTotal peptides: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\nLabel distribution:")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\nBalance ratio: {label_counts['immunogenic']}/{label_counts['non-immunogenic']}")

print("\nFirst 5 peptides:")
print(df.head())

# ============================================================================
# 2. SEQUENCE CHARACTERISTICS
# ============================================================================
print("\n" + "=" * 80)
print("2. SEQUENCE CHARACTERISTICS")
print("=" * 80)

# Sequence lengths
df['length'] = df['sequence'].apply(len)
print(f"\nSequence length statistics:")
print(df['length'].describe())
print(f"All peptides are 9-mers: {df['length'].nunique() == 1 and df['length'].iloc[0] == 9}")

# Amino acid composition analysis
def analyze_aa_composition(sequences):
    """Analyze amino acid composition across sequences"""
    all_aas = ''.join(sequences)
    aa_counts = Counter(all_aas)
    total = len(all_aas)
    aa_freqs = {aa: count/total * 100 for aa, count in aa_counts.items()}
    return aa_freqs

immuno_seqs = df[df['label'] == 'immunogenic']['sequence']
non_immuno_seqs = df[df['label'] == 'non-immunogenic']['sequence']

print("\nAmino acid composition (overall):")
overall_aa = analyze_aa_composition(df['sequence'])
for aa, freq in sorted(overall_aa.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {aa}: {freq:.2f}%")

# ============================================================================
# 3. HLA-A2 ANCHOR POSITIONS (P2 and P9)
# ============================================================================
print("\n" + "=" * 80)
print("3. HLA-A2 ANCHOR POSITIONS ANALYSIS")
print("=" * 80)

# Extract anchor positions (P2 = position 1, P9 = position 8 in 0-indexed)
df['P2'] = df['sequence'].str[1]
df['P9'] = df['sequence'].str[8]

print("\nP2 anchor residues distribution:")
p2_counts = df['P2'].value_counts()
print(p2_counts)

print("\nP9 anchor residues distribution:")
p9_counts = df['P9'].value_counts()
print(p9_counts)

# HLA-A2 preferred anchors (L, M, V, I at P2; L, V at P9)
p2_preferred = ['L', 'M', 'V', 'I']
p9_preferred = ['L', 'V']

df['P2_valid'] = df['P2'].isin(p2_preferred)
df['P9_valid'] = df['P9'].isin(p9_preferred)

print(f"\nP2 validation (L/M/V/I):")
print(f"  Valid: {df['P2_valid'].sum()} ({df['P2_valid'].sum()/len(df)*100:.1f}%)")

print(f"\nP9 validation (L/V):")
print(f"  Valid: {df['P9_valid'].sum()} ({df['P9_valid'].sum()/len(df)*100:.1f}%)")

print("\nP2-P9 anchor combinations:")
anchor_combos = df.groupby(['P2', 'P9']).size().sort_values(ascending=False)
print(anchor_combos.head(15))

# ============================================================================
# 4. IMMUNOGENIC vs NON-IMMUNOGENIC COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("4. IMMUNOGENIC vs NON-IMMUNOGENIC COMPARISON")
print("=" * 80)

print("\nP2 anchors by label:")
p2_by_label = pd.crosstab(df['P2'], df['label'])
print(p2_by_label)

print("\nP9 anchors by label:")
p9_by_label = pd.crosstab(df['P9'], df['label'])
print(p9_by_label)

# Position-specific amino acid preferences
print("\nPosition-specific amino acid analysis (Immunogenic vs Non-immunogenic):")
for pos in range(9):
    print(f"\n  Position {pos+1}:")
    immuno_aas = Counter(immuno_seqs.str[pos])
    non_immuno_aas = Counter(non_immuno_seqs.str[pos])

    all_aas_at_pos = set(immuno_aas.keys()) | set(non_immuno_aas.keys())

    differences = []
    for aa in all_aas_at_pos:
        immuno_freq = immuno_aas.get(aa, 0) / len(immuno_seqs) * 100
        non_immuno_freq = non_immuno_aas.get(aa, 0) / len(non_immuno_seqs) * 100
        diff = immuno_freq - non_immuno_freq
        if abs(diff) > 10:  # Show only significant differences
            differences.append((aa, immuno_freq, non_immuno_freq, diff))

    if differences:
        for aa, imm_freq, non_imm_freq, diff in sorted(differences, key=lambda x: abs(x[3]), reverse=True)[:3]:
            print(f"    {aa}: Immuno={imm_freq:.1f}%, Non-immuno={non_imm_freq:.1f}%, Diff={diff:+.1f}%")

# ============================================================================
# 5. PHYSICOCHEMICAL PROPERTIES
# ============================================================================
print("\n" + "=" * 80)
print("5. PHYSICOCHEMICAL PROPERTIES")
print("=" * 80)

# Define amino acid properties
hydrophobic = set('AILMFVPGW')
polar = set('STNQCY')
charged_pos = set('KRH')
charged_neg = set('DE')
aromatic = set('FYW')

def calculate_properties(seq):
    """Calculate physicochemical properties of a sequence"""
    seq_set = set(seq)
    return {
        'hydrophobic_count': sum(1 for aa in seq if aa in hydrophobic),
        'polar_count': sum(1 for aa in seq if aa in polar),
        'charged_pos_count': sum(1 for aa in seq if aa in charged_pos),
        'charged_neg_count': sum(1 for aa in seq if aa in charged_neg),
        'aromatic_count': sum(1 for aa in seq if aa in aromatic),
    }

# Calculate properties for all peptides
properties_df = df['sequence'].apply(calculate_properties).apply(pd.Series)
df = pd.concat([df, properties_df], axis=1)

print("\nPhysicochemical properties by label:")
for prop in ['hydrophobic_count', 'polar_count', 'charged_pos_count', 'charged_neg_count', 'aromatic_count']:
    print(f"\n  {prop}:")
    print(df.groupby('label')[prop].agg(['mean', 'std', 'min', 'max']))

# ============================================================================
# 6. ANCHOR DIVERSITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. ANCHOR DIVERSITY ANALYSIS")
print("=" * 80)

print("\nUnique P2-P9 combinations:")
unique_combos = df.groupby(['P2', 'P9']).size()
print(f"Total unique combinations: {len(unique_combos)}")
print("\nCombinations used:")
print(unique_combos.sort_values(ascending=False))

print("\nCombination diversity by label:")
immuno_combos = df[df['label'] == 'immunogenic'].groupby(['P2', 'P9']).size()
non_immuno_combos = df[df['label'] == 'non-immunogenic'].groupby(['P2', 'P9']).size()
print(f"  Immunogenic: {len(immuno_combos)} unique combinations")
print(f"  Non-immunogenic: {len(non_immuno_combos)} unique combinations")

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("7. SUMMARY STATISTICS")
print("=" * 80)

summary = {
    'Total peptides': len(df),
    'Immunogenic': len(df[df['label'] == 'immunogenic']),
    'Non-immunogenic': len(df[df['label'] == 'non-immunogenic']),
    'All 9-mers': df['length'].nunique() == 1 and df['length'].iloc[0] == 9,
    'P2 anchor diversity': df['P2'].nunique(),
    'P9 anchor diversity': df['P9'].nunique(),
    'P2-P9 combinations': len(df.groupby(['P2', 'P9'])),
    'P2 validation rate': f"{df['P2_valid'].sum()/len(df)*100:.1f}%",
    'P9 validation rate': f"{df['P9_valid'].sum()/len(df)*100:.1f}%",
}

print("\nDataset summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
