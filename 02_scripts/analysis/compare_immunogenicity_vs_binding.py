#!/usr/bin/env python3
"""
Compare immunogenicity labels from paper with SwiftMHC binding predictions
"""
import pandas as pd

# Read immunogenicity labels from paper
paper_labels = pd.read_csv('peptides/selected/peptide_list_from_paper.csv')
paper_labels = paper_labels[['sequence', 'label']].rename(columns={'sequence': 'peptide'})

# Read SwiftMHC predictions
swiftmhc = pd.read_csv('swiftmhc-inference-main/results_our_peptides/results.csv')
# Remove duplicates (we ran SwiftMHC multiple times)
swiftmhc = swiftmhc.drop_duplicates(subset=['peptide'])

# Merge
comparison = paper_labels.merge(swiftmhc[['peptide', 'affinity', 'class']], on='peptide')
comparison = comparison.rename(columns={'class': 'binder_class'})
comparison['binder_label'] = comparison['binder_class'].map({0: 'non-binder', 1: 'binder'})

# Sort by immunogenicity label
comparison = comparison.sort_values(['label', 'affinity'], ascending=[False, False])

print("=" * 80)
print("IMMUNOGENICITY vs BINDING AFFINITY COMPARISON")
print("=" * 80)
print()
print(comparison.to_string(index=False))
print()

# Calculate statistics
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

# Cross-tabulation
crosstab = pd.crosstab(comparison['label'], comparison['binder_label'])
print("Cross-tabulation:")
print(crosstab)
print()

# Find mismatches
immunogenic_nonbinders = comparison[(comparison['label'] == 'immunogenic') &
                                     (comparison['binder_class'] == 0)]
non_immunogenic_binders = comparison[(comparison['label'] == 'non-immunogenic') &
                                      (comparison['binder_class'] == 1)]

print(f"Immunogenic peptides classified as NON-BINDERS: {len(immunogenic_nonbinders)}")
if len(immunogenic_nonbinders) > 0:
    print("  ⚠️  CRITICAL FINDING:")
    for _, row in immunogenic_nonbinders.iterrows():
        print(f"     {row['peptide']}: affinity={row['affinity']:.3f}")
print()

print(f"Non-immunogenic peptides classified as BINDERS: {len(non_immunogenic_binders)}")
if len(non_immunogenic_binders) > 0:
    print("  ⚠️  CRITICAL FINDING:")
    for _, row in non_immunogenic_binders.iterrows():
        print(f"     {row['peptide']}: affinity={row['affinity']:.3f}")
print()

# Average affinities
print("Average binding affinity by immunogenicity:")
print(comparison.groupby('label')['affinity'].agg(['mean', 'min', 'max']))
print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
Binding affinity measures MHC binding strength (necessary for presentation)
Immunogenicity measures actual T-cell immune response (the ultimate goal)

Key findings:
- If immunogenic peptides are predicted as non-binders → binding prediction alone
  is INSUFFICIENT for predicting immunogenicity
- If non-immunogenic peptides are predicted as binders → strong MHC binding does
  NOT guarantee immune response

This validates the need for structural analysis beyond binding affinity!
""")
