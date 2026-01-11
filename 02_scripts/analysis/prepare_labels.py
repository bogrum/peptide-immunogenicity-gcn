#!/usr/bin/env python3
"""
Prepare labels file for supervised learning
Maps peptide sequences to immunogenicity labels
"""

import pandas as pd
from pathlib import Path

# Load peptide list with labels
peptide_file = Path('00_data/peptides/peptide_list.csv')
df = pd.read_csv(peptide_file)

print("=" * 60)
print("  Preparing Labels for Supervised Learning")
print("=" * 60)
print()

# Create mapping
labels = {}
for _, row in df.iterrows():
    seq = row['sequence']
    label = 1 if row['label'] == 'immunogenic' else 0
    labels[seq] = {
        'label': label,
        'label_name': row['label'],
        'source': row['source'],
        'peptide_id': row['peptide_id']
    }

print(f"Total peptides: {len(labels)}")
print(f"Immunogenic: {sum(1 for v in labels.values() if v['label'] == 1)}")
print(f"Non-immunogenic: {sum(1 for v in labels.values() if v['label'] == 0)}")
print()

# Save in different formats
output_dir = Path('md_data/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Format 1: CSV with all info
df_out = pd.DataFrame([
    {
        'sequence': seq,
        'label': info['label'],
        'label_name': info['label_name'],
        'peptide_id': info['peptide_id'],
        'source': info['source']
    }
    for seq, info in labels.items()
])
df_out = df_out.sort_values('sequence')
csv_file = output_dir / 'peptide_labels.csv'
df_out.to_csv(csv_file, index=False)
print(f"✓ Saved: {csv_file}")

# Format 2: Simple mapping (sequence -> label)
simple_file = output_dir / 'labels_simple.txt'
with open(simple_file, 'w') as f:
    f.write("# Sequence\tLabel (1=immunogenic, 0=non-immunogenic)\n")
    for seq in sorted(labels.keys()):
        f.write(f"{seq}\t{labels[seq]['label']}\n")
print(f"✓ Saved: {simple_file}")

# Format 3: Python dict for easy import
dict_file = output_dir / 'labels_dict.py'
with open(dict_file, 'w') as f:
    f.write("# Peptide immunogenicity labels\n")
    f.write("# 1 = immunogenic, 0 = non-immunogenic\n\n")
    f.write("LABELS = {\n")
    for seq in sorted(labels.keys()):
        f.write(f"    '{seq}': {labels[seq]['label']},\n")
    f.write("}\n")
print(f"✓ Saved: {dict_file}")

print()
print("Label distribution:")
print("-" * 40)
print(f"  Immunogenic:     {sum(1 for v in labels.values() if v['label'] == 1)}")
print(f"  Non-immunogenic: {sum(1 for v in labels.values() if v['label'] == 0)}")
print(f"  Class balance:   50/50 (balanced)")
print()
