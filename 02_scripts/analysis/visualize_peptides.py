#!/usr/bin/env python3
"""
Visualization of selected peptides dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the dataset
df = pd.read_csv('peptides/selected/peptide_list_from_paper.csv')

# Extract anchor positions
df['P2'] = df['sequence'].str[1]
df['P9'] = df['sequence'].str[8]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# 1. Label Distribution
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
label_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.bar(label_counts.index, label_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Label Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xlabel('Label')
for i, v in enumerate(label_counts.values):
    ax1.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# ============================================================================
# 2. P2 Anchor Distribution
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
p2_counts = df['P2'].value_counts().sort_index()
p2_colors = ['#3498db' if aa in ['L', 'M', 'V', 'I'] else '#95a5a6' for aa in p2_counts.index]
ax2.bar(p2_counts.index, p2_counts.values, color=p2_colors, alpha=0.7, edgecolor='black')
ax2.set_title('P2 Anchor Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count')
ax2.set_xlabel('Amino Acid at P2')
ax2.axhline(y=0, color='black', linewidth=0.5)

# ============================================================================
# 3. P9 Anchor Distribution
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
p9_counts = df['P9'].value_counts().sort_index()
p9_colors = ['#3498db' if aa in ['L', 'V'] else '#95a5a6' for aa in p9_counts.index]
ax3.bar(p9_counts.index, p9_counts.values, color=p9_colors, alpha=0.7, edgecolor='black')
ax3.set_title('P9 Anchor Distribution', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count')
ax3.set_xlabel('Amino Acid at P9')
ax3.axhline(y=0, color='black', linewidth=0.5)

# ============================================================================
# 4. P2-P9 Anchor Combinations Heatmap
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
p2_p9_combo = df.groupby(['P2', 'P9']).size().unstack(fill_value=0)
sns.heatmap(p2_p9_combo, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_title('P2-P9 Anchor Combinations', fontsize=12, fontweight='bold')
ax4.set_xlabel('P9 Anchor')
ax4.set_ylabel('P2 Anchor')

# ============================================================================
# 5. Position-specific amino acid frequency (Immunogenic)
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
immuno_seqs = df[df['label'] == 'immunogenic']['sequence']
pos_freq_immuno = []
for pos in range(9):
    aa_counts = Counter(immuno_seqs.str[pos])
    pos_freq_immuno.append(aa_counts)

# Create heatmap data
all_aas = sorted(set(aa for pos_dict in pos_freq_immuno for aa in pos_dict.keys()))
heatmap_data_immuno = np.zeros((len(all_aas), 9))
for i, aa in enumerate(all_aas):
    for pos in range(9):
        heatmap_data_immuno[i, pos] = pos_freq_immuno[pos].get(aa, 0)

sns.heatmap(heatmap_data_immuno, yticklabels=all_aas, xticklabels=range(1, 10),
            cmap='Greens', ax=ax5, cbar_kws={'label': 'Count'})
ax5.set_title('Position-specific AA (Immunogenic)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Position')
ax5.set_ylabel('Amino Acid')

# ============================================================================
# 6. Position-specific amino acid frequency (Non-immunogenic)
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
non_immuno_seqs = df[df['label'] == 'non-immunogenic']['sequence']
pos_freq_non_immuno = []
for pos in range(9):
    aa_counts = Counter(non_immuno_seqs.str[pos])
    pos_freq_non_immuno.append(aa_counts)

# Create heatmap data
heatmap_data_non_immuno = np.zeros((len(all_aas), 9))
for i, aa in enumerate(all_aas):
    for pos in range(9):
        heatmap_data_non_immuno[i, pos] = pos_freq_non_immuno[pos].get(aa, 0)

sns.heatmap(heatmap_data_non_immuno, yticklabels=all_aas, xticklabels=range(1, 10),
            cmap='Reds', ax=ax6, cbar_kws={'label': 'Count'})
ax6.set_title('Position-specific AA (Non-immunogenic)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Position')
ax6.set_ylabel('Amino Acid')

# ============================================================================
# 7. Physicochemical Properties Comparison
# ============================================================================
ax7 = plt.subplot(3, 3, 7)

# Define amino acid properties
hydrophobic = set('AILMFVPGW')
polar = set('STNQCY')
charged_pos = set('KRH')
charged_neg = set('DE')
aromatic = set('FYW')

def calculate_properties(seq):
    return {
        'Hydrophobic': sum(1 for aa in seq if aa in hydrophobic),
        'Polar': sum(1 for aa in seq if aa in polar),
        'Pos. Charged': sum(1 for aa in seq if aa in charged_pos),
        'Neg. Charged': sum(1 for aa in seq if aa in charged_neg),
        'Aromatic': sum(1 for aa in seq if aa in aromatic),
    }

immuno_props = immuno_seqs.apply(calculate_properties).apply(pd.Series).mean()
non_immuno_props = non_immuno_seqs.apply(calculate_properties).apply(pd.Series).mean()

x = np.arange(len(immuno_props))
width = 0.35

bars1 = ax7.bar(x - width/2, immuno_props, width, label='Immunogenic', color='#2ecc71', alpha=0.7, edgecolor='black')
bars2 = ax7.bar(x + width/2, non_immuno_props, width, label='Non-immunogenic', color='#e74c3c', alpha=0.7, edgecolor='black')

ax7.set_title('Physicochemical Properties', fontsize=12, fontweight='bold')
ax7.set_ylabel('Average Count per Peptide')
ax7.set_xticks(x)
ax7.set_xticklabels(immuno_props.index, rotation=45, ha='right')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# ============================================================================
# 8. Amino Acid Composition (Overall)
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
all_aas_overall = ''.join(df['sequence'])
aa_counts_overall = Counter(all_aas_overall)
top_10_aas = dict(sorted(aa_counts_overall.items(), key=lambda x: x[1], reverse=True)[:10])

ax8.bar(top_10_aas.keys(), top_10_aas.values(), color='#9b59b6', alpha=0.7, edgecolor='black')
ax8.set_title('Top 10 Amino Acids (Overall)', fontsize=12, fontweight='bold')
ax8.set_ylabel('Count')
ax8.set_xlabel('Amino Acid')
ax8.grid(axis='y', alpha=0.3)

# ============================================================================
# 9. P2 and P9 by Label
# ============================================================================
ax9 = plt.subplot(3, 3, 9)

p2_by_label = df.groupby(['label', 'P2']).size().unstack(fill_value=0)
p2_by_label.T.plot(kind='bar', ax=ax9, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax9.set_title('P2 Anchors by Label', fontsize=12, fontweight='bold')
ax9.set_ylabel('Count')
ax9.set_xlabel('P2 Amino Acid')
ax9.legend(title='Label')
ax9.grid(axis='y', alpha=0.3)
plt.setp(ax9.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('peptides_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: peptides_analysis_visualization.png")

# ============================================================================
# Additional: Sequence Logo-like visualization
# ============================================================================
fig2, (ax_logo_immuno, ax_logo_non_immuno) = plt.subplots(2, 1, figsize=(14, 8))

# Immunogenic sequence patterns
immuno_pattern_data = []
for pos in range(9):
    aa_counts = Counter(immuno_seqs.str[pos])
    total = len(immuno_seqs)
    aa_freqs = {aa: count/total for aa, count in aa_counts.items()}
    immuno_pattern_data.append(aa_freqs)

# Plot immunogenic pattern
for pos in range(9):
    sorted_aas = sorted(immuno_pattern_data[pos].items(), key=lambda x: x[1], reverse=True)
    bottom = 0
    for aa, freq in sorted_aas:
        ax_logo_immuno.bar(pos + 1, freq, bottom=bottom, width=0.8,
                           label=aa if pos == 0 else "", alpha=0.7, edgecolor='black')
        if freq > 0.1:
            ax_logo_immuno.text(pos + 1, bottom + freq/2, aa, ha='center', va='center',
                                fontweight='bold', fontsize=10)
        bottom += freq

ax_logo_immuno.set_title('Immunogenic Peptides - Position-specific AA Frequency', fontsize=14, fontweight='bold')
ax_logo_immuno.set_xlabel('Position', fontsize=12)
ax_logo_immuno.set_ylabel('Frequency', fontsize=12)
ax_logo_immuno.set_xticks(range(1, 10))
ax_logo_immuno.set_ylim(0, 1)
ax_logo_immuno.grid(axis='y', alpha=0.3)
ax_logo_immuno.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='P2 Anchor')
ax_logo_immuno.axvline(x=8.5, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='P9 Anchor')

# Non-immunogenic sequence patterns
non_immuno_pattern_data = []
for pos in range(9):
    aa_counts = Counter(non_immuno_seqs.str[pos])
    total = len(non_immuno_seqs)
    aa_freqs = {aa: count/total for aa, count in aa_counts.items()}
    non_immuno_pattern_data.append(aa_freqs)

# Plot non-immunogenic pattern
for pos in range(9):
    sorted_aas = sorted(non_immuno_pattern_data[pos].items(), key=lambda x: x[1], reverse=True)
    bottom = 0
    for aa, freq in sorted_aas:
        ax_logo_non_immuno.bar(pos + 1, freq, bottom=bottom, width=0.8,
                                alpha=0.7, edgecolor='black')
        if freq > 0.1:
            ax_logo_non_immuno.text(pos + 1, bottom + freq/2, aa, ha='center', va='center',
                                     fontweight='bold', fontsize=10)
        bottom += freq

ax_logo_non_immuno.set_title('Non-immunogenic Peptides - Position-specific AA Frequency', fontsize=14, fontweight='bold')
ax_logo_non_immuno.set_xlabel('Position', fontsize=12)
ax_logo_non_immuno.set_ylabel('Frequency', fontsize=12)
ax_logo_non_immuno.set_xticks(range(1, 10))
ax_logo_non_immuno.set_ylim(0, 1)
ax_logo_non_immuno.grid(axis='y', alpha=0.3)
ax_logo_non_immuno.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='P2 Anchor')
ax_logo_non_immuno.axvline(x=8.5, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='P9 Anchor')

plt.tight_layout()
plt.savefig('peptides_sequence_patterns.png', dpi=300, bbox_inches='tight')
print("Sequence pattern visualization saved to: peptides_sequence_patterns.png")
