#!/usr/bin/env python3
"""
Compare ML results between different frame stride settings

Shows improvement from stride=100 (71 frames) to stride=50 (141 frames)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def main():
    print("="*60)
    print("  Comparing Stride Results")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent

    # Load both results
    results_stride100 = pd.read_csv(
        project_root / 'md_data' / 'analysis' / 'ml_results_stride100' / 'classification_results.csv'
    )
    results_stride50 = pd.read_csv(
        project_root / 'md_data' / 'analysis' / 'ml_results' / 'classification_results.csv'
    )

    # Add stride column
    results_stride100['stride'] = 100
    results_stride100['frames'] = 71
    results_stride50['stride'] = 50
    results_stride50['frames'] = 141

    # Combine
    combined = pd.concat([results_stride100, results_stride50])

    output_dir = project_root / 'md_data' / 'analysis' / 'ml_results' / 'visualizations'

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Focus on graph-based models only
    graph_only = combined[~combined['model'].str.contains('Sequence')]

    models = graph_only['model'].unique()

    # 1. AUC comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35

    stride100_auc = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 100)]['auc'].values[0]
                     for m in models]
    stride50_auc = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 50)]['auc'].values[0]
                    for m in models]

    bars1 = ax.bar(x - width/2, stride100_auc, width, label='Stride=100 (71 frames)',
                   alpha=0.8, color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, stride50_auc, width, label='Stride=50 (141 frames)',
                   alpha=0.8, color='lightgreen', edgecolor='black')

    # Annotate changes
    for i, (v1, v2) in enumerate(zip(stride100_auc, stride50_auc)):
        change = ((v2 - v1) / v1 * 100) if v1 > 0 else 0
        if abs(change) > 5:
            ax.text(i, max(v1, v2) + 0.05, f'+{change:.0f}%' if change > 0 else f'{change:.0f}%',
                   ha='center', fontsize=9, fontweight='bold',
                   color='green' if change > 0 else 'red')

    ax.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax.set_title('AUC-ROC Improvement with 2x More Data', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(axis='y', alpha=0.3)

    # 2. Accuracy comparison
    ax = axes[0, 1]
    stride100_acc = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 100)]['accuracy'].values[0]
                     for m in models]
    stride50_acc = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 50)]['accuracy'].values[0]
                    for m in models]

    bars1 = ax.bar(x - width/2, stride100_acc, width, label='Stride=100 (71 frames)',
                   alpha=0.8, color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, stride50_acc, width, label='Stride=50 (141 frames)',
                   alpha=0.8, color='lightgreen', edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # 3. F1 Score comparison
    ax = axes[1, 0]
    stride100_f1 = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 100)]['f1'].values[0]
                    for m in models]
    stride50_f1 = [graph_only[(graph_only['model'] == m) & (graph_only['stride'] == 50)]['f1'].values[0]
                   for m in models]

    bars1 = ax.bar(x - width/2, stride100_f1, width, label='Stride=100 (71 frames)',
                   alpha=0.8, color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, stride50_f1, width, label='Stride=50 (141 frames)',
                   alpha=0.8, color='lightgreen', edgecolor='black')

    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate improvements
    improvements = []
    for model in models:
        s100 = graph_only[(graph_only['model'] == model) & (graph_only['stride'] == 100)].iloc[0]
        s50 = graph_only[(graph_only['model'] == model) & (graph_only['stride'] == 50)].iloc[0]

        auc_change = ((s50['auc'] - s100['auc']) / s100['auc'] * 100) if s100['auc'] > 0 else 0
        acc_change = ((s50['accuracy'] - s100['accuracy']) / s100['accuracy'] * 100) if s100['accuracy'] > 0 else 0

        improvements.append({
            'Model': model,
            'AUC Δ': f"{auc_change:+.1f}%",
            'Acc Δ': f"{acc_change:+.1f}%"
        })

    # Create summary text
    summary_text = "Performance Improvements\n"
    summary_text += "(Stride 100 → Stride 50)\n"
    summary_text += "=" * 40 + "\n\n"

    for imp in improvements:
        summary_text += f"{imp['Model']}:\n"
        summary_text += f"  AUC: {imp['AUC Δ']}\n"
        summary_text += f"  Accuracy: {imp['Acc Δ']}\n\n"

    summary_text += "\n" + "=" * 40 + "\n"
    summary_text += f"Dataset Size:\n"
    summary_text += f"  Stride=100: 71 frames/peptide\n"
    summary_text += f"  Stride=50: 141 frames/peptide\n"
    summary_text += f"  Increase: 2x frames\n"

    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Impact of Frame Stride on Model Performance\n(Graph-Based Features)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'stride_comparison.png', bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'stride_comparison.png'}")
    plt.close()

    # Create detailed comparison table
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for model in models:
        s100 = graph_only[(graph_only['model'] == model) & (graph_only['stride'] == 100)].iloc[0]
        s50 = graph_only[(graph_only['model'] == model) & (graph_only['stride'] == 50)].iloc[0]

        table_data.append([
            model,
            f"{s100['auc']:.3f}",
            f"{s50['auc']:.3f}",
            f"{((s50['auc'] - s100['auc']) / s100['auc'] * 100):+.1f}%" if s100['auc'] > 0 else "N/A",
            f"{s100['accuracy']:.3f}",
            f"{s50['accuracy']:.3f}",
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'AUC\n(Stride 100)', 'AUC\n(Stride 50)', 'AUC\nChange',
                              'Acc\n(Stride 100)', 'Acc\n(Stride 50)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.13, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight improvements
    for i in range(1, len(table_data) + 1):
        # Highlight change column
        change_text = table_data[i-1][3]
        if change_text != "N/A" and '+' in change_text:
            table[(i, 3)].set_facecolor('#d4edda')
            table[(i, 3)].set_text_props(weight='bold', color='darkgreen')

        for j in range(6):
            if j != 3:
                table[(i, j)].set_facecolor('#f8f9fa')

    plt.title('Detailed Stride Comparison (Graph-Based Models)', fontsize=13, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'stride_comparison_table.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stride_comparison_table.png'}")
    plt.close()

    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
