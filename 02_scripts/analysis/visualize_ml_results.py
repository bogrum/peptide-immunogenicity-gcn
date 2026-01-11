#!/usr/bin/env python3
"""
Visualize machine learning results for peptide immunogenicity prediction

Creates:
1. Model performance comparison
2. Feature importance analysis
3. Graph statistics distributions
4. Confusion matrices
5. Class distribution plots

Usage: python visualize_ml_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_data(graphs_dir, labels_file):
    """Load graph features and labels"""
    # Load labels
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['sequence'], labels_df['label']))

    features_list = []

    for pkl_file in sorted(Path(graphs_dir).glob('*_graphs.pkl')):
        peptide = pkl_file.stem.replace('_graphs', '')

        if peptide not in labels_dict:
            continue

        with open(pkl_file, 'rb') as f:
            graph_data = pickle.load(f)

        summary = graph_data['summary']

        features = {
            'peptide': peptide,
            'label': labels_dict[peptide],
            'label_name': 'Immunogenic' if labels_dict[peptide] == 1 else 'Non-immunogenic',
            'LL_edges_mean': summary['LL_edges_mean'],
            'LL_edges_std': summary['LL_edges_std'],
            'LL_edges_min': summary['LL_edges_min'],
            'LL_edges_max': summary['LL_edges_max'],
            'LP_edges_mean': summary['LP_edges_mean'],
            'LP_edges_std': summary['LP_edges_std'],
            'LP_edges_min': summary['LP_edges_min'],
            'LP_edges_max': summary['LP_edges_max'],
        }

        features_list.append(features)

    return pd.DataFrame(features_list)


def plot_model_performance(results_df, output_dir):
    """Plot model performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Separate graph-based and sequence-based models
    graph_models = results_df[~results_df['model'].str.contains('Sequence')]
    seq_models = results_df[results_df['model'].str.contains('Sequence')]

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        # Plot both model types
        x_pos = np.arange(len(graph_models))
        width = 0.35

        ax.bar(x_pos - width/2, graph_models[metric], width,
               label='Graph-based', alpha=0.8, color='steelblue')
        ax.bar(x_pos + width/2, seq_models[metric], width,
               label='Sequence-based', alpha=0.8, color='coral')

        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_models['model'].str.replace(' \(Sequence\)', ''),
                          rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', bbox_inches='tight')
    print(f"  ✓ Saved: model_performance_comparison.png")
    plt.close()


def plot_feature_distributions(df, output_dir):
    """Plot graph feature distributions by class"""
    features_to_plot = [
        'LL_edges_mean', 'LL_edges_std',
        'LP_edges_mean', 'LP_edges_std'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]

        # Separate by class
        immuno = df[df['label'] == 1][feature]
        non_immuno = df[df['label'] == 0][feature]

        # Box plot
        data = [immuno, non_immuno]
        bp = ax.boxplot(data, labels=['Immunogenic', 'Non-immunogenic'],
                       patch_artist=True, widths=0.6)

        # Color boxes
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points
        for i, d in enumerate(data):
            y = d
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=30, color=colors[i], edgecolors='black', linewidth=0.5)

        # Statistical test
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(immuno, non_immuno)

        title = feature.replace('_', ' ').title()
        ax.set_title(f'{title}\n(p={p_value:.4f})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Graph Feature Distributions by Immunogenicity',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', bbox_inches='tight')
    print(f"  ✓ Saved: feature_distributions.png")
    plt.close()


def plot_feature_correlations(df, output_dir):
    """Plot correlation matrix of graph features"""
    features = ['LL_edges_mean', 'LL_edges_std', 'LL_edges_min', 'LL_edges_max',
                'LP_edges_mean', 'LP_edges_std', 'LP_edges_min', 'LP_edges_max']

    # Compute correlation matrix
    corr_matrix = df[features].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    # Clean labels
    labels = [l.replace('_', ' ').title() for l in features]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', bbox_inches='tight')
    print(f"  ✓ Saved: feature_correlations.png")
    plt.close()


def plot_feature_importance(df, output_dir):
    """Plot feature importance from Random Forest"""
    # Prepare data
    feature_cols = ['LL_edges_mean', 'LL_edges_std', 'LL_edges_min', 'LL_edges_max',
                   'LP_edges_mean', 'LP_edges_std', 'LP_edges_min', 'LP_edges_max']

    X = df[feature_cols].values
    y = df['label'].values

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue' if 'LL' in feature_cols[i] else 'coral' for i in indices]

    ax.barh(range(len(importances)), importances[indices], color=colors, alpha=0.8)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_cols[i].replace('_', ' ').title() for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest Feature Importance\n(Graph-based Features)',
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.8, label='LL (Peptide-Peptide)'),
                      Patch(facecolor='coral', alpha=0.8, label='LP (Peptide-MHC)')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', bbox_inches='tight')
    print(f"  ✓ Saved: feature_importance.png")
    plt.close()


def plot_class_distribution(df, output_dir):
    """Plot class distribution and peptide characteristics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Class distribution
    ax = axes[0]
    class_counts = df['label_name'].value_counts()
    colors = ['lightcoral', 'lightblue']
    wedges, texts, autotexts = ax.pie(class_counts, labels=class_counts.index,
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Class Distribution\n(30 Peptides)', fontsize=12, fontweight='bold')

    # 2. LL vs LP edges scatter
    ax = axes[1]
    immuno = df[df['label'] == 1]
    non_immuno = df[df['label'] == 0]

    ax.scatter(immuno['LL_edges_mean'], immuno['LP_edges_mean'],
              c='red', s=100, alpha=0.6, label='Immunogenic', edgecolors='black', linewidth=1)
    ax.scatter(non_immuno['LL_edges_mean'], non_immuno['LP_edges_mean'],
              c='blue', s=100, alpha=0.6, label='Non-immunogenic', edgecolors='black', linewidth=1)

    ax.set_xlabel('LL Edges (Peptide-Peptide)', fontsize=10, fontweight='bold')
    ax.set_ylabel('LP Edges (Peptide-MHC)', fontsize=10, fontweight='bold')
    ax.set_title('LL vs LP Edge Counts', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Edge variability comparison
    ax = axes[2]

    x = [1, 2, 3, 4]
    labels = ['LL\nImmuno', 'LL\nNon-immuno', 'LP\nImmuno', 'LP\nNon-immuno']

    means = [
        immuno['LL_edges_std'].mean(),
        non_immuno['LL_edges_std'].mean(),
        immuno['LP_edges_std'].mean(),
        non_immuno['LP_edges_std'].mean()
    ]

    stds = [
        immuno['LL_edges_std'].std(),
        non_immuno['LL_edges_std'].std(),
        immuno['LP_edges_std'].std(),
        non_immuno['LP_edges_std'].std()
    ]

    colors_bar = ['lightcoral', 'lightblue', 'lightcoral', 'lightblue']

    bars = ax.bar(x, means, yerr=stds, color=colors_bar, alpha=0.8,
                  capsize=5, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Edge Count Std Dev', fontsize=10, fontweight='bold')
    ax.set_title('Graph Flexibility\n(Edge Count Variability)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_characteristics.png', bbox_inches='tight')
    print(f"  ✓ Saved: class_characteristics.png")
    plt.close()


def plot_auc_comparison(results_df, output_dir):
    """Plot AUC-ROC comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate graph-based and sequence-based
    graph_models = results_df[~results_df['model'].str.contains('Sequence')]
    seq_models = results_df[results_df['model'].str.contains('Sequence')]

    x_pos = np.arange(len(graph_models))
    width = 0.35

    ax.bar(x_pos - width/2, graph_models['auc'], width,
           label='Graph-based', alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, seq_models['auc'], width,
           label='Sequence-based', alpha=0.8, color='coral', edgecolor='black')

    # Add CV scores as error bars
    ax.errorbar(x_pos - width/2, graph_models['cv_mean'],
                yerr=graph_models['cv_std'], fmt='none',
                color='navy', capsize=5, linewidth=2, alpha=0.7)
    ax.errorbar(x_pos + width/2, seq_models['cv_mean'],
                yerr=seq_models['cv_std'], fmt='none',
                color='darkred', capsize=5, linewidth=2, alpha=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')

    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC-ROC / CV Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance: AUC-ROC (bars) vs 3-Fold CV (error bars)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(graph_models['model'].str.replace(' \(Sequence\)', ''),
                      rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'auc_cv_comparison.png', bbox_inches='tight')
    print(f"  ✓ Saved: auc_cv_comparison.png")
    plt.close()


def create_summary_table(results_df, output_dir):
    """Create a formatted summary table as an image"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Format data
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['model'],
            f"{row['accuracy']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['auc']:.3f}",
            f"{row['cv_mean']:.3f} ± {row['cv_std']:.3f}"
        ])

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', '3-Fold CV'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        if 'Sequence' in table_data[i-1][0]:
            color = '#ffe6e6'
        else:
            color = '#e6f2ff'

        for j in range(7):
            table[(i, j)].set_facecolor(color)

    plt.title('Machine Learning Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'results_summary_table.png', bbox_inches='tight')
    print(f"  ✓ Saved: results_summary_table.png")
    plt.close()


def main():
    """Main function"""
    print("="*60)
    print("  Visualizing ML Results")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    graphs_dir = project_root / 'md_data' / 'analysis' / 'graph_features'
    labels_file = project_root / 'md_data' / 'analysis' / 'peptide_labels.csv'
    results_file = project_root / 'md_data' / 'analysis' / 'ml_results' / 'classification_results.csv'
    output_dir = project_root / 'md_data' / 'analysis' / 'ml_results' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("\nGenerating visualizations...")

    # Load data
    df = load_data(graphs_dir, labels_file)
    results_df = pd.read_csv(results_file)

    # Generate plots
    plot_model_performance(results_df, output_dir)
    plot_feature_distributions(df, output_dir)
    plot_feature_correlations(df, output_dir)
    plot_feature_importance(df, output_dir)
    plot_class_distribution(df, output_dir)
    plot_auc_comparison(results_df, output_dir)
    create_summary_table(results_df, output_dir)

    print("\n" + "="*60)
    print(f"✓ All visualizations saved to:")
    print(f"  {output_dir}")
    print("="*60)

    # List all generated files
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
