#!/usr/bin/env python3
"""
Visualize GCN results and compare with other ML models

Creates comprehensive comparison plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_all_results(project_root):
    """Load results from all models"""

    # Load GCN results
    gcn_file = project_root / 'md_data' / 'analysis' / 'gcn_models' / 'test_results.csv'
    gcn_df = pd.read_csv(gcn_file)
    gcn_df['model'] = 'GCN (Graph)'
    gcn_df['type'] = 'Deep Learning'
    gcn_df['data_type'] = 'Graph'
    gcn_df['stride'] = 50  # NOW SAME AS CLASSICAL ML!
    gcn_df['frames_per_peptide'] = 20

    # Load classical ML results
    ml_file = project_root / 'md_data' / 'analysis' / 'ml_results' / 'classification_results.csv'
    ml_df = pd.read_csv(ml_file)
    ml_df['type'] = ml_df['model'].apply(lambda x: 'Sequence-based' if 'Sequence' in x else 'Graph-based')
    ml_df['data_type'] = ml_df['model'].apply(lambda x: 'Sequence' if 'Sequence' in x else 'Graph')
    ml_df['stride'] = 50
    ml_df['frames_per_peptide'] = 141

    # Combine
    all_results = pd.concat([gcn_df, ml_df], ignore_index=True)

    return all_results, gcn_df, ml_df


def plot_model_comparison(all_results, output_dir):
    """Create comprehensive comparison plots"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison: GCN vs Classical ML', fontsize=14, fontweight='bold')

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

    # Plot each metric
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Prepare data
        plot_data = all_results[['model', metric]].copy()
        plot_data = plot_data.sort_values(metric, ascending=False)

        # Color code: GCN in red, others by type
        colors = []
        for model in plot_data['model']:
            if 'GCN' in model:
                colors.append('#e74c3c')  # Red for GCN
            elif 'Sequence' in model:
                colors.append('#3498db')  # Blue for sequence
            else:
                colors.append('#2ecc71')  # Green for graph

        # Create bar plot
        bars = ax.barh(range(len(plot_data)), plot_data[metric], color=colors, alpha=0.7)
        ax.set_yticks(range(len(plot_data)))

        # Sadece ilk sütunda (col=0) model isimlerini göster
        if col == 0:
            ax.set_yticklabels(plot_data['model'], fontsize=8)
        else:
            ax.set_yticklabels([])  # Diğer sütunlarda Y-eksen etiketlerini gizle

        ax.set_xlabel(name, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, plot_data[metric])):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8)

        # Highlight best
        best_idx = plot_data[metric].idxmax()
        best_model = plot_data.loc[best_idx, 'model']
        if 'GCN' in best_model:
            ax.axhline(y=0, color='red', linewidth=2, alpha=0.5)

    # Remove last subplot (we only have 5 metrics)
    fig.delaxes(axes[1, 2])

    # Add legend in the empty space
    ax_legend = axes[1, 2]
    ax_legend.axis('off')
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.7, label='GCN (Deep Learning)'),
        plt.Rectangle((0, 0), 1, 1, fc='#2ecc71', alpha=0.7, label='Classical ML (Graph)'),
        plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.7, label='Classical ML (Sequence)')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, frameon=True,
                     title='Color Legend', title_fontsize=13, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_dir / 'gcn_vs_ml_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: gcn_vs_ml_comparison.png")
    plt.close()


def plot_graph_vs_sequence(all_results, output_dir):
    """Compare graph-based vs sequence-based models"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MD-based vs Sequence-based Models', fontsize=14, fontweight='bold')

    metrics = ['auc', 'f1']
    titles = ['AUC-ROC', 'F1 Score']

    for plot_idx, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        # Group by data type
        graph_models = all_results[all_results['data_type'] == 'Graph'].sort_values(metric, ascending=False)
        seq_models = all_results[all_results['data_type'] == 'Sequence'].sort_values(metric, ascending=False)

        # Ayrı bölgeler: graph solda, sequence sağda
        n_graph = len(graph_models)
        n_seq = len(seq_models)
        x_pos_graph = np.arange(n_graph)
        x_pos_seq = np.arange(n_seq) + n_graph + 1  # +1 boşluk bırak

        width = 0.7
        all_bars = []
        all_xticks = []
        all_xlabels = []

        # Plot graph models
        for i, (idx, row) in enumerate(graph_models.iterrows()):
            color = '#e74c3c' if 'GCN' in row['model'] else '#2ecc71'
            # Sadece ilk grafikte lejant göster
            if plot_idx == 0 and i == 0 and 'GCN' in row['model']:
                label = 'GCN (MD-based Deep Learning)'
            elif plot_idx == 0 and i == 1:
                label = 'Classical ML (MD-based)'
            else:
                label = ''
            bars = ax.bar(x_pos_graph[i], row[metric], width,
                         label=label, color=color, alpha=0.7)
            all_bars.append(bars)
            all_xticks.append(x_pos_graph[i])
            all_xlabels.append(row['model'].split()[0])

        # Plot sequence models
        for i, (idx, row) in enumerate(seq_models.iterrows()):
            # Sadece ilk sequence model için lejant
            label = 'Sequence Models' if (plot_idx == 0 and i == 0) else ''
            bars = ax.bar(x_pos_seq[i], row[metric], width,
                         label=label, color='#3498db', alpha=0.7)
            all_bars.append(bars)
            all_xticks.append(x_pos_seq[i])
            all_xlabels.append(row['model'].split()[0])

        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison')
        ax.set_xticks(all_xticks)
        ax.set_xticklabels(all_xlabels, rotation=45, ha='right')
        ax.set_ylim(0, 1)

        # "Graph" ve "Sequence" etiketleri - model isimlerinin çok altında
        if n_graph > 0 and n_seq > 0:
            # X-axis label pozisyonunun altına yerleştir
            ax.text((n_graph - 1) / 2, -0.20, 'MD-based Models', ha='center', va='top',
                   transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold',
                   color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3, edgecolor='darkgreen'))
            ax.text(n_graph + 1 + (n_seq - 1) / 2, -0.20, 'Sequence-based Models', ha='center', va='top',
                   transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold',
                   color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3, edgecolor='darkblue'))

        if plot_idx == 0:  # Sadece ilk grafikte lejant göster
            ax.legend(fontsize=9, loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels - tüm barlar için
        for bars_group in all_bars:
            for bar in bars_group:
                height = bar.get_height()
                if height > 0:  # Sadece 0'dan büyük değerler için
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'graph_vs_sequence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: graph_vs_sequence.png")
    plt.close()


def plot_top_models(all_results, output_dir):
    """Highlight top performing models"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top 5 models by AUC
    top_models = all_results.nlargest(5, 'auc')[['model', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'data_type']]

    # Prepare data for grouped bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    x = np.arange(len(metrics))
    width = 0.15

    # Renk paleti: Her model için unique renk ama kategoriye göre gruplandırılmış tonlar
    color_map = {
        'GCN (Graph)': '#e74c3c',  # Kırmızı - GCN
        'Gradient Boosting': '#27ae60',  # Koyu yeşil - MD-based
        'SVM (RBF)': '#7dcea0',  # Açık yeşil - MD-based
        'Random Forest (Sequence)': '#2980b9',  # Koyu mavi - Sequence
        'Gradient Boosting (Sequence)': '#5dade2',  # Orta mavi - Sequence
        'SVM (RBF) (Sequence)': '#85c1e9',  # Açık mavi - Sequence
        'Logistic Regression (Sequence)': '#aed6f1'  # Çok açık mavi - Sequence
    }

    for i, (idx, row) in enumerate(top_models.iterrows()):
        model_name = row['model']

        # Renk ve label belirleme
        if 'GCN' in model_name:
            color = color_map.get(model_name, '#e74c3c')
            label = f"{model_name} (MD-based DL)"
        elif row['data_type'] == 'Graph':
            color = color_map.get(model_name, '#2ecc71')
            label = f"{model_name} (MD-based)"
        else:
            color = color_map.get(model_name, '#3498db')
            label = f"{model_name} (Sequence-based)"

        values = [row[m] for m in metrics]
        offset = (i - len(top_models)/2) * width
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xlabel('Metrics', fontweight='bold', fontsize=11)
    ax.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax.set_title('Top 5 Models by AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)  # GCN'in Recall=1.0 barı için daha fazla boşluk

    # Lejantı daha organize şekilde göster
    ax.legend(loc='upper left', fontsize=8, frameon=True, fancybox=True, shadow=True,
              title='Model (Data Type)', title_fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_models.png', dpi=300, bbox_inches='tight')
    print(f"Saved: top_models.png")
    plt.close()


def plot_stride_comparison(all_results, output_dir):
    """Compare models trained on different strides"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Focus on graph-based models
    graph_data = all_results[all_results['data_type'] == 'Graph'].copy()

    # Create comparison
    models = graph_data['model'].tolist()
    strides = graph_data['stride'].tolist()
    frames = graph_data['frames_per_peptide'].tolist()
    aucs = graph_data['auc'].tolist()

    colors = ['#e74c3c' if 'GCN' in m else '#2ecc71' for m in models]

    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, aucs, color=colors, alpha=0.7)

    ax.set_ylabel('AUC-ROC', fontweight='bold')
    ax.set_title('Graph-based Models (All: Stride=50)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Sadece AUC değerlerini göster - stride/frame bilgisi başlıkta
    for i, (bar, frame, auc) in enumerate(zip(bars, frames, aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{auc:.3f}',  # Sadece AUC değeri
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Frame sayısını bar içinde göster
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{frame} frames',
               ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'stride_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: stride_comparison.png")
    plt.close()


def create_summary_table(all_results, output_dir):
    """Create a summary table with all results"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = all_results[['model', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'stride', 'frames_per_peptide', 'data_type']].copy()
    table_data = table_data.sort_values('auc', ascending=False)

    # Model isimlerine MD-based veya Sequence-based ekle
    def add_data_type_label(row):
        if 'GCN' in row['model']:
            return f"{row['model']} (MD-based DL)"
        elif row['data_type'] == 'Graph':
            return f"{row['model']} (MD-based)"
        else:
            return f"{row['model']} (Seq-based)"

    table_data['model'] = table_data.apply(add_data_type_label, axis=1)

    # Round values
    for col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        table_data[col] = table_data[col].round(3)

    # Drop data_type column and rename
    table_data = table_data.drop('data_type', axis=1)
    table_data.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Stride', 'Frames/Peptide']

    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)  # Biraz küçültüldü - model isimlerinin sığması için
    table.scale(1, 2.2)  # Sadece yükseklik - genişlik manuel ayarlanacak

    # Manuel sütun genişlikleri ayarla
    for i in range(len(table_data)):
        # Model sütunu (0) - orta genişlik
        table[(i+1, 0)].set_width(0.26)  # Model sütunu - "(MD-based)" için yeterli
        table[(i+1, 1)].set_width(0.095)  # Accuracy
        table[(i+1, 2)].set_width(0.095)  # Precision
        table[(i+1, 3)].set_width(0.095)  # Recall
        table[(i+1, 4)].set_width(0.095)  # F1
        table[(i+1, 5)].set_width(0.095)  # AUC
        table[(i+1, 6)].set_width(0.095)  # Stride
        table[(i+1, 7)].set_width(0.13)  # Frames/Peptide - biraz geniş

    # Header sütun genişlikleri
    table[(0, 0)].set_width(0.26)  # Model header
    for i in range(1, 7):
        table[(0, i)].set_width(0.095)
    table[(0, 7)].set_width(0.13)  # Frames/Peptide header

    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Model sütununu sola yasla
    for i in range(len(table_data)):
        table[(i+1, 0)].set_text_props(ha='left')

    # En iyi değerleri sadece bold yap (renksiz)
    for col_idx, col in enumerate(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']):
        max_val = table_data[col].max()
        for row_idx, val in enumerate(table_data[col]):
            if val == max_val:
                table[(row_idx+1, col_idx+1)].set_text_props(weight='bold')

    plt.title('Complete Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'results_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results_table.png")
    plt.close()


def plot_gcn_architecture(output_dir):
    """Visualize GCN architecture"""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'GCN Architecture for Peptide Immunogenicity Prediction',
            ha='center', fontsize=14, fontweight='bold')

    # Draw architecture
    layers = [
        ('Input: LP Graph\n(Peptide-MHC contacts)', 8.5, '#3498db', 'Node features: 72-dim\nEdges: distance < 8Å'),
        ('GCN Conv 1\n(72 → 128)', 7.5, '#2ecc71', 'ReLU + BatchNorm + Dropout'),
        ('GCN Conv 2\n(128 → 128)', 6.5, '#2ecc71', 'ReLU + BatchNorm + Dropout'),
        ('GCN Conv 3\n(128 → 128)', 5.5, '#2ecc71', 'ReLU + BatchNorm + Dropout'),
        ('Global Pooling\n(Max + Mean)', 4.5, '#e67e22', '2 × 128 = 256 features'),
        ('FC Layer 1\n(256 → 128)', 3.5, '#9b59b6', 'ReLU + Dropout'),
        ('FC Layer 2\n(128 → 64)', 2.5, '#9b59b6', 'ReLU + Dropout'),
        ('Output Layer\n(64 → 2)', 1.5, '#e74c3c', 'Binary classification'),
    ]

    for layer_name, y_pos, color, detail in layers:
        # Main box
        rect = plt.Rectangle((2, y_pos-0.3), 6, 0.6,
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(5, y_pos, layer_name, ha='center', va='center',
               fontweight='bold', fontsize=10)

        # Detail text
        ax.text(8.5, y_pos, detail, ha='left', va='center',
               fontsize=8, style='italic', color='gray')

        # Arrow
        if y_pos > 1.5:
            ax.arrow(5, y_pos-0.35, 0, -0.35, head_width=0.2,
                    head_length=0.1, fc='black', ec='black')

    # Add training details
    details_text = """
Training Configuration:
• Dataset: 30 peptides (15 immunogenic, 15 non-immunogenic)
• Frames: 10 per peptide (sampled from stride=100 data)
• Total samples: 300 (240 train, 60 test)
• Split: 80/20 by peptide (prevents data leakage)
• Optimizer: Adam (lr=0.001, weight_decay=1e-5)
• Loss: Cross-Entropy
• Early stopping: patience=20 epochs
• Training time: ~2 minutes on CPU
    """
    ax.text(0.5, 0.3, details_text, ha='left', va='top',
           fontsize=8, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_dir / 'gcn_architecture.png', dpi=300, bbox_inches='tight')
    print(f"Saved: gcn_architecture.png")
    plt.close()


def main():
    print("="*60)
    print("  GCN Results Visualization")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'md_data' / 'analysis' / 'gcn_models' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Load all results
    print("\nLoading results...")
    all_results, gcn_df, ml_df = load_all_results(project_root)

    print(f"  GCN results: {len(gcn_df)} entries")
    print(f"  Classical ML results: {len(ml_df)} entries")
    print(f"  Total: {len(all_results)} entries")

    # Create visualizations
    print("\nCreating visualizations...")

    plot_model_comparison(all_results, output_dir)
    plot_graph_vs_sequence(all_results, output_dir)
    plot_top_models(all_results, output_dir)
    plot_stride_comparison(all_results, output_dir)
    create_summary_table(all_results, output_dir)
    plot_gcn_architecture(output_dir)

    print("\n" + "="*60)
    print(f"All visualizations saved to:")
    print(f"  {output_dir}")
    print("="*60)

    # Print summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nBest Models by AUC:")
    top_5 = all_results.nlargest(5, 'auc')[['model', 'auc', 'accuracy', 'f1', 'stride', 'frames_per_peptide']]
    for idx, row in top_5.iterrows():
        print(f"  {row['model']:40s} AUC: {row['auc']:.3f} | Acc: {row['accuracy']:.3f} | F1: {row['f1']:.3f}")
        print(f"  {'':40s} → Stride: {row['stride']}, Frames/peptide: {row['frames_per_peptide']}")

    print("\nGCN Performance:")
    gcn_row = gcn_df.iloc[0]
    print(f"  AUC:       {gcn_row['auc']:.3f}")
    print(f"  Accuracy:  {gcn_row['accuracy']:.3f}")
    print(f"  Precision: {gcn_row['precision']:.3f}")
    print(f"  Recall:    {gcn_row['recall']:.3f}")
    print(f"  F1 Score:  {gcn_row['f1']:.3f}")
    print(f"  Data:      Stride={gcn_row['stride']}, {gcn_row['frames_per_peptide']} frames/peptide")

    print("\nGraph-based Models (Best AUC):")
    graph_models = all_results[all_results['data_type'] == 'Graph'].nlargest(3, 'auc')
    for idx, row in graph_models.iterrows():
        print(f"  {row['model']:40s} AUC: {row['auc']:.3f}")

    print("\nSequence-based Models (Best AUC):")
    seq_models = all_results[all_results['data_type'] == 'Sequence'].nlargest(3, 'auc')
    for idx, row in seq_models.iterrows():
        print(f"  {row['model']:40s} AUC: {row['auc']:.3f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
