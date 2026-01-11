#!/usr/bin/env python3
"""
Memory-efficient GCN training for peptide immunogenicity prediction
Loads data on-the-fly instead of loading everything into memory

Usage: python train_gcn_model_efficient.py [--use-stride100]
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import argparse


class PeptideGCN(nn.Module):
    """Graph Convolutional Network for peptide immunogenicity prediction"""

    def __init__(self, input_dim=72, hidden_dim=128, num_conv_layers=3, dropout=0.3):
        super(PeptideGCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_max, x_mean], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class LazyGraphDataset(Dataset):
    """
    Memory-efficient dataset that loads graphs on-the-fly
    """
    def __init__(self, graphs_dir, labels_file, peptide_list=None, max_frames_per_peptide=None):
        """
        Args:
            graphs_dir: Directory with pkl files
            labels_file: CSV with labels
            peptide_list: List of peptides to include (for train/test split)
            max_frames_per_peptide: Limit frames per peptide to reduce memory
        """
        self.graphs_dir = Path(graphs_dir)

        # Load labels
        labels_df = pd.read_csv(labels_file)
        self.labels_dict = dict(zip(labels_df['sequence'], labels_df['label']))

        # Build index of (peptide, frame_idx) tuples
        self.samples = []

        for pkl_file in sorted(self.graphs_dir.glob('*_graphs.pkl')):
            peptide = pkl_file.stem.replace('_graphs', '')

            if peptide not in self.labels_dict:
                continue

            if peptide_list is not None and peptide not in peptide_list:
                continue

            # Count frames without loading full file
            with open(pkl_file, 'rb') as f:
                graph_data = pickle.load(f)
                n_frames = len(graph_data['features'])

            # Limit frames if specified
            if max_frames_per_peptide is not None:
                n_frames = min(n_frames, max_frames_per_peptide)

            # Add all frame indices for this peptide
            for frame_idx in range(n_frames):
                self.samples.append({
                    'peptide': peptide,
                    'pkl_file': pkl_file,
                    'frame_idx': frame_idx,
                    'label': self.labels_dict[peptide]
                })

        print(f"Dataset created: {len(self.samples)} samples from {len(set([s['peptide'] for s in self.samples]))} peptides")

        # Print class distribution
        labels = [s['label'] for s in self.samples]
        print(f"  Immunogenic (1): {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
        print(f"  Non-immunogenic (0): {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a single graph on-the-fly"""
        sample_info = self.samples[idx]

        # Load only the specific frame we need
        with open(sample_info['pkl_file'], 'rb') as f:
            graph_data = pickle.load(f)

        frame_data = graph_data['features'][sample_info['frame_idx']]

        # Convert to PyTorch Geometric Data
        return self._create_pytorch_geometric_data(frame_data, sample_info['label'])

    def _create_pytorch_geometric_data(self, frame_data, label):
        """Convert graph dictionary to PyTorch Geometric Data object"""
        # Use LP graph (peptide-MHC interactions)
        adj_matrix = frame_data['LP_adj']
        weight_matrix = frame_data['LP_weights']

        # Convert adjacency to edge_index and edge_attr
        edge_index_np = np.array(np.where(adj_matrix > 0))
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        # Get edge weights
        edge_weights = weight_matrix[edge_index_np[0], edge_index_np[1]]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        # Node features: one-hot encoding
        n_nodes = adj_matrix.shape[0]
        x = torch.eye(n_nodes, dtype=torch.float)

        # Adjust to fixed size (72 dimensions)
        if n_nodes > 72:
            x = x[:, :72]
        elif n_nodes < 72:
            padding = torch.zeros((n_nodes, 72 - n_nodes))
            x = torch.cat([x, padding], dim=1)

        # Label
        y = torch.tensor([label], dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )


def collate_fn(batch):
    """Custom collate function for PyTorch Geometric Data objects"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def train_model(model, train_loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(), batch.batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate_model(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(), batch.batch)
            loss = criterion(out, batch.y)

            total_loss += loss.item()

            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GCN model for peptide immunogenicity')
    parser.add_argument('--use-stride100', action='store_true',
                       help='Use stride100 data (smaller, 71 frames/peptide)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Max frames per peptide (for memory efficiency)')
    args = parser.parse_args()

    print("="*60)
    print("  Memory-Efficient GCN Training")
    print("  Following Weber et al. 2024 (bbad504)")
    print("="*60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Paths
    project_root = Path(__file__).parent.parent.parent

    if args.use_stride100:
        graphs_dir = project_root / 'md_data' / 'analysis' / 'graph_features_stride100'
        print("\nUsing stride=100 data (71 frames per peptide)")
    else:
        graphs_dir = project_root / 'md_data' / 'analysis' / 'graph_features'
        print("\nUsing stride=50 data (141 frames per peptide)")

    if args.max_frames:
        print(f"Limiting to {args.max_frames} frames per peptide")

    labels_file = project_root / 'md_data' / 'analysis' / 'peptide_labels.csv'
    output_dir = project_root / 'md_data' / 'analysis' / 'gcn_models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Get unique peptides for splitting
    labels_df = pd.read_csv(labels_file)
    unique_peptides = labels_df['sequence'].tolist()
    peptide_labels = dict(zip(labels_df['sequence'], labels_df['label']))

    # Train/test split by peptide (stratified)
    train_peptides, test_peptides = train_test_split(
        unique_peptides,
        test_size=0.2,
        stratify=[peptide_labels[p] for p in unique_peptides],
        random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training peptides: {len(train_peptides)}")
    print(f"  Test peptides: {len(test_peptides)}")

    # Create lazy datasets
    print("\nCreating training dataset...")
    train_dataset = LazyGraphDataset(
        graphs_dir,
        labels_file,
        peptide_list=train_peptides,
        max_frames_per_peptide=args.max_frames
    )

    print("\nCreating test dataset...")
    test_dataset = LazyGraphDataset(
        graphs_dir,
        labels_file,
        peptide_list=test_peptides,
        max_frames_per_peptide=args.max_frames
    )

    # Create data loaders
    batch_size = 32
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 to avoid multiprocessing issues with large files
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Initialize model
    model = PeptideGCN(input_dim=72, hidden_dim=128, num_conv_layers=3, dropout=0.3)
    model = model.to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Training loop
    num_epochs = 100
    best_val_auc = 0.0
    patience = 20
    patience_counter = 0

    print(f"\nTraining for {num_epochs} epochs...")
    print("="*60)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)

        # Evaluate
        train_metrics = evaluate_model(model, train_loader, criterion, device)
        test_metrics = evaluate_model(model, test_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(test_metrics['loss'])

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")

        # Save best model
        if test_metrics['auc'] > best_val_auc:
            best_val_auc = test_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': test_metrics,
            }, output_dir / 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best test AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")

    # Load best model and final evaluation
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = evaluate_model(model, test_loader, criterion, device)

    print("\nFinal Test Performance:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {final_metrics['auc']:.4f}")
    print("="*60)

    # Save results
    results_df = pd.DataFrame([final_metrics])
    results_df.to_csv(output_dir / 'test_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'test_results.csv'}")


if __name__ == '__main__':
    main()
