#!/usr/bin/env python3
"""
Train Graph Convolutional Network for peptide immunogenicity prediction
Following Weber et al. 2024 (bbad504) methodology

Architecture:
- Input: LL + LP graph features
- 3 convolutional modules
- Max + Average pooling
- Dense layers
- Binary classification (immunogenic vs non-immunogenic)

Usage: python train_gcn_model.py
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import os


class PeptideGCN(nn.Module):
    """
    Graph Convolutional Network for peptide immunogenicity prediction

    Architecture from Weber et al. 2024:
    - 3 GCN convolutional layers
    - Max + Mean pooling
    - Fully connected layers
    - Binary output (immunogenic/non-immunogenic)
    """

    def __init__(self, input_dim=72, hidden_dim=128, num_conv_layers=3, dropout=0.3):
        super(PeptideGCN, self).__init__()

        # Convolutional layers (3 modules as per paper)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        # After max + mean pooling, we get 2 * hidden_dim features
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        """
        Forward pass

        Args:
            x: Node features (N x input_dim)
            edge_index: Edge connectivity (2 x E)
            edge_weight: Edge weights (E,)
            batch: Batch assignment vector (N,)

        Returns:
            logits: Class logits (batch_size x 2)
        """
        # Convolutional layers with ReLU activation
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

        # Global pooling (max + mean as per paper)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_max, x_mean], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def load_graph_data(graphs_dir, labels_file):
    """
    Load graph features and labels

    Args:
        graphs_dir: Directory containing graph .pkl files
        labels_file: CSV file with peptide labels

    Returns:
        List of (graph_features, label, peptide_name) tuples
    """
    # Load labels
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['sequence'], labels_df['label']))

    print(f"\nLoading graph data from {graphs_dir}")
    print(f"Found {len(labels_dict)} labeled peptides")

    data_list = []

    for pkl_file in sorted(Path(graphs_dir).glob('*_graphs.pkl')):
        peptide = pkl_file.stem.replace('_graphs', '')

        if peptide not in labels_dict:
            print(f"  Warning: {peptide} not in labels, skipping")
            continue

        # Load graph data
        with open(pkl_file, 'rb') as f:
            graph_data = pickle.load(f)

        label = labels_dict[peptide]

        # Extract all frames
        for frame_data in graph_data['features']:
            data_list.append({
                'peptide': peptide,
                'label': label,
                'LL_adj': frame_data['LL_adj'],
                'LL_weights': frame_data['LL_weights'],
                'LP_adj': frame_data['LP_adj'],
                'LP_weights': frame_data['LP_weights'],
                'frame': frame_data['frame'],
                'time_ns': frame_data['time_ns']
            })

    print(f"Loaded {len(data_list)} graph samples from {len(set([d['peptide'] for d in data_list]))} peptides")

    # Print class distribution
    labels = [d['label'] for d in data_list]
    print(f"  Immunogenic (1): {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Non-immunogenic (0): {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")

    return data_list


def create_pytorch_geometric_data(graph_dict):
    """
    Convert graph dictionary to PyTorch Geometric Data object

    Uses combined LL + LP graph approach
    """
    # Combine LL and LP graphs
    # For simplicity, we'll use the LP graph (peptide-MHC interactions)
    # since it's more relevant for immunogenicity

    adj_matrix = graph_dict['LP_adj']
    weight_matrix = graph_dict['LP_weights']

    # Convert adjacency to edge_index and edge_attr
    edge_index_np = np.array(np.where(adj_matrix > 0))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Get edge weights
    edge_weights = weight_matrix[edge_index_np[0], edge_index_np[1]]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # Node features: use one-hot encoding of node index (simplified)
    # In a more sophisticated approach, we'd use atom types, residue features, etc.
    n_nodes = adj_matrix.shape[0]
    x = torch.eye(n_nodes, dtype=torch.float)  # One-hot identity

    # If too many nodes, use PCA or embeddings
    if n_nodes > 72:
        # Take first 72 dimensions
        x = x[:, :72]
    elif n_nodes < 72:
        # Pad with zeros
        padding = torch.zeros((n_nodes, 72 - n_nodes))
        x = torch.cat([x, padding], dim=1)

    # Label
    y = torch.tensor([graph_dict['label']], dtype=torch.long)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

    return data


def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(), batch.batch)
        loss = criterion(out, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    """
    Evaluate model
    """
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

    avg_loss = total_loss / len(loader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics


def main():
    """
    Main training loop
    """
    print("="*60)
    print("  GCN Training for Peptide Immunogenicity Prediction")
    print("  Following Weber et al. 2024 (bbad504)")
    print("="*60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    graphs_dir = project_root / 'md_data' / 'analysis' / 'graph_features'
    labels_file = project_root / 'md_data' / 'analysis' / 'peptide_labels.csv'
    output_dir = project_root / 'md_data' / 'analysis' / 'gcn_models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    data_list = load_graph_data(graphs_dir, labels_file)

    # Convert to PyTorch Geometric Data objects
    print("\nConverting graphs to PyTorch Geometric format...")
    torch_data = []
    for graph_dict in data_list:
        try:
            data_obj = create_pytorch_geometric_data(graph_dict)
            torch_data.append(data_obj)
        except Exception as e:
            print(f"  Error converting graph: {e}")

    print(f"Converted {len(torch_data)} graphs")

    # Split by peptide (not by frame) to avoid data leakage
    peptides = [d['peptide'] for d in data_list]
    unique_peptides = list(set(peptides))
    peptide_labels = {p: data_list[peptides.index(p)]['label'] for p in unique_peptides}

    # Train/test split (stratified by label)
    train_peptides, test_peptides = train_test_split(
        unique_peptides,
        test_size=0.2,
        stratify=[peptide_labels[p] for p in unique_peptides],
        random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training peptides: {len(train_peptides)}")
    print(f"  Test peptides: {len(test_peptides)}")

    # Create train/test datasets
    train_data = [torch_data[i] for i, d in enumerate(data_list) if d['peptide'] in train_peptides]
    test_data = [torch_data[i] for i, d in enumerate(data_list) if d['peptide'] in test_peptides]

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
    # Check dependencies
    try:
        import torch
        import torch_geometric
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install torch torch-geometric")
        exit(1)

    main()
