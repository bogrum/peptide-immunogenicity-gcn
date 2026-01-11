#!/usr/bin/env python3
"""
Train a simpler classifier for peptide immunogenicity prediction using graph features

Approach:
- Aggregate graph statistics per peptide (mean/std of LL and LP edge counts)
- Use traditional ML classifiers (Random Forest, SVM, Logistic Regression)
- Compare with sequence-based baseline

Usage: python train_simple_classifier.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


def aggregate_graph_features(graphs_dir, labels_file):
    """
    Aggregate graph features per peptide

    Features extracted:
    - Mean/std of LL edges (peptide-peptide contacts)
    - Mean/std of LP edges (peptide-MHC contacts)
    - Min/max of LL edges
    - Min/max of LP edges
    """
    # Load labels
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['sequence'], labels_df['label']))

    print(f"\nAggregating graph features from {graphs_dir}")

    features_list = []

    for pkl_file in sorted(Path(graphs_dir).glob('*_graphs.pkl')):
        peptide = pkl_file.stem.replace('_graphs', '')

        if peptide not in labels_dict:
            print(f"  Warning: {peptide} not in labels, skipping")
            continue

        # Load graph data
        with open(pkl_file, 'rb') as f:
            graph_data = pickle.load(f)

        # Get summary statistics (already computed during extraction)
        summary = graph_data['summary']

        # Create feature vector
        features = {
            'peptide': peptide,
            'label': labels_dict[peptide],

            # LL graph features (peptide-peptide contacts)
            'LL_edges_mean': summary['LL_edges_mean'],
            'LL_edges_std': summary['LL_edges_std'],
            'LL_edges_min': summary['LL_edges_min'],
            'LL_edges_max': summary['LL_edges_max'],

            # LP graph features (peptide-MHC contacts)
            'LP_edges_mean': summary['LP_edges_mean'],
            'LP_edges_std': summary['LP_edges_std'],
            'LP_edges_min': summary['LP_edges_min'],
            'LP_edges_max': summary['LP_edges_max'],

            # Derived features
            'LL_LP_ratio_mean': summary['LL_edges_mean'] / summary['LP_edges_mean'] if summary['LP_edges_mean'] > 0 else 0,
            'total_edges_mean': summary['LL_edges_mean'] + summary['LP_edges_mean'],
            'edge_variability_LL': summary['LL_edges_std'] / summary['LL_edges_mean'] if summary['LL_edges_mean'] > 0 else 0,
            'edge_variability_LP': summary['LP_edges_std'] / summary['LP_edges_mean'] if summary['LP_edges_mean'] > 0 else 0,
        }

        features_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(features_list)

    print(f"\nLoaded features for {len(df)} peptides")
    print(f"  Immunogenic (1): {sum(df['label'])} ({100*sum(df['label'])/len(df):.1f}%)")
    print(f"  Non-immunogenic (0): {len(df) - sum(df['label'])} ({100*(len(df)-sum(df['label']))/len(df):.1f}%)")

    return df


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, classifier, name):
    """
    Train and evaluate a classifier
    """
    print(f"\n{name}:")
    print("-" * 60)

    # Train
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    if y_proba is not None and len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  AUC-ROC:   {auc:.4f}")
    else:
        auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]:<4} FP: {cm[0, 1]:<4}")
    print(f"    FN: {cm[1, 0]:<4} TP: {cm[1, 1]:<4}")

    # Cross-validation on training set
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=3, scoring='accuracy')
    print(f"\n  3-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results = {
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    return results, classifier


def create_sequence_baseline(df):
    """
    Create simple sequence-based features as baseline

    Features:
    - Amino acid composition (20 features)
    - Hydrophobicity score
    - Charge
    - Molecular weight (approximation)
    """
    from collections import Counter

    # Amino acid properties
    aa_properties = {
        'A': {'hydro': 1.8, 'charge': 0, 'mw': 89},
        'R': {'hydro': -4.5, 'charge': 1, 'mw': 174},
        'N': {'hydro': -3.5, 'charge': 0, 'mw': 132},
        'D': {'hydro': -3.5, 'charge': -1, 'mw': 133},
        'C': {'hydro': 2.5, 'charge': 0, 'mw': 121},
        'Q': {'hydro': -3.5, 'charge': 0, 'mw': 146},
        'E': {'hydro': -3.5, 'charge': -1, 'mw': 147},
        'G': {'hydro': -0.4, 'charge': 0, 'mw': 75},
        'H': {'hydro': -3.2, 'charge': 0.5, 'mw': 155},
        'I': {'hydro': 4.5, 'charge': 0, 'mw': 131},
        'L': {'hydro': 3.8, 'charge': 0, 'mw': 131},
        'K': {'hydro': -3.9, 'charge': 1, 'mw': 146},
        'M': {'hydro': 1.9, 'charge': 0, 'mw': 149},
        'F': {'hydro': 2.8, 'charge': 0, 'mw': 165},
        'P': {'hydro': -1.6, 'charge': 0, 'mw': 115},
        'S': {'hydro': -0.8, 'charge': 0, 'mw': 105},
        'T': {'hydro': -0.7, 'charge': 0, 'mw': 119},
        'W': {'hydro': -0.9, 'charge': 0, 'mw': 204},
        'Y': {'hydro': -1.3, 'charge': 0, 'mw': 181},
        'V': {'hydro': 4.2, 'charge': 0, 'mw': 117}
    }

    sequence_features = []

    for peptide in df['peptide']:
        # AA composition
        counts = Counter(peptide)
        aa_comp = {f'aa_{aa}': counts.get(aa, 0) / len(peptide) for aa in 'ACDEFGHIKLMNPQRSTVWY'}

        # Aggregate properties
        hydro = np.mean([aa_properties.get(aa, {}).get('hydro', 0) for aa in peptide])
        charge = np.sum([aa_properties.get(aa, {}).get('charge', 0) for aa in peptide])
        mw = np.sum([aa_properties.get(aa, {}).get('mw', 0) for aa in peptide])

        features = {**aa_comp, 'hydrophobicity': hydro, 'charge': charge, 'molecular_weight': mw}
        sequence_features.append(features)

    seq_df = pd.DataFrame(sequence_features)
    return seq_df


def main():
    """
    Main function
    """
    print("="*60)
    print("  Peptide Immunogenicity Classification")
    print("  Graph Features + Traditional ML")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    graphs_dir = project_root / 'md_data' / 'analysis' / 'graph_features'
    labels_file = project_root / 'md_data' / 'analysis' / 'peptide_labels.csv'
    output_dir = project_root / 'md_data' / 'analysis' / 'ml_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate features
    df = aggregate_graph_features(graphs_dir, labels_file)

    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['peptide', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    peptides = df['peptide'].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {', '.join(feature_cols)}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test, pep_train, pep_test = train_test_split(
        X, y, peptides, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} peptides ({sum(y_train)} immunogenic)")
    print(f"  Test: {len(X_test)} peptides ({sum(y_test)} immunogenic)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers
    classifiers = [
        (RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
         "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
         "Gradient Boosting"),
        (SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
         "SVM (RBF)"),
        (LogisticRegression(C=1.0, max_iter=1000, random_state=42),
         "Logistic Regression"),
    ]

    # Train and evaluate
    print("\n" + "="*60)
    print("  Graph-Based Model Results")
    print("="*60)

    all_results = []

    for clf, name in classifiers:
        results, trained_clf = train_and_evaluate_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, clf, name
        )
        all_results.append(results)

    # Sequence baseline
    print("\n" + "="*60)
    print("  Sequence-Based Baseline")
    print("="*60)

    seq_df = create_sequence_baseline(df)
    X_seq = seq_df.values

    X_seq_train, X_seq_test = X_seq[df['peptide'].isin(pep_train)], X_seq[df['peptide'].isin(pep_test)]

    # Scale sequence features
    scaler_seq = StandardScaler()
    X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train)
    X_seq_test_scaled = scaler_seq.transform(X_seq_test)

    for clf, name in classifiers:
        results, _ = train_and_evaluate_classifier(
            X_seq_train_scaled, y_train, X_seq_test_scaled, y_test, clf, f"{name} (Sequence)"
        )
        all_results.append(results)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / 'classification_results.csv'
    results_df.to_csv(results_file, index=False)

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_file}")

    # Save test predictions
    test_results_df = pd.DataFrame({
        'peptide': pep_test,
        'true_label': y_test,
    })
    test_results_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    print(f"Test predictions saved to: {output_dir / 'test_predictions.csv'}")


if __name__ == '__main__':
    main()
