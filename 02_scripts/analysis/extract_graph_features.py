#!/usr/bin/env python3
"""
Extract graph features from MD trajectories for GCN-based immunogenicity prediction
Following Weber et al. 2024 (bbad504) methodology

Graph Construction:
1. LL Graph (Ligand-Ligand): Intramolecular peptide contacts
2. LP Graph (Ligand-Protein): Intermolecular peptide-MHC contacts

Parameters (from paper, page 10, Methods):
- Global 8 Å cutoff
- Heavy atoms only
- Exclude covalent bonds (within 2 bonded radii)
- Exponential distance weighting

Usage: python extract_graph_features.py
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import networkx as nx
import os
from pathlib import Path
import pickle
from tqdm.auto import tqdm

# Covalent bond distance threshold (2 bonded radii)
COVALENT_CUTOFF = 2.0  # Å (typical C-C bond is ~1.5 Å, so 2 Å excludes direct bonds)
GLOBAL_CUTOFF = 8.0    # Å (from Weber et al. 2024)
FRAME_STRIDE = 50      # Sample every Nth frame to reduce memory usage (was 100, now 50 for 2x data)


def get_heavy_atoms(atom_group):
    """
    Select only heavy atoms (exclude hydrogens)
    """
    return atom_group.select_atoms('not name H*')


def is_covalent_bond(atom1_idx, atom2_idx, peptide_atoms):
    """
    Check if two atoms are covalently bonded (within same residue or adjacent residues)
    Weber et al.: exclude contacts within 2 bonded radii
    """
    res1 = peptide_atoms[atom1_idx].resid
    res2 = peptide_atoms[atom2_idx].resid

    # Same residue or adjacent residues = likely covalent
    if abs(res1 - res2) <= 1:
        return True
    return False


def compute_exponential_weights(distances_array, alpha=1.0):
    """
    Compute exponential distance weights: w = exp(-alpha * d)
    Weber et al. uses exponential weighting for distance-resolved graphs
    """
    return np.exp(-alpha * distances_array)


def build_LL_graph(peptide_heavy, frame_idx):
    """
    Build LL (Ligand-Ligand) graph: intramolecular peptide contacts

    Parameters from Weber et al. 2024:
    - 8 Å cutoff (global)
    - Heavy atoms only
    - Exclude covalent bonds (within 2 bonded radii)
    - Distance weighting (exponential)

    Returns:
        NetworkX graph with node features and edge weights
    """
    G = nx.Graph()

    # Add nodes (all heavy atoms in peptide)
    for i, atom in enumerate(peptide_heavy):
        G.add_node(i,
                   atom_type=atom.name,
                   resname=atom.resname,
                   resid=atom.resid,
                   position=atom.position.copy())

    # Compute pairwise distances
    n_atoms = len(peptide_heavy)
    coords = peptide_heavy.positions

    # Calculate distance matrix
    dist_matrix = distances.distance_array(coords, coords)

    # Add edges for contacts within 8 Å, excluding covalent bonds
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = dist_matrix[i, j]

            # Check cutoff
            if dist > GLOBAL_CUTOFF:
                continue

            # Exclude covalent bonds
            if is_covalent_bond(i, j, peptide_heavy):
                continue

            # Add edge with exponential distance weight
            weight = np.exp(-dist)
            G.add_edge(i, j, distance=dist, weight=weight)

    return G


def build_LP_graph(peptide_heavy, mhc_heavy, frame_idx):
    """
    Build LP (Ligand-Protein) graph: intermolecular peptide-MHC contacts

    Parameters from Weber et al. 2024:
    - 8 Å cutoff (global)
    - Heavy atoms only (both peptide and MHC)
    - Distance weighting (exponential)

    Returns:
        NetworkX graph (bipartite: peptide nodes + MHC nodes)
    """
    G = nx.Graph()

    n_pep = len(peptide_heavy)
    n_mhc = len(mhc_heavy)

    # Add peptide nodes (0 to n_pep-1)
    for i, atom in enumerate(peptide_heavy):
        G.add_node(i,
                   node_type='peptide',
                   atom_type=atom.name,
                   resname=atom.resname,
                   resid=atom.resid,
                   position=atom.position.copy())

    # Add MHC nodes (n_pep to n_pep+n_mhc-1)
    for i, atom in enumerate(mhc_heavy):
        G.add_node(n_pep + i,
                   node_type='mhc',
                   atom_type=atom.name,
                   resname=atom.resname,
                   resid=atom.resid,
                   position=atom.position.copy())

    # Compute pairwise distances (peptide to MHC)
    pep_coords = peptide_heavy.positions
    mhc_coords = mhc_heavy.positions
    dist_matrix = distances.distance_array(pep_coords, mhc_coords)

    # Add edges for contacts within 8 Å
    for i in range(n_pep):
        for j in range(n_mhc):
            dist = dist_matrix[i, j]

            if dist <= GLOBAL_CUTOFF:
                # Add edge with exponential distance weight
                weight = np.exp(-dist)
                G.add_edge(i, n_pep + j, distance=dist, weight=weight)

    return G


def graph_to_features(G_LL, G_LP):
    """
    Convert graphs to feature matrices for GCN input

    Returns:
        dict with:
        - LL_adj: adjacency matrix for LL graph
        - LL_weights: edge weights for LL graph
        - LP_adj: adjacency matrix for LP graph
        - LP_weights: edge weights for LP graph
        - n_pep_atoms: number of peptide heavy atoms
        - n_mhc_atoms: number of MHC heavy atoms
    """
    features = {}

    # LL graph features
    if G_LL.number_of_nodes() > 0:
        LL_adj = nx.to_numpy_array(G_LL, weight=None)  # Binary adjacency
        LL_weights = nx.to_numpy_array(G_LL, weight='weight')  # Weighted adjacency
        features['LL_adj'] = LL_adj
        features['LL_weights'] = LL_weights
        features['n_pep_atoms'] = G_LL.number_of_nodes()
    else:
        features['LL_adj'] = np.array([])
        features['LL_weights'] = np.array([])
        features['n_pep_atoms'] = 0

    # LP graph features
    if G_LP.number_of_nodes() > 0:
        LP_adj = nx.to_numpy_array(G_LP, weight=None)
        LP_weights = nx.to_numpy_array(G_LP, weight='weight')
        features['LP_adj'] = LP_adj
        features['LP_weights'] = LP_weights

        # Count peptide vs MHC nodes
        peptide_nodes = [n for n, d in G_LP.nodes(data=True) if d.get('node_type') == 'peptide']
        mhc_nodes = [n for n, d in G_LP.nodes(data=True) if d.get('node_type') == 'mhc']
        features['n_mhc_atoms'] = len(mhc_nodes)
    else:
        features['LP_adj'] = np.array([])
        features['LP_weights'] = np.array([])
        features['n_mhc_atoms'] = 0

    return features


def extract_graphs_for_peptide(peptide_name, system_dir, output_dir):
    """
    Extract LL and LP graphs for all frames of a peptide trajectory

    Args:
        peptide_name: Peptide sequence identifier
        system_dir: Directory containing md.tpr and processed trajectory
        output_dir: Directory to save graph features

    Returns:
        Dictionary with aggregated graph statistics
    """
    print(f"\n{'='*60}")
    print(f"Processing: {peptide_name}")
    print(f"{'='*60}")

    # File paths
    tpr_file = os.path.join(system_dir, 'md.tpr')
    xtc_file = os.path.join(system_dir, 'md_30-100ns.xtc')

    # Check files exist
    if not os.path.exists(tpr_file):
        print(f"  ✗ Missing topology: {tpr_file}")
        return None

    if not os.path.exists(xtc_file):
        print(f"  ✗ Missing trajectory: {xtc_file}")
        print(f"     Run process_all_trajectories.sh first!")
        return None

    # Load trajectory
    print(f"  Loading trajectory...")
    u = mda.Universe(tpr_file, xtc_file)

    # Select peptide and MHC
    # Peptide is typically chain P or last 9 residues
    try:
        peptide = u.select_atoms('segid PEPTIDE')
        if len(peptide) == 0:
            # Try alternative selection: last 9 residues of protein
            all_protein = u.select_atoms('protein')
            max_resid = max([r.resid for r in all_protein.residues])
            peptide = u.select_atoms(f'protein and resid {max_resid-8}-{max_resid}')
    except:
        # Fallback: assume last 9 residues
        all_protein = u.select_atoms('protein')
        max_resid = max([r.resid for r in all_protein.residues])
        peptide = u.select_atoms(f'protein and resid {max_resid-8}-{max_resid}')

    mhc = u.select_atoms('protein and not group peptide_atoms', peptide_atoms=peptide)

    # Get heavy atoms only
    peptide_heavy = get_heavy_atoms(peptide)
    mhc_heavy = get_heavy_atoms(mhc)

    n_frames = len(u.trajectory)
    n_sampled = len(range(0, n_frames, FRAME_STRIDE))
    print(f"  Total frames: {n_frames}")
    print(f"  Sampled frames (every {FRAME_STRIDE}th): {n_sampled}")
    print(f"  Peptide heavy atoms: {len(peptide_heavy)}")
    print(f"  MHC heavy atoms: {len(mhc_heavy)}")

    # Storage for sampled frames
    all_LL_graphs = []
    all_LP_graphs = []
    all_features = []

    # Process sampled frames
    print(f"  Extracting graphs...")
    for frame_idx, ts in enumerate(u.trajectory):
        # Skip frames according to stride
        if frame_idx % FRAME_STRIDE != 0:
            continue
        # Build LL graph (peptide-peptide)
        G_LL = build_LL_graph(peptide_heavy, frame_idx)

        # Build LP graph (peptide-MHC)
        G_LP = build_LP_graph(peptide_heavy, mhc_heavy, frame_idx)

        # Convert to features
        features = graph_to_features(G_LL, G_LP)
        features['frame'] = frame_idx
        features['time_ns'] = ts.time / 1000.0  # Convert ps to ns

        all_LL_graphs.append(G_LL)
        all_LP_graphs.append(G_LP)
        all_features.append(features)

    # Aggregate statistics
    LL_edges = [G.number_of_edges() for G in all_LL_graphs]
    LP_edges = [G.number_of_edges() for G in all_LP_graphs]

    summary = {
        'peptide': peptide_name,
        'n_frames_total': n_frames,
        'n_frames_sampled': n_sampled,
        'frame_stride': FRAME_STRIDE,
        'n_peptide_heavy_atoms': len(peptide_heavy),
        'n_mhc_heavy_atoms': len(mhc_heavy),
        'LL_edges_mean': np.mean(LL_edges),
        'LL_edges_std': np.std(LL_edges),
        'LL_edges_min': np.min(LL_edges),
        'LL_edges_max': np.max(LL_edges),
        'LP_edges_mean': np.mean(LP_edges),
        'LP_edges_std': np.std(LP_edges),
        'LP_edges_min': np.min(LP_edges),
        'LP_edges_max': np.max(LP_edges),
    }

    print(f"\n  Graph Statistics:")
    print(f"    LL edges: {summary['LL_edges_mean']:.1f} ± {summary['LL_edges_std']:.1f}")
    print(f"    LP edges: {summary['LP_edges_mean']:.1f} ± {summary['LP_edges_std']:.1f}")

    # Save graph data
    output_file = os.path.join(output_dir, f'{peptide_name}_graphs.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'peptide': peptide_name,
            'LL_graphs': all_LL_graphs,
            'LP_graphs': all_LP_graphs,
            'features': all_features,
            'summary': summary
        }, f)

    print(f"  ✓ Saved to: {output_file}")

    return summary


def process_all_peptides():
    """
    Process all completed MD simulations and extract graph features
    """
    project_root = Path(__file__).parent.parent.parent
    systems_dir = project_root / 'md_data' / 'systems'
    analysis_dir = project_root / 'md_data' / 'analysis'
    graphs_dir = analysis_dir / 'graph_features'
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Graph Feature Extraction for GCN")
    print("  Following Weber et al. 2024 (bbad504)")
    print("="*60)
    print("\nParameters:")
    print(f"  - Global cutoff: {GLOBAL_CUTOFF} Å")
    print(f"  - Heavy atoms only: Yes")
    print(f"  - Exclude covalent bonds: Yes (within {COVALENT_CUTOFF} Å)")
    print(f"  - Distance weighting: Exponential (exp(-d))")
    print(f"  - Time window: 30-100 ns")
    print(f"  - Frame sampling: Every {FRAME_STRIDE}th frame (~{70*FRAME_STRIDE/7000:.1f} ns interval)")
    print("\nGraph types:")
    print("  - LL (Ligand-Ligand): Intramolecular peptide contacts")
    print("  - LP (Ligand-Protein): Intermolecular peptide-MHC contacts")
    print()

    # Find all peptides with processed trajectories
    peptides = []
    for peptide_dir in sorted(systems_dir.iterdir()):
        if peptide_dir.is_dir():
            xtc_file = peptide_dir / 'md_30-100ns.xtc'
            if xtc_file.exists():
                peptides.append(peptide_dir.name)

    print(f"Found {len(peptides)} peptides with processed trajectories\n")

    if len(peptides) == 0:
        print("ERROR: No processed trajectories found!")
        print("Run: bash 02_scripts/analysis/process_all_trajectories.sh")
        return

    # Process each peptide
    all_summaries = []
    for peptide in peptides:
        system_dir = systems_dir / peptide
        try:
            summary = extract_graphs_for_peptide(peptide, system_dir, graphs_dir)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"\n  ✗ Error processing {peptide}:")
            print(f"     {e}")
            import traceback
            traceback.print_exc()

    # Save overall summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_file = graphs_dir / 'graph_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        print("\n" + "="*60)
        print(f"✓ Processed {len(all_summaries)}/{len(peptides)} peptides")
        print(f"\nOutput directory: {graphs_dir}")
        print(f"Summary file: {summary_file}")
        print("="*60)

        # Print overall statistics
        print("\nOverall Statistics:")
        print(f"  Average LL edges per frame: {summary_df['LL_edges_mean'].mean():.1f}")
        print(f"  Average LP edges per frame: {summary_df['LP_edges_mean'].mean():.1f}")
        print(f"  Peptide heavy atoms: {summary_df['n_peptide_heavy_atoms'].mean():.1f}")
        print(f"  MHC heavy atoms: {summary_df['n_mhc_heavy_atoms'].mean():.1f}")

    return all_summaries


if __name__ == '__main__':
    # Check dependencies
    try:
        import MDAnalysis
        import networkx
        import tqdm
        print(f"MDAnalysis version: {MDAnalysis.__version__}")
        print(f"NetworkX version: {networkx.__version__}")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install MDAnalysis networkx tqdm")
        exit(1)

    # Run processing
    process_all_peptides()
