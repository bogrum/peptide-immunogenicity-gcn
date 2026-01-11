# Trajectory Analysis Scripts

Following **Weber et al. 2024** (*Briefings in Bioinformatics*) methodology for MHC-peptide immunogenicity prediction.

## Overview

These scripts process MD trajectories to extract features for:
1. **Unsupervised learning** (Markov State Models)
2. **Supervised learning** (Graph Convolutional Networks)

## Methodology (from Weber et al. 2024)

### Time Window
- **Use 30-200 ns** (exclude first 30 ns equilibration)
- 200 ns chosen as "local equilibration timescale" where system forgets starting structure

### Features Extracted

1. **SASA (Solvent Accessible Surface Area)**
   - Per-residue calculation
   - Separated into hydrophobic vs hydrophilic
   - Key for immunogenicity (hydrophobic exposure correlates with immunogenicity)

2. **RMSD (Root Mean Square Deviation)**
   - Heavy atom RMSD from starting structure
   - **No fitting/alignment** (raw displacement)
   - Immunogenic peptides show higher RMSD on average

3. **Backbone Dihedral Angles (φ, ψ)**
   - Used for tICA dimensionality reduction
   - Input for Markov State Model building

4. **Anchor Dynamics**
   - P2 and P9 anchor residue distances to MHC pockets
   - Immunogenic peptides show more flexible anchors (Figure 6)

5. **RMSF (Root Mean Square Fluctuation)**
   - Per-residue fluctuations
   - Immunogenic peptides more dynamic at all positions

6. **Contact Maps**
   - 8 Å cutoff
   - Intramolecular (peptide-peptide)
   - Intermolecular (peptide-MHC)

### Important Notes
- **Keep explicit solvent** (needed for proper SASA calculation)
- **No trajectory alignment** (just raw measurements)
- **No water removal** (hydrophobic effect requires explicit solvent)

## Usage

### Step 1: Process All Trajectories

```bash
cd /home/emre/workspace/seminar_study/analysis
bash 02_scripts/analysis/process_all_trajectories.sh
```

This will:
- Remove PBC artifacts (make molecules whole)
- Extract 30-200 ns time window
- Calculate RMSD, SASA, RMSF, dihedral angles
- Generate summary statistics

**Output:** `md_data/analysis/PEPTIDE_NAME/`
- `rmsd.xvg` - RMSD over time
- `sasa_total.xvg` - Total SASA
- `sasa_per_residue.xvg` - Per-residue SASA
- `rmsf.xvg` - Per-residue fluctuations
- `ramachandran.xvg` - Backbone dihedrals
- `summary.txt` - Key statistics

### Step 2: Extract ML Features (Optional)

```bash
python 02_scripts/analysis/extract_features_for_ml.py
```

Requires: `pip install MDAnalysis`

This extracts features in NumPy format for machine learning.

### Step 3: Process Single Peptide

```bash
bash 02_scripts/analysis/process_trajectory_single.sh PEPTIDE_NAME
```

## Expected Output

After processing all 30 peptides:

```
md_data/analysis/
├── all_peptides_summary.csv          # Combined statistics
├── AIYDTMQYV/
│   ├── rmsd.xvg
│   ├── sasa_total.xvg
│   ├── sasa_per_residue.xvg
│   ├── rmsf.xvg
│   ├── ramachandran.xvg
│   └── summary.txt
├── AMNDILAQV/
│   └── ...
└── ... (all 30 peptides)
```

## Key Findings from Paper

1. **Immunogenic peptides have:**
   - Higher hydrophobic SASA (especially at P4, P5)
   - Larger RMSD from starting structure
   - More flexible anchor positions (P2, P9)
   - Greater overall fluctuations (RMSF)

2. **Anchor dynamics critical:**
   - P9 anchor dynamics especially important
   - ~1 Å difference in mean distance between classes
   - No correlation with binding affinity (Figure 6F)

3. **200 ns equilibration:**
   - System forgets starting template structure
   - Sufficient for local conformational sampling
   - Not sufficient for rare events (they used Markov models to extend to ms scale)

## Next Steps for Analysis

1. **Unsupervised Learning:**
   - Build Markov State Models from dihedral angles
   - Use tICA for dimensionality reduction
   - Cluster into discrete states (~3000 states)
   - Analyze slow relaxation timescales

2. **Supervised Learning:**
   - Build molecular graphs (8 Å cutoff)
   - Train Graph Convolutional Network
   - Predict immunogenicity from structure/dynamics

3. **Feature Analysis:**
   - Compare immunogenic vs non-immunogenic peptides
   - Calculate hydrophobic vs hydrophilic SASA
   - Analyze anchor position dynamics
   - Identify presentation modes

## References

Weber JK et al. (2024) "Unsupervised and supervised AI on molecular dynamics
simulations reveals complex characteristics of HLA-A2-peptide immunogenicity"
*Briefings in Bioinformatics*, 25(1), 1-12.
