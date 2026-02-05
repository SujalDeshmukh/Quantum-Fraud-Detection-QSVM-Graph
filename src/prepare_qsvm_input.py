#!/usr/bin/env python3
"""
STEP 3: Prepare low-dimensional PCA input for QSVM (with label support).
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True, 
                    help="Input CSV with node features (must include node_id and label)")
parser.add_argument("--out", required=True, 
                    help="Output QSVM input CSV path")
parser.add_argument("--n-components", type=int, default=8, 
                    help="Number of PCA components (qubits)")
args = parser.parse_args()

# ---------- Load data ----------
df = pd.read_csv(args.features)

# Detect label column (case-insensitive)
label_col = None
for c in df.columns:
    if c.lower() in ["label", "isfraud", "target", "class"]:
        label_col = c
        break

if label_col is None:
    raise ValueError("No label column found (expected 'label', 'IsFraud', 'target', or 'class').")

# Separate features and labels
X = df.drop(columns=["node_id", label_col]).values
y = df[label_col].values

# Standardize + PCA
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=args.n_components, random_state=42)
Xpca = pca.fit_transform(X)

# ---------- Build output ----------
out = pd.DataFrame({
    "node_id": df.node_id,
    "label": y
})
for i in range(args.n_components):
    out[f"pca_{i}"] = Xpca[:, i]

out.to_csv(args.out, index=False)
print(f"[OK] Saved QSVM input → {args.out}, shape={out.shape}")