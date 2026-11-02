#!/usr/bin/env python3
"""
STEP 2: Prepare low-dimensional PCA input for QSVM (with robust label support).
Reduces high-dimensional graph embeddings down to N qubits for quantum feature mapping.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_qsvm_input")


def prepare_qsvm_input(
    features_path: str | Path,
    out_path: str | Path,
    n_components: int = 8,
    random_state: int = 42,
    raw_nodes_path: str | Path | None = None,
) -> pd.DataFrame:
    """Prepares scaled PCA features suitable for encoding into an n-qubit quantum state."""
    logger.info("Loading feature matrix from %s ...", features_path)
    df = pd.read_csv(features_path)

    # Detect label column
    label_col = None
    for c in df.columns:
        if c.lower() in ["label", "isfraud", "target", "class"]:
            label_col = c
            break

    # If not present in features_path, try to retrieve from raw_nodes_path if provided
    if label_col is None and raw_nodes_path and Path(raw_nodes_path).exists():
        logger.info("Reading label column from raw nodes at %s ...", raw_nodes_path)
        nodes_df = pd.read_csv(raw_nodes_path)
        for c in nodes_df.columns:
            if c.lower() in ["isfraud", "label", "target", "class"]:
                nodes_df["node_id"] = nodes_df["node_id"].astype(str)
                df["node_id"] = df["node_id"].astype(str)
                label_map = dict(zip(nodes_df["node_id"], nodes_df[c]))
                df["label"] = df["node_id"].map(label_map).fillna(0).astype(int)
                label_col = "label"
                break

    if label_col is None:
        raise ValueError("No label column found in features or raw nodes. Expected 'label' or 'IsFraud'.")

    logger.info("Using target label column: '%s'", label_col)

    # Isolate feature columns (exclude node_id and label)
    feature_cols = [c for c in df.columns if c not in ["node_id", label_col]]
    X = df[feature_cols].values
    y = df[label_col].values

    # Standardize + PCA reduction
    logger.info("Standardizing %d features and reducing to %d PCA components (qubits)...", X.shape[1], n_components)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = float(np.sum(pca.explained_variance_ratio_) * 100)
    logger.info("Top %d PCA components capture %.2f%% of feature variance.", n_components, explained_var)

    # Build output dataframe
    out_df = pd.DataFrame({
        "node_id": df["node_id"].values,
        "label": y,
    })
    for i in range(n_components):
        out_df[f"pca_{i}"] = X_pca[:, i]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    logger.info("[OK] Saved QSVM input to %s, shape=%s", out_file, out_df.shape)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Prepare low-dimensional PCA input for QSVM.")
    parser.add_argument("--features", required=True, help="Input CSV with node features")
    parser.add_argument("--out", required=True, help="Output QSVM input CSV path")
    parser.add_argument("--n-components", type=int, default=8, help="Number of PCA components (qubits)")
    parser.add_argument("--raw-nodes", default=None, help="Optional raw nodes file to retrieve labels if missing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PCA")
    args = parser.parse_args()

    prepare_qsvm_input(
        features_path=args.features,
        out_path=args.out,
        n_components=args.n_components,
        random_state=args.seed,
        raw_nodes_path=args.raw_nodes,
    )


if __name__ == "__main__":
    main()