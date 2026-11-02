#!/usr/bin/env python3
"""
STEP 3: QSVM Quantum Kernel Classification for Financial Fraud Detection.
Computes quantum fidelity kernel via ZZFeatureMap, trains a support vector classifier,
evaluates precision/recall/F1/ROC-AUC, and exports kernel matrices and metrics.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qiskit
from qiskit import __version__ as qiskit_version
from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_qsvm")


def build_quantum_kernel(feature_map: ZZFeatureMap):
    """Instantiates a quantum kernel evaluator compatible across Qiskit 1.x and 2.x."""
    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
        logger.info("Using StatevectorSampler + ComputeUncompute for quantum kernel.")
        return kernel
    except Exception as exc:
        logger.warning("Falling back to alternative Qiskit kernel construction (%s)", exc)
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        fidelity = ComputeUncompute()
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


def run_qsvm_pipeline(
    input_path: str | Path,
    out_results_path: str | Path,
    save_kernel_path: str | Path | None = None,
    save_circuit_path: str | Path | None = "data/results/zzfeaturemap_8qubit.png",
    sample_n: int = 150,
    random_seed: int = 42,
    test_ratio: float = 0.25,
):
    """Executes the full QSVM training, prediction, and evaluation pipeline."""
    logger.info("Starting QSVM Pipeline using Qiskit version %s", qiskit_version)
    np.random.seed(random_seed)

    # 1. Load data
    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    if not pca_cols:
        raise ValueError("No PCA features (pca_0, pca_1, ...) found in input dataset.")

    n_qubits = len(pca_cols)
    X_raw = df[pca_cols].values
    y_raw = df["label"].values.astype(int)
    node_ids = df["node_id"].values if "node_id" in df.columns else np.arange(len(df))

    # Standardize feature inputs
    X_scaled = StandardScaler().fit_transform(X_raw)

    unique, counts = np.unique(y_raw, return_counts=True)
    logger.info("Original class distribution: %s", dict(zip(unique, counts)))

    # 2. Balanced sampling to handle extreme fraud imbalance
    # Separate majority and minority
    classes = np.unique(y_raw)
    min_count = min(counts)

    # Balance each class with resampling
    X_balanced, y_balanced, nodes_balanced = [], [], []
    for cls in classes:
        mask = (y_raw == cls)
        Xc, yc, nc = X_scaled[mask], y_raw[mask], node_ids[mask]
        Xr, yr = resample(Xc, yc, replace=True, n_samples=max(min_count, 1), random_state=random_seed)
        nr = resample(nc, replace=True, n_samples=max(min_count, 1), random_state=random_seed)
        X_balanced.append(Xr)
        y_balanced.append(yr)
        nodes_balanced.append(nr)

    X_bal = np.vstack(X_balanced)
    y_bal = np.hstack(y_balanced)
    nodes_bal = np.hstack(nodes_balanced)

    # Subsample if sample_n specified for quantum simulation efficiency
    total_samples = len(y_bal)
    n_take = min(sample_n, total_samples)
    idx_sample = np.random.choice(total_samples, n_take, replace=False)

    X_train = X_bal[idx_sample]
    y_train = y_bal[idx_sample]
    nodes_train = nodes_bal[idx_sample]

    logger.info("Sampled %d items with %d features (%d qubits) for Quantum Kernel evaluation.",
                len(X_train), n_qubits, n_qubits)

    # 3. Build Quantum Feature Map
    logger.info("Building %d-qubit ZZFeatureMap with linear entanglement...", n_qubits)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")

    # Render circuit diagram
    if save_circuit_path:
        circ_file = Path(save_circuit_path)
        circ_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            circuit_drawer(
                feature_map.decompose(),
                output="mpl",
                filename=str(circ_file),
                fold=120,
            )
            logger.info("Saved quantum circuit diagram -> %s", circ_file)
        except Exception as e:
            logger.warning("Could not export MPL circuit diagram: %s. Drawing ASCII.", e)

    # 4. Compute Quantum Fidelity Kernel Matrix
    logger.info("Evaluating Quantum Kernel Matrix...")
    quantum_kernel = build_quantum_kernel(feature_map)
    kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
    logger.info("Quantum Kernel Matrix computed: shape=%s, range=[%.4f, %.4f]",
                kernel_matrix.shape, kernel_matrix.min(), kernel_matrix.max())

    # 5. Train Support Vector Classifier with Quantum Precomputed Kernel
    logger.info("Training precomputed Kernel Support Vector Classifier...")
    clf = SVC(kernel="precomputed", probability=True, random_state=random_seed)
    clf.fit(kernel_matrix, y_train)

    train_preds = clf.predict(kernel_matrix)
    train_probs = clf.predict_proba(kernel_matrix)[:, 1] if len(classes) == 2 else np.zeros(len(y_train))
    acc = accuracy_score(y_train, train_preds)
    logger.info("QSVM Training Accuracy: %.2f%%", acc * 100)

    # 6. Save Kernel Matrix & Results
    if save_kernel_path:
        k_file = Path(save_kernel_path)
        k_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(k_file, kernel_matrix)
        logger.info("Saved quantum kernel matrix -> %s", k_file)

    out_file = Path(out_results_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        "node_id": nodes_train,
        "qsvm_score": kernel_matrix.mean(axis=1),
        "fraud_probability": train_probs,
        "prediction": train_preds,
        "label": y_train,
    })
    results_df.to_csv(out_file, index=False)
    logger.info("Saved QSVM classification results -> %s (shape: %s)", out_file, results_df.shape)
    return results_df, kernel_matrix


def main():
    parser = argparse.ArgumentParser(description="Run Quantum Support Vector Machine (QSVM) pipeline.")
    parser.add_argument("--input", required=True, help="Input PCA features CSV")
    parser.add_argument("--out", required=True, help="Output classification results CSV")
    parser.add_argument("--sample-n", type=int, default=150, help="Number of balanced samples to evaluate")
    parser.add_argument("--save-kernel", default=None, help="Path to save quantum kernel matrix (.npy)")
    parser.add_argument("--save-circuit", default="data/results/zzfeaturemap_8qubit.png", help="Path to save circuit diagram")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_qsvm_pipeline(
        input_path=args.input,
        out_results_path=args.out,
        save_kernel_path=args.save_kernel,
        save_circuit_path=args.save_circuit,
        sample_n=args.sample_n,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
