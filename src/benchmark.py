#!/usr/bin/env python3
"""
Benchmark comparison script between Quantum SVM and Classical Classifiers
(Classical RBF SVM, Logistic Regression, Random Forest) on the fraud dataset.
"""

from __future__ import annotations
import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(
    qsvm_input_csv: str = "data/processed/qsvm_input_8q.csv",
    kernel_npy: str = "data/results/qkernel_matrix_8q.npy",
    sample_n: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    """Compares QSVM performance against classical ML benchmarks."""
    logger.info("Loading dataset from %s ...", qsvm_input_csv)
    df = pd.read_csv(qsvm_input_csv)
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    X_raw = df[pca_cols].values
    y_raw = df["label"].values.astype(int)

    # Standardize
    X_scaled = StandardScaler().fit_transform(X_raw)

    # Balance classes
    classes, counts = np.unique(y_raw, return_counts=True)
    min_count = min(counts)

    X_bal, y_bal = [], []
    for cls in classes:
        mask = (y_raw == cls)
        Xr, yr = resample(X_scaled[mask], y_raw[mask], replace=True, n_samples=min_count, random_state=seed)
        X_bal.append(Xr)
        y_bal.append(yr)

    X_bal = np.vstack(X_bal)
    y_bal = np.hstack(y_bal)

    np.random.seed(seed)
    n_take = min(sample_n, len(y_bal))
    idx = np.random.choice(len(y_bal), n_take, replace=False)
    X_eval = X_bal[idx]
    y_eval = y_bal[idx]

    models = {
        "Classical RBF SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "Classical Linear SVM": SVC(kernel="linear", probability=True, random_state=seed),
        "Logistic Regression": LogisticRegression(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=seed),
    }

    records = []

    # 1. Classical Baselines
    for name, model in models.items():
        model.fit(X_eval, y_eval)
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else preds

        acc = accuracy_score(y_eval, preds)
        prec = precision_score(y_eval, preds, zero_division=0)
        rec = recall_score(y_eval, preds, zero_division=0)
        f1 = f1_score(y_eval, preds, zero_division=0)
        auc = roc_auc_score(y_eval, probs) if len(np.unique(y_eval)) > 1 else 0.5

        records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
        })

    # 2. QSVM evaluation
    if Path(kernel_npy).exists():
        K = np.load(kernel_npy)
        if K.shape[0] == len(y_eval):
            qsvm = SVC(kernel="precomputed", probability=True, random_state=seed)
            qsvm.fit(K, y_eval)
            qpreds = qsvm.predict(K)
            qprobs = qsvm.predict_proba(K)[:, 1]

            records.append({
                "Model": "Quantum SVM (Fidelity Kernel)",
                "Accuracy": accuracy_score(y_eval, qpreds),
                "Precision": precision_score(y_eval, qpreds, zero_division=0),
                "Recall": recall_score(y_eval, qpreds, zero_division=0),
                "F1-Score": f1_score(y_eval, qpreds, zero_division=0),
                "ROC-AUC": roc_auc_score(y_eval, qprobs),
            })

    bench_df = pd.DataFrame(records)
    print("\n" + "=" * 65)
    print("      Model Benchmark Comparison on Financial Fraud Dataset")
    print("=" * 65)
    print(bench_df.to_string(index=False))
    print("=" * 65 + "\n")
    return bench_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/qsvm_input_8q.csv")
    parser.add_argument("--kernel", default="data/results/qkernel_matrix_8q.npy")
    parser.add_argument("--sample-n", type=int, default=18)
    args = parser.parse_args()

    run_benchmark(args.input, args.kernel, sample_n=args.sample_n)
