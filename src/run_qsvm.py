#!/usr/bin/env python3
"""
QSVM kernel computation with automatic compatibility for old and new Qiskit versions.
Balances data, computes quantum kernel, and saves results.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from qiskit import __version__ as qiskit_version
from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer

# ----------------- Parse args -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--sample-n", type=int, default=150)
parser.add_argument("--save-kernel", default=None)
args = parser.parse_args()

print(f"[INFO] Using Qiskit version: {qiskit_version}")

# ----------------- Load data -----------------
df = pd.read_csv(args.input)
if "label" not in df.columns:
    raise ValueError("Input CSV must contain a 'label' column.")

X = df[[c for c in df.columns if c.startswith("pca_")]].values
y = df["label"].values
X = StandardScaler().fit_transform(X)

# ----------------- Balance classes -----------------
unique, counts = np.unique(y, return_counts=True)
print("[INFO] Original class distribution:", dict(zip(unique, counts)))

min_class = min(counts)
X_bal, y_bal = [], []
for label in np.unique(y):
    Xc = X[y == label]
    yc = np.ones(len(Xc)) * label
    Xr, yr = resample(Xc, yc, replace=True, n_samples=min_class, random_state=42)
    X_bal.append(Xr)
    y_bal.append(yr)

X_bal = np.vstack(X_bal)
y_bal = np.hstack(y_bal)
print(f"[INFO] Balanced dataset → {len(y_bal)} samples")

# ----------------- Random sampling -----------------
N = len(X_bal)
np.random.seed(42)
idxs = np.random.choice(N, min(args.sample_n, N), replace=False)
Xsample = X_bal[idxs]
ysample = y_bal[idxs]
print(f"[INFO] Using {len(Xsample)} samples → {Xsample.shape[1]} features (qubits)")

# ----------------- Feature map -----------------
n_qubits = Xsample.shape[1]
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")

circuit_drawer(feature_map.decompose(), output="mpl",
               filename="results/zzfeaturemap_8qubit.png", fold=120)
print("[OK] Saved 8-qubit circuit diagram → results/zzfeaturemap_8qubit.png")

# ----------------- Version-safe imports -----------------
try:
    # Newer Qiskit (>=1.2)
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    print("[INFO] Using modern StatevectorSampler + ComputeUncompute interface")

except Exception as e:
    print(f"[WARN] Falling back to legacy Qiskit interface ({e})")
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.primitives import Sampler
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    print("[INFO] Using legacy Sampler + ComputeUncompute interface")

# ----------------- Compute Kernel -----------------
print("[INFO] Computing Quantum Kernel…")
K = quantum_kernel.evaluate(x_vec=Xsample)
print("[OK] Kernel computed, shape:", K.shape)

# ----------------- Save outputs -----------------
out_df = pd.DataFrame({"qsvm_score": K.mean(axis=1), "label": ysample})
out_df.to_csv(args.out, index=False)
print(f"[OK] Saved QSVM scores → {args.out}")

if args.save_kernel:
    
    import os

    os.makedirs(os.path.dirname(args.save_kernel), exist_ok=True)
    np.save(args.save_kernel, K)

    print(f"[OK] Saved kernel matrix → {args.save_kernel}")
