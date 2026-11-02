#!/usr/bin/env python3
"""
Full Pipeline Runner for Quantum Fraud Detection (Graph + Node2Vec + QSVM).
Executes feature building, PCA dimensionality reduction, and quantum kernel classification end-to-end.
"""

from __future__ import annotations
import argparse
import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.build_features import build_graph_features
from src.prepare_qsvm_input import prepare_qsvm_input
from src.run_qsvm import run_qsvm_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline_runner")


def run_pipeline(
    nodes_csv: str = "data/raw/graph_nodes_7000.csv",
    edges_csv: str = "data/raw/graph_edges_7000.csv",
    graph_features_csv: str = "data/processed/graph_features.csv",
    qsvm_input_csv: str = "data/processed/qsvm_input_8q.csv",
    results_csv: str = "data/results/qsvm_results_8q.csv",
    kernel_npy: str = "data/results/qkernel_matrix_8q.npy",
    circuit_png: str = "data/results/zzfeaturemap_8qubit.png",
    n_qubits: int = 8,
    sample_n: int = 150,
    emb_dim: int = 64,
    skip_graph: bool = False,
):
    """Executes the complete three-stage quantum fraud detection pipeline."""
    start_time = time.time()
    logger.info("==================================================")
    logger.info("   Quantum Fraud Detection Pipeline (QSVM + Graph)")
    logger.info("==================================================")

    # Stage 1: Build graph structural and Node2Vec features
    if skip_graph and Path(graph_features_csv).exists():
        logger.info("[STAGE 1/3] Skipping graph feature extraction (using cached %s)", graph_features_csv)
    else:
        logger.info("[STAGE 1/3] Building Graph Structural & Node2Vec Features...")
        build_graph_features(
            nodes_path=nodes_csv,
            edges_path=edges_csv,
            out_path=graph_features_csv,
            emb_dim=emb_dim,
            workers=2,
        )

    # Stage 2: Dimensionality reduction to N qubits via PCA
    logger.info("[STAGE 2/3] Reducing Features to %d PCA Components (Qubits)...", n_qubits)
    prepare_qsvm_input(
        features_path=graph_features_csv,
        out_path=qsvm_input_csv,
        n_components=n_qubits,
        raw_nodes_path=nodes_csv,
    )

    # Stage 3: Quantum Support Vector Machine Classification
    logger.info("[STAGE 3/3] Evaluating Quantum Kernel & Training QSVM...")
    results_df, K = run_qsvm_pipeline(
        input_path=qsvm_input_csv,
        out_results_path=results_csv,
        save_kernel_path=kernel_npy,
        save_circuit_path=circuit_png,
        sample_n=sample_n,
    )

    elapsed = time.time() - start_time
    logger.info("==================================================")
    logger.info("Pipeline completed successfully in %.2f seconds!", elapsed)
    logger.info("Outputs:")
    logger.info("  - Graph Features: %s", graph_features_csv)
    logger.info("  - QSVM PCA Input: %s", qsvm_input_csv)
    logger.info("  - Classification: %s", results_csv)
    logger.info("  - Quantum Kernel: %s (shape: %s)", kernel_npy, K.shape)
    logger.info("  - Circuit Diagram: %s", circuit_png)
    logger.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Quantum Fraud Detection Runner")
    parser.add_argument("--nodes", default="data/raw/graph_nodes_7000.csv", help="Path to nodes CSV")
    parser.add_argument("--edges", default="data/raw/graph_edges_7000.csv", help="Path to edges CSV")
    parser.add_argument("--features", default="data/processed/graph_features.csv", help="Path to graph features CSV")
    parser.add_argument("--qsvm-input", default="data/processed/qsvm_input_8q.csv", help="Path to PCA input CSV")
    parser.add_argument("--out", default="data/results/qsvm_results_8q.csv", help="Output results CSV")
    parser.add_argument("--kernel", default="data/results/qkernel_matrix_8q.npy", help="Output kernel matrix NPY")
    parser.add_argument("--circuit", default="data/results/zzfeaturemap_8qubit.png", help="Output circuit diagram PNG")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits (PCA dimensions)")
    parser.add_argument("--sample-n", type=int, default=150, help="Number of samples for quantum kernel evaluation")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Node2Vec if graph_features.csv exists")
    args = parser.parse_args()

    run_pipeline(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        graph_features_csv=args.features,
        qsvm_input_csv=args.qsvm_input,
        results_csv=args.out,
        kernel_npy=args.kernel,
        circuit_png=args.circuit,
        n_qubits=args.n_qubits,
        sample_n=args.sample_n,
        skip_graph=args.skip_graph,
    )


if __name__ == "__main__":
    main()
