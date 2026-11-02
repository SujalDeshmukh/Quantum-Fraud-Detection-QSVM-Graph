"""
Tests for Quantum-Fraud-Detection-QSVM-Graph pipeline components.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import ZZFeatureMap

from src.build_features import build_graph_features
from src.prepare_qsvm_input import prepare_qsvm_input
from src.run_qsvm import run_qsvm_pipeline, build_quantum_kernel
from src.benchmark import run_benchmark


@pytest.fixture
def mock_graph_data(tmp_path):
    nodes_csv = tmp_path / "mock_nodes.csv"
    edges_csv = tmp_path / "mock_edges.csv"

    nodes_df = pd.DataFrame({
        "node_id": [f"N{i}" for i in range(10)],
        "feat1": np.random.randn(10),
        "feat2": np.random.randn(10),
        "IsFraud": [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    })
    nodes_df.to_csv(nodes_csv, index=False)

    edges_df = pd.DataFrame({
        "source": ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"],
        "target": ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"],
        "weight": [1.0] * 8,
    })
    edges_df.to_csv(edges_csv, index=False)
    return nodes_csv, edges_csv, tmp_path


def test_build_features(mock_graph_data):
    nodes_csv, edges_csv, tmp_path = mock_graph_data
    out_csv = tmp_path / "features.csv"

    df = build_graph_features(
        nodes_path=nodes_csv,
        edges_path=edges_csv,
        out_path=out_csv,
        emb_dim=8,
        walk_length=5,
        num_walks=2,
        workers=1,
    )
    assert out_csv.exists()
    assert "node_id" in df.columns
    assert "deg" in df.columns
    assert "pagerank" in df.columns
    assert "emb_0" in df.columns
    assert "label" in df.columns
    assert len(df) == 10


def test_prepare_qsvm_input(mock_graph_data):
    nodes_csv, edges_csv, tmp_path = mock_graph_data
    feat_csv = tmp_path / "features.csv"
    qsvm_in_csv = tmp_path / "qsvm_input.csv"

    build_graph_features(
        nodes_path=nodes_csv,
        edges_path=edges_csv,
        out_path=feat_csv,
        emb_dim=8,
        walk_length=5,
        num_walks=2,
        workers=1,
    )

    df_qsvm = prepare_qsvm_input(
        features_path=feat_csv,
        out_path=qsvm_in_csv,
        n_components=4,
    )
    assert qsvm_in_csv.exists()
    assert "label" in df_qsvm.columns
    assert "pca_0" in df_qsvm.columns
    assert "pca_3" in df_qsvm.columns
    assert len(df_qsvm) == 10


def test_quantum_kernel_and_qsvm(mock_graph_data):
    nodes_csv, edges_csv, tmp_path = mock_graph_data
    feat_csv = tmp_path / "features.csv"
    qsvm_in_csv = tmp_path / "qsvm_input.csv"
    res_csv = tmp_path / "results.csv"
    kernel_npy = tmp_path / "kernel.npy"
    circuit_png = tmp_path / "circuit.png"

    build_graph_features(
        nodes_path=nodes_csv,
        edges_path=edges_csv,
        out_path=feat_csv,
        emb_dim=8,
        walk_length=5,
        num_walks=2,
        workers=1,
    )
    prepare_qsvm_input(
        features_path=feat_csv,
        out_path=qsvm_in_csv,
        n_components=2,
    )

    results_df, K = run_qsvm_pipeline(
        input_path=qsvm_in_csv,
        out_results_path=res_csv,
        save_kernel_path=kernel_npy,
        save_circuit_path=circuit_png,
        sample_n=4,
    )

    assert res_csv.exists()
    assert kernel_npy.exists()
    assert K.shape == (4, 4)
    assert "qsvm_score" in results_df.columns
    assert "prediction" in results_df.columns


def test_benchmark_runner():
    bench_df = run_benchmark(
        qsvm_input_csv="data/processed/qsvm_input_8q.csv",
        kernel_npy="data/results/qkernel_matrix_8q.npy",
        sample_n=18,
    )
    assert "Model" in bench_df.columns
    assert "Accuracy" in bench_df.columns
    assert len(bench_df) >= 4
