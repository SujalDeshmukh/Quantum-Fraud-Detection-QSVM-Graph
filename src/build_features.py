#!/usr/bin/env python3
"""
STEP 1: Build graph-based features (Degree, PageRank, Node2Vec embeddings)
and preserve ground-truth node labels (IsFraud) for the QSVM pipeline.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_features")


def build_graph_features(
    nodes_path: str | Path,
    edges_path: str | Path,
    out_path: str | Path,
    emb_dim: int = 64,
    walk_length: int = 20,
    num_walks: int = 10,
    workers: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Build graph structural and embedding features for nodes."""
    logger.info("Loading nodes from %s and edges from %s ...", nodes_path, edges_path)
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Normalize column names
    edges_df.columns = [c.lower() for c in edges_df.columns]
    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        raise ValueError("Edges CSV must have 'source' and 'target' columns.")

    # Convert node_id to string for consistent graph indexing
    nodes_df["node_id"] = nodes_df["node_id"].astype(str)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)

    # Build graph
    G = nx.from_pandas_edgelist(
        edges_df, "source", "target", edge_attr="weight" if "weight" in edges_df.columns else None,
        create_using=nx.Graph()
    )

    # Ensure all nodes exist in graph
    for nid in nodes_df["node_id"]:
        if nid not in G:
            G.add_node(nid)

    nodes_sorted = sorted(G.nodes())
    logger.info("Graph constructed: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Structural graph features
    logger.info("Computing degree and PageRank centrality...")
    degrees = np.array([float(G.degree(n)) for n in nodes_sorted])
    pagerank_dict = nx.pagerank(G, weight="weight" if "weight" in edges_df.columns else None)
    pagerank = np.array([float(pagerank_dict.get(n, 0.0)) for n in nodes_sorted])

    # Node2Vec walk and embedding generation
    logger.info("Training Node2Vec embeddings (dim=%d, walks=%d, len=%d)...", emb_dim, num_walks, walk_length)
    n2v = Node2Vec(
        G,
        dimensions=emb_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    emb_matrix = np.vstack([
        model.wv[str(n)] if str(n) in model.wv else np.zeros(emb_dim)
        for n in nodes_sorted
    ])

    # Combine degree, pagerank, and node2vec embeddings
    feature_matrix = np.hstack([degrees[:, None], pagerank[:, None], emb_matrix])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    feature_cols = ["deg", "pagerank"] + emb_cols
    df_features = pd.DataFrame(scaled_features, columns=feature_cols)
    df_features.insert(0, "node_id", nodes_sorted)

    # Retain ground truth label if present in nodes.csv
    label_col = None
    for c in nodes_df.columns:
        if c.lower() in ["isfraud", "label", "target", "class"]:
            label_col = c
            break

    if label_col:
        labels_map = dict(zip(nodes_df["node_id"], nodes_df[label_col]))
        df_features["label"] = [labels_map.get(nid, 0) for nid in nodes_sorted]
        logger.info("Preserved node target labels '%s' in feature output.", label_col)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_file, index=False)
    logger.info("Successfully saved graph features to %s (shape: %s)", out_file, df_features.shape)
    return df_features


def main():
    parser = argparse.ArgumentParser(description="Build graph and Node2Vec features for QSVM fraud detection.")
    parser.add_argument("--nodes", required=True, help="Path to nodes.csv (must contain node_id)")
    parser.add_argument("--edges", required=True, help="Path to edges.csv (source, target)")
    parser.add_argument("--out", required=True, help="Output CSV path for graph features")
    parser.add_argument("--emb-dim", type=int, default=64, help="Embedding dimension (default 64)")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for Node2Vec")
    args = parser.parse_args()

    build_graph_features(
        nodes_path=args.nodes,
        edges_path=args.edges,
        out_path=args.out,
        emb_dim=args.emb_dim,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
