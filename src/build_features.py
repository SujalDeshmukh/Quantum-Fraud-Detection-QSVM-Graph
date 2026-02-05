#!/usr/bin/env python3
"""
Build graph-based features (degree, PageRank, Node2Vec embeddings)
for the SVM + QSVM pipeline.
"""

import argparse, pandas as pd, numpy as np, networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler

# ---------- Parse arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", required=True, help="Path to nodes.csv (node_id column)")
parser.add_argument("--edges", required=True, help="Path to edges.csv (source,target columns)")
parser.add_argument("--out", required=True, help="Output CSV path")
parser.add_argument("--emb-dim", type=int, default=64, help="Embedding dimension (default 64)")
args = parser.parse_args()

# ---------- Load data ----------
print("Loading nodes and edges …")
nodes = pd.read_csv(args.nodes, dtype=str)
edges = pd.read_csv(args.edges, dtype=str)

# Normalize column names just in case
edges.columns = [c.lower() for c in edges.columns]
if "source" not in edges or "target" not in edges:
    raise ValueError("Edges file must have 'source' and 'target' columns")

# ---------- Build graph ----------
G = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.Graph())

# Ensure all nodes appear
for n in nodes["node_id"]:
    if n not in G:
        G.add_node(n)
nodes_sorted = sorted(G.nodes())
print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges")

# ---------- Structural features ----------
deg = np.array([G.degree(n) for n in nodes_sorted], float)
pagerank_dict = nx.pagerank(G)
pagerank = np.array([pagerank_dict.get(n, 0.0) for n in nodes_sorted], float)

# ---------- Node2Vec embeddings ----------
print("Running Node2Vec (this may take a few minutes)…")
n2v = Node2Vec(G, dimensions=args.emb_dim, walk_length=20,
               num_walks=10, workers=2, seed=42)
model = n2v.fit(window=10, min_count=1, batch_words=4)

emb = np.vstack([
    model.wv[str(n)] if str(n) in model.wv else np.zeros(args.emb_dim)
    for n in nodes_sorted
])

# ---------- Combine and scale ----------
X = np.hstack([deg[:, None], pagerank[:, None], emb])
X = StandardScaler().fit_transform(X)

df = pd.DataFrame(
    X,
    columns=["deg", "pagerank"] + [f"emb_{i}" for i in range(args.emb_dim)],
)
df.insert(0, "node_id", nodes_sorted)
df.to_csv(args.out, index=False)
print(f"Saved features to {args.out} — shape {df.shape}")
