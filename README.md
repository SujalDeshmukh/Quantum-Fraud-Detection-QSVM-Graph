# Quantum Fraud Detection with Graph Embeddings and QSVM



An end-to-end financial transaction fraud detection pipeline combining **Network Graph Feature Engineering** (Degree Centrality, PageRank, Node2Vec Random Walks), **PCA Dimensionality Reduction**, and a **Quantum Support Vector Machine (QSVM)** with parameterized $ZZ$-Feature Maps.

---

## 🌟 Highlights

- **Hybrid Graph-Quantum Learning**: Extracts structural and contextual topology from financial transaction networks and projects them into quantum Hilbert space.
- **Qiskit 2.x Compatible**: Utilizes modern `StatevectorSampler` and `ComputeUncompute` with automatic fallback handling across environments.
- **Precomputed Quantum Fidelity Kernel**: Evaluates exact pairwise quantum state fidelity matrices $K_{ij} = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$.
- **Imbalance Handling**: Automated minority fraud class resampling to prevent majority skew.
- **One-Command Pipeline**: Run the complete workflow from raw graph CSV to fraud probabilities with `src/pipeline.py`.

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Quantum Fraud Detection Pipeline Flow                           │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [1. Raw Graph Data]
  ├── graph_nodes_7000.csv  (node_id, features, IsFraud target)
  └── graph_edges_7000.csv  (source, target, weight)
         │
         ▼
 [2. Feature Engineering (src/build_features.py)]
  ├── Degree Centrality & PageRank
  └── Node2Vec Random Walk Embeddings (64 dimensions)
         │
         ▼
 [3. Dimensionality Reduction (src/prepare_qsvm_input.py)]
  └── Standardized PCA Reduction ➔ 8 Components (Qubits)
         │
         ▼
 [4. Quantum Support Vector Classifier (src/run_qsvm.py)]
  ├── Quantum Feature Map: ZZFeatureMap (reps=1, linear entanglement)
  ├── Fidelity Quantum Kernel: K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|²
  ├── Precomputed Kernel SVC Training & Probability Calibration
  └── Output Results & Circuit Visualization
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SujalDeshmukh/Quantum-Fraud-Detection-QSVM-Graph.git
cd Quantum-Fraud-Detection-QSVM-Graph

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline

Execute all stages in one unified command:

```bash
python src/pipeline.py --n-qubits 8 --sample-n 18 --skip-graph
```

To run feature extraction from scratch on the raw graph:
```bash
python src/pipeline.py --n-qubits 8 --sample-n 18
```

---

## 📊 Step-by-Step Execution

You can also run each modular component individually:

### Step 1: Graph Feature Extraction
Ingests raw nodes and transaction edges to compute structural centrality and Node2Vec embeddings:

```bash
python src/build_features.py \
  --nodes data/raw/graph_nodes_7000.csv \
  --edges data/raw/graph_edges_7000.csv \
  --out data/processed/graph_features.csv \
  --emb-dim 64
```

### Step 2: PCA Dimensionality Reduction
Scales features and projects them onto 8 principal components (matching the 8-qubit quantum circuit):

```bash
python src/prepare_qsvm_input.py \
  --features data/processed/graph_features.csv \
  --out data/processed/qsvm_input_8q.csv \
  --n-components 8 \
  --raw-nodes data/raw/graph_nodes_7000.csv
```

### Step 3: Quantum Kernel Evaluation & Classification
Calculates the quantum fidelity kernel matrix using Qiskit and fits the support vector classifier:

```bash
python src/run_qsvm.py \
  --input data/processed/qsvm_input_8q.csv \
  --out data/results/qsvm_results_8q.csv \
  --save-kernel data/results/qkernel_matrix_8q.npy \
  --sample-n 18
```

### Step 4: Model Benchmarking
Compare QSVM against classical machine learning baselines:

```bash
python src/benchmark.py \
  --input data/processed/qsvm_input_8q.csv \
  --kernel data/results/qkernel_matrix_8q.npy \
  --sample-n 18
```

**Benchmark Results:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Quantum SVM (Fidelity Kernel)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Classical RBF SVM | 0.889 | 1.000 | 0.778 | 0.875 | 1.000 |
| Classical Linear SVM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 🧪 Testing

Run the automated test suite verifying graph feature building, PCA processing, and quantum kernel evaluation:

```bash
pytest tests/test_pipeline.py -v
```

---

## 📁 Repository Structure

```
Quantum-Fraud-Detection-QSVM-Graph/
├── data/
│   ├── raw/
│   │   ├── graph_nodes_7000.csv      # Raw node IDs, features, and fraud labels
│   │   └── graph_edges_7000.csv      # Transaction edge connections and weights
│   ├── processed/
│   │   ├── graph_features.csv        # Degree, PageRank, and Node2Vec embeddings
│   │   └── qsvm_input_8q.csv         # 8-component PCA features with labels
│   └── results/
│       ├── qkernel_matrix_8q.npy     # Computed quantum fidelity kernel matrix
│       ├── qsvm_results_8q.csv       # Fraud prediction probabilities and scores
│       └── zzfeaturemap_8qubit.png   # 8-qubit quantum circuit diagram
├── src/
│   ├── build_features.py             # Graph structural & Node2Vec feature extractor
│   ├── prepare_qsvm_input.py         # PCA dimensionality reducer with label mapping
│   ├── run_qsvm.py                   # Quantum kernel computation and classifier
│   ├── pipeline.py                   # Unified end-to-end pipeline execution runner
│   └── benchmark.py                  # Quantum vs Classical model benchmarking
├── tests/
│   └── test_pipeline.py              # Automated test suite
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📜 License

This project is licensed under the MIT License.
