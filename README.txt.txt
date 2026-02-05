Gen-Q: Quantum Fraud Detection Pipeline
=======================================

This repository contains the complete source code, data flow, and reproduction instructions for our Gen-Q Hackathon submission. Our solution utilizes a Quantum Support Vector Machine (QSVM) enhanced with Node2Vec graph embeddings to detect fraudulent transactions in financial networks.

Repository Structure
--------------------
* src/              : Python scripts for feature engineering, preprocessing, and quantum classification.
* data/             : Directory for raw inputs, intermediate processed files, and final results.
* requirements.txt  : List of dependencies.


Setup
-----
Ensure you have Python 3.8+ installed. Install the required dependencies using:

    pip install -r requirements.txt


Execution Pipeline
------------------
To reproduce our deliverables, execute the scripts in the following order. This pipeline takes raw graph data and produces fraud prediction scores.

Step 1: Feature Engineering (Graph & Node2Vec)
Ingests the raw nodes and edges to generate structural embeddings (Degree, PageRank, and Node2Vec).

    python src/build_features.py \
      --nodes data/raw/graph_nodes_7000.csv \
      --edges data/raw/graph_edges_7000.csv \
      --out data/processed/graph_features.csv

Step 2: Dimensionality Reduction (PCA)
Reduces the high-dimensional graph features to 8 components to fit our 8-qubit quantum circuit.

    python src/prepare_qsvm_input.py \
      --features data/processed/graph_features.csv \
      --out data/processed/qsvm_input_8q.csv \
      --n-components 8

Step 3: Quantum Kernel Classification
Runs the Quantum Support Vector Machine (QSVM). This script calculates the fidelity quantum kernel and saves the results.

    python src/run_qsvm.py \
      --input data/processed/qsvm_input_8q.csv \
      --out data/results/qsvm_results.csv \
      --save-kernel data/results/kernel_matrix.npy


Output Files & Deliverables
---------------------------
After running the pipeline, the 'data/results/' folder will contain:

1. qsvm_results.csv
   The final classification scores (probabilities) for the nodes processed.

2. kernel_matrix.npy
   - Description: The pre-computed Quantum Kernel Matrix (Fidelity Matrix).
   - Usage: This binary file contains the pairwise similarity values between quantum states. It is saved to allow for faster re-training of the SVM without needing to re-run the quantum simulation (which is computationally expensive).
   - Loading: You can load this file in Python using `numpy.load('data/results/kernel_matrix.npy')`.

3. zzfeaturemap_8qubit.png
   The visualization of the 8-qubit quantum circuit used for the kernel estimation.