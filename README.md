# Quantum vs. Classical: A Comparative Benchmark of Kernel Methods

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-Machine%20Learning-purple?logo=qiskit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-green)

> **Abstract:** A comprehensive study benchmarking Quantum Machine Learning (QML) against Classical Machine Learning (CML) across 13 diverse datasets. This project identifies the specific "structural niche" where quantum kernels outperform classical RBF kernels.

## Executive Summary & Key Results

We tested **13 datasets** ranging from standard tabular data (Iris, Diabetes) to synthetic quantum-native problems (AdHoc, Parity).

**The Verdict:** QML is not a universal accelerator. It dominates in domains requiring **global, non-local logic** (entanglement) but fails to beat classical optimization on standard geometric clusters.

![Benchmark Summary](benchmark_summary.png)


### Benchmark Table (Top 10 Datasets)

| Dataset | Task | CML Score (SVM) | QML Score (QSVC) | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Iris (Simplified)** | Classification | **1.00** | 0.45 | CML Wins |
| **Diabetes (PCA)** | Regression ($R^2$) | **0.17** | -0.012 | CML Wins |
| **Make Circles** | Non-Linear | **0.93** | 0.80 | CML Wins |
| **Moons** | Clustering | **0.54** (ARI) | -0.004 | CML Wins |
| **MNIST (3 vs 8)** | Image Rec. | **0.95** | 0.43 | CML Wins |
| **HIGGS (PCA)** | Physics | **0.52** | 0.50 | CML Wins |
| **Bars & Stripes** | Pattern Rec. | 0.90 | **0.92** | **QML Slight Edge** |
| **6-bit Parity** | Logical XOR | 0.37 | **1.00** | **QML Dominates** |
| **AdHoc** | Synthetic | 0.65 | **1.00** | **QML Dominates** |

---

## Methodology

The core objective was to compare the **Radial Basis Function (RBF)** kernel against the **Fidelity Quantum Kernel**.

### 1. Classical Approach (Baseline)
* **Algorithm:** Support Vector Classifier (SVC) / SVR.
* **Implementation:** `scikit-learn`.
* **Mechanism:** Maps data to infinite-dimensional space using the RBF kernel to find a separating hyperplane .

### 2. Quantum Approach (Proposed)
* **Algorithm:** Quantum SVC / QSVR.
* **Implementation:** `qiskit-machine-learning`.
* **Mechanism:** Maps data into a high-dimensional Hilbert space using a parameterized quantum circuit.The kernel is calculated via the "kernel trick" (Fidelity test).
* **Feature Maps (Encodings) Used:**
    * `ZFeatureMap`: For simple geometric data (linear, no entanglement).
    * `ZZFeatureMap`: For logic problems (creates pairwise entanglement).
    * `PauliFeatureMap`: For complex correlation testing.

---

## Result Overview

Our results highlight two critical insights regarding the utility of QML:

### 1. The Parity Problem (XOR)
The **6-bit Parity** dataset represents a problem where the output depends on the global correlation of all inputs (XOR logic).
* **Classical Failure:** Standard RBF kernels struggle to capture this dependency (Accuracy: ~37%).
* **Quantum Success:** Using the `ZZFeatureMap` with entanglement, the quantum kernel naturally maps these correlations, achieving **100% accuracy**.

### 2. The "Simulation Barrier"
While QML showed accuracy gains in specific niches, the computational cost on a classical simulator was immense. We observed an approximate **10,000x slowdown** due to the exponential cost ($2^n$) of simulating entanglement. This confirms QML is designed for real QPUs, not simulators.

## Case Study
### 1. The "Quantum Win" (6-bit Parity)
The Parity problem requires global entanglement, which the classical kernel fails to capture.
![Parity Results](results/parity_comparison_plot.png)

### 2. The "Classical Win" (Iris Dataset)
For simple geometric clusters, the quantum kernel adds unnecessary noise.
![Iris Results](results/iris_comparison_plot.png)

<details>
<summary><strong> Click to see results for other datasets (Higgs, Circles, Moons)</strong></summary>

### Make Circles (Non-Linear)
![Circles Results](results/circles_comparison_plot.png)

### HIGGS Boson (Physics Data)
![Higgs Results](results/higgs_comparison_plot.png)

### AdHoc (Synthetic)
![AdHoc Results](results/cml_vs_qml_comparison.png)

</details>

---

## Repository Structure

```text
├── experiments/            # Python scripts for each dataset
│   ├── adhoc_qml.py
│   ├── parity_qml.py
│   ├── iris_qml.py
│   └── ...
├── results/                # Generated plots and logs
├── visual.py               # Represents Dataset as plotted points
├── benchmark_summary.png   # Chart showing the results obtained
├── requirements.txt        # Project dependencies
└── README.md
