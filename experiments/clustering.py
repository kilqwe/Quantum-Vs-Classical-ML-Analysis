import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score # ARI for clustering
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

print("Generating and preparing Moons dataset...")

n_samples = 100
noise = 0.1 # Adding some noise
n_qubits = 2 # Moons data is 2D

X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data prepared: {len(X_scaled)} samples.")


print("Running Classical KMeans...")

kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10) # n_init suppresses warning

start_time = time.time()
kmeans_labels = kmeans_model.fit_predict(X_scaled)
kmeans_time = time.time() - start_time

kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)

print("\n--- CML (KMeans) RESULTS ---")
print(f"Clustering Time:  {kmeans_time:.4f} seconds")
print(f"Adjusted Rand Index (ARI): {kmeans_ari:.4f}")
print("-" * 50)


print("Running Classical Spectral Clustering (RBF Kernel)...")
cml_spectral_model = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42)

start_time = time.time()
cml_spectral_labels = cml_spectral_model.fit_predict(X_scaled)
cml_spectral_time = time.time() - start_time


cml_spectral_ari = adjusted_rand_score(y_true, cml_spectral_labels)

print("\n--- CML (Spectral RBF) RESULTS ---")
print(f"Clustering Time:  {cml_spectral_time:.4f} seconds")
print(f"Adjusted Rand Index (ARI): {cml_spectral_ari:.4f}")
print("-" * 50)

print("Running QML Hybrid Spectral Clustering (Quantum Kernel)")
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

print("  Calculating QML Kernel Matrix...")
start_time_kernel = time.time()
# Evaluate the kernel matrix between all pairs of points in X_scaled
qml_kernel_matrix = qml_kernel.evaluate(x_vec=X_scaled)
kernel_calc_time = time.time() - start_time_kernel
print(f"  Kernel Matrix Calculation Time: {kernel_calc_time:.4f} seconds")


qml_spectral_model = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)

start_time_cluster = time.time()
qml_spectral_labels = qml_spectral_model.fit_predict(qml_kernel_matrix)
qml_cluster_time = time.time() - start_time_cluster

qml_total_time = kernel_calc_time + qml_cluster_time
qml_spectral_ari = adjusted_rand_score(y_true, qml_spectral_labels)

print("\n--- QML Hybrid (Spectral Quantum Kernel) RESULTS ---")
print(f"Kernel Matrix Time: {kernel_calc_time:.4f} seconds")
print(f"Clustering Time:    {qml_cluster_time:.4f} seconds")
print(f"Total QML Time:     {qml_total_time:.4f} seconds")
print(f"Adjusted Rand Index (ARI): {qml_spectral_ari:.4f}")


print("\n--- FINAL COMPARISON (MOONS DATASET - CLUSTERING) ---")
print(f"CML (KMeans) ARI:        {kmeans_ari:.4f}  |  Total Time: {kmeans_time:.4f}s")
print(f"CML (Spectral RBF) ARI:  {cml_spectral_ari:.4f}  |  Total Time: {cml_spectral_time:.4f}s")
print(f"QML (Spectral QK) ARI:   {qml_spectral_ari:.4f}  |  Total Time: {qml_total_time:.4f}s")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['KMeans Clustering', 'Spectral Clustering (RBF)', 'Spectral Clustering (Quantum Kernel)']
all_labels = [kmeans_labels, cml_spectral_labels, qml_spectral_labels]
scores = [kmeans_ari, cml_spectral_ari, qml_spectral_ari]

for i, ax in enumerate(axes):
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=all_labels[i], cmap='viridis', s=50)
    ax.set_title(f"{titles[i]}\nARI = {scores[i]:.3f}")
    ax.set_xlabel("Feature 1 (Scaled)")
    ax.set_ylabel("Feature 2 (Scaled)")
    ax.grid(True)

plt.tight_layout()
plt.show()