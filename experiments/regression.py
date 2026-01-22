import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA 
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR # Quantum SVR
from qiskit_machine_learning.state_fidelities import ComputeUncompute
print("Loading and preparing Diabetes dataset...")

diabetes = load_diabetes()
X = diabetes.data # Has 10 features
y = diabetes.target

n_features_original = X.shape[1]
n_qubits = 2 # Number of qubits/features for QML
print(f"Reducing features from {n_features_original} to {n_qubits} using PCA...")
pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# Scale the PCA-reduced data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")
print("-" * 50)

print("Running Classical SVR...")

cml_model = SVR(kernel='rbf')

start_time = time.time()
cml_model.fit(X_train_scaled, y_train)
cml_train_time = time.time() - start_time

start_time = time.time()
cml_predictions = cml_model.predict(X_test_scaled)
cml_predict_time = time.time() - start_time

cml_r2 = r2_score(y_test, cml_predictions)

print("\n--- CML (SVR) RESULTS ---")
print(f"Training Time:    {cml_train_time:.4f} seconds")
print(f"Prediction Time:  {cml_predict_time:.4f} seconds")
print(f"Total CML Time:   {cml_train_time + cml_predict_time:.4f} seconds")
print(f"\nR-squared Score:  {cml_r2:.4f}")
print("-" * 50)

print("Running Quantum SVR (QSVR)... (This may take several minutes)")

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
# Using ZZFeatureMap again for consistency
feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=2)
qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
qml_model = QSVR(quantum_kernel=qml_kernel)

start_time = time.time()
qml_model.fit(X_train_scaled, y_train)
qml_train_time = time.time() - start_time

start_time = time.time()
qml_predictions = qml_model.predict(X_test_scaled)
qml_predict_time = time.time() - start_time

qml_r2 = r2_score(y_test, qml_predictions)

print("\n--- QML (QSVR) RESULTS ---")
print(f"Training Time:    {qml_train_time:.4f} seconds")
print(f"Prediction Time:  {qml_predict_time:.4f} seconds")
print(f"Total QML Time:   {qml_train_time + qml_predict_time:.4f} seconds")
print(f"\nR-squared Score:  {qml_r2:.4f}")
print("-" * 50)

print("\n--- FINAL COMPARISON (DIABETES DATASET - REGRESSION) ---")
print(f"CML (SVR) R2 Score:   {cml_r2:.4f}  |  Total Time: {cml_train_time + cml_predict_time:.4f}s")
print(f"QML (QSVR) R2 Score:  {qml_r2:.4f}  |  Total Time: {qml_train_time + qml_predict_time:.4f}s")
print("-" * 50)