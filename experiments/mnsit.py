import numpy as np
import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute


X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='auto')
print("MNIST dataset loaded.")

# --- Filter for digits 3 and 8 ---
X_filtered = X[(y == '3') | (y == '8')]
y_filtered = y[(y == '3') | (y == '8')]

# Convert labels '3' and '8' to 0 and 1
y_binary = np.where(y_filtered == '3', 0, 1)


# Using the full dataset (13k+ images) is impossible to simulate.
n_samples = 400
n_qubits = 9 

X_sample, _, y_sample, _ = train_test_split(
    X_filtered, y_binary, train_size=n_samples, random_state=42, stratify=y_binary
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
)

# Preprocessing
# Scale pixel values (0-255) to have mean 0 and std 1
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
print(f"Applying PCA to reduce {X_train.shape[1]} features to {n_qubits}...")
pca = PCA(n_components=n_qubits).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")
print(f"Feature dimension (qubits): {n_qubits}")
target_names_mnist = ["Digit 3", "Digit 8"]



print("Running Classical SVM (SVC) on PCA data...")

# 1. Initialize CML model
cml_model = SVC(kernel='rbf')

# 2. Train (on 9-feature PCA data)
start_time = time.time()
cml_model.fit(X_train_pca, y_train)
cml_train_time = time.time() - start_time

# 3. Test
start_time = time.time()
cml_predictions = cml_model.predict(X_test_pca)
cml_predict_time = time.time() - start_time

# 4. Report CML Scores
cml_accuracy = accuracy_score(y_test, cml_predictions)

print("\n--- CML (SVC) RESULTS ---")
print(f"Training Time:    {cml_train_time:.4f} seconds")
print(f"Prediction Time:  {cml_predict_time:.4f} seconds")
print(f"Total CML Time:   {cml_train_time + cml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {cml_accuracy:.4f}")
print(classification_report(y_test, cml_predictions, target_names=target_names_mnist))



print(f"Running Quantum SVM (QSVC) with {n_qubits} qubits...")
print("!!! SLOW RUN TIME (10-30+ min) !!!")

# 1. Define Quantum Kernel (using 9 qubits)
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=2)

qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 2. Initialize QML model
qml_model = QSVC(quantum_kernel=qml_kernel)

# 3. Train 
start_time = time.time()
qml_model.fit(X_train_pca, y_train)
qml_train_time = time.time() - start_time

# 4. Test 
start_time = time.time()
qml_predictions = qml_model.predict(X_test_pca)
qml_predict_time = time.time() - start_time

# 5. Report QML Scores
qml_accuracy = accuracy_score(y_test, qml_predictions)

print("\n--- QML (QSVC) RESULTS ---")
print(f"Training Time:    {qml_train_time:.4f} seconds")
print(f"Prediction Time:  {qml_predict_time:.4f} seconds")
print(f"Total QML Time:   {qml_train_time + qml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {qml_accuracy:.4f}")
print(classification_report(y_test, qml_predictions, target_names=target_names_mnist))

print("\n--- FINAL COMPARISON (MNIST 3 vs 8 DATASET) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}  |  Total Time: {cml_train_time + cml_predict_time:.4f}s")
print(f"QML (QSVC) Accuracy:  {qml_accuracy:.4f}  |  Total Time: {qml_train_time + qml_predict_time:.4f}s")
