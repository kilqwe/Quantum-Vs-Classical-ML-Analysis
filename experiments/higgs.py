import numpy as np
import time
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



X, y = fetch_openml(data_id=42769, return_X_y=True, as_frame=False, parser='auto')
print("HIGGS dataset loaded.")

# Convert labels (1.0 = signal, 0.0 = background) to integers
y = y.astype(int)


n_samples = 400 
n_qubits = 9 

X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=n_samples, random_state=42, stratify=y
)

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
)

# Preprocessing
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
target_names_higgs = ["Background", "Signal (Higgs)"]

print("Running Classical SVM (SVC) on PCA data...")

# Initialize CML model
cml_model = SVC(kernel='rbf')
start_time = time.time()
cml_model.fit(X_train_pca, y_train)
cml_train_time = time.time() - start_time
start_time = time.time()
cml_predictions = cml_model.predict(X_test_pca)
cml_predict_time = time.time() - start_time
cml_accuracy = accuracy_score(y_test, cml_predictions)

print("\n--- CML (SVC) RESULTS ---")
print(f"Training Time:    {cml_train_time:.4f} seconds")
print(f"Prediction Time:  {cml_predict_time:.4f} seconds")
print(f"Total CML Time:   {cml_train_time + cml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {cml_accuracy:.4f}")
print(classification_report(y_test, cml_predictions, target_names=target_names_higgs, zero_division=0))

def run_qml_experiment(feature_map, map_name, n_qubits):
    print(f"Running Quantum SVM (QSVC) with {map_name} ({n_qubits} qubits)...")
    if n_qubits >= 9:
        print("!!! SLOW RUN TIME(10-30+ min) !!!")

    # 1. Define Quantum Kernel
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
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

    # 5. Report
    qml_accuracy = accuracy_score(y_test, qml_predictions)
    total_time = qml_train_time + qml_predict_time

    print(f"\n--- QML (QSVC {map_name}) RESULTS ---")
    print(f"Training Time:    {qml_train_time:.4f} seconds")
    print(f"Prediction Time:  {qml_predict_time:.4f} seconds")
    print(f"Total QML Time:   {total_time:.4f} seconds")
    print(f"\nAccuracy Score:   {qml_accuracy:.4f}")
    print(classification_report(y_test, qml_predictions, target_names=target_names_higgs, zero_division=0))
  
    
    return qml_accuracy, total_time


zz_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
qml_zz_acc, qml_zz_time = run_qml_experiment(zz_map, "ZZFeatureMap", n_qubits)

z_map = ZFeatureMap(feature_dimension=n_qubits, reps=2)
qml_z_acc, qml_z_time = run_qml_experiment(z_map, "ZFeatureMap", n_qubits)


print("\n--- FINAL COMPARISON (HIGGS DATASET) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}  |  Total Time: {cml_train_time + cml_predict_time:.4f}s")
print(f"QML (QSVC ZZMap) Accuracy: {qml_zz_acc:.4f}  |  Total Time: {qml_zz_time:.4f}s")
print(f"QML (QSVC ZMap) Accuracy:  {qml_z_acc:.4f}  |  Total Time: {qml_z_time:.4f}s")
