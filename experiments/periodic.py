import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

def generate_parity_data(n_samples, n_features):
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.sum(X, axis=1) % 2
    return X, y

# 12 Qubits = The breaking point for CML on small data
n_qubits = 12  
n_samples = 100 # Very sparse data for 12 dimensions

print(f"Generating {n_qubits}-bit Parity dataset...")
X, y = generate_parity_data(n_samples, n_qubits)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Data prepared: {len(X_train)} training samples.")
print(f"Dimensionality: {n_qubits} (Hypercube corners: {2**n_qubits})")



print("Running Classical SVM (RBF Kernel)...")
cml_model = SVC(kernel='rbf', gamma='scale') 

start_time = time.time()
cml_model.fit(X_train, y_train)
cml_train_time = time.time() - start_time
cml_accuracy = accuracy_score(y_test, cml_model.predict(X_test))

print("\n--- CML (SVC) RESULTS ---")
print(f"Accuracy Score:   {cml_accuracy:.4f}")

print(f"Running Quantum SVM (QSVC) with ZZFeatureMap ({n_qubits} qubits)...")
print("This uses 'Full' entanglement to capture global correlations.")

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

# Entanglement='full' connects every qubit to every other qubit.
# This is computationally expensive but logically necessary for Parity.
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')

qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
qml_model = QSVC(quantum_kernel=qml_kernel)

start_time = time.time()
qml_model.fit(X_train, y_train)
qml_train_time = time.time() - start_time
qml_accuracy = accuracy_score(y_test, qml_model.predict(X_test))

print("\n--- QML (QSVC) RESULTS ---")
print(f"Accuracy Score:   {qml_accuracy:.4f}")

print(f"\n--- FINAL COMPARISON ({n_qubits}-BIT PARITY) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}")
print(f"QML (QSVC) Accuracy:  {qml_accuracy:.4f}")
