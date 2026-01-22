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

# -------------------------------------------------------------------
#  STEP 1: GENERATE PARITY DATASET
# -------------------------------------------------------------------
def generate_parity_data(n_samples, n_features):
    """Generates n-bit parity data."""
    # Generate random binary strings (0 or 1)
    X = np.random.randint(2, size=(n_samples, n_features))
    # Calculate parity: sum across rows, modulo 2
    y = np.sum(X, axis=1) % 2
    return X, y

print("Generating 6-bit Parity dataset...")
n_qubits = 6  # 6 features
n_samples = 100 # Keep it small to show CML struggling to generalize

X, y = generate_parity_data(n_samples, n_qubits)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# No scaling needed (data is binary)
print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")
print(f"Feature dimension: {n_qubits}")
target_names_parity = ["Even Parity", "Odd Parity"]
print("-" * 50)

# -------------------------------------------------------------------
#  STEP 2: CLASSICAL MACHINE LEARNING (CML)
# -------------------------------------------------------------------
print("Running Classical SVM (RBF Kernel)...")
# We use RBF. Linear kernel would fail completely (0.5 accuracy).
cml_model = SVC(kernel='rbf', gamma='scale') 

start_time = time.time()
cml_model.fit(X_train, y_train)
cml_train_time = time.time() - start_time

start_time = time.time()
cml_predictions = cml_model.predict(X_test)
cml_predict_time = time.time() - start_time
cml_accuracy = accuracy_score(y_test, cml_predictions)

print("\n--- CML (SVC) RESULTS ---")
print(f"Training Time:    {cml_train_time:.4f} seconds")
print(f"Accuracy Score:   {cml_accuracy:.4f}")
print(classification_report(y_test, cml_predictions, target_names=target_names_parity))
print("-" * 50)

# -------------------------------------------------------------------
#  STEP 3: QUANTUM MACHINE LEARNING (QML)
# -------------------------------------------------------------------
print(f"Running Quantum SVM (QSVC) with ZZFeatureMap ({n_qubits} qubits)...")

# 1. Quantum Kernel
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

# CRITICAL: Use 'full' entanglement to capture global parity
# 'linear' might fail. 'full' connects everything.
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')

qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
qml_model = QSVC(quantum_kernel=qml_kernel)

# 2. Train
start_time = time.time()
qml_model.fit(X_train, y_train)
qml_train_time = time.time() - start_time

# 3. Test
start_time = time.time()
qml_predictions = qml_model.predict(X_test)
qml_predict_time = time.time() - start_time
qml_accuracy = accuracy_score(y_test, qml_predictions)

print("\n--- QML (QSVC) RESULTS ---")
print(f"Training Time:    {qml_train_time:.4f} seconds")
print(f"Accuracy Score:   {qml_accuracy:.4f}")
print(classification_report(y_test, qml_predictions, target_names=target_names_parity))
print("-" * 50)

# -------------------------------------------------------------------
#  STEP 4: FINAL COMPARISON
# -------------------------------------------------------------------
print("\n--- FINAL COMPARISON (6-BIT PARITY) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}")
print(f"QML (QSVC) Accuracy:  {qml_accuracy:.4f}")
print("-" * 50)