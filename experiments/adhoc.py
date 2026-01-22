import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from qiskit_machine_learning.datasets import ad_hoc_data
print("Loading Ad Hoc dataset...")

feature_dim = 2 # Ad Hoc dataset is typically 2D
training_size = 100 # Kept small for reasonable QML simulation time
test_size = 20
adhoc_total = None # We don't need the full dataset object here

# Load the dataset
X_train, y_train, X_test, y_test = ad_hoc_data(
    training_size=training_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.3,    
    plot_data=False, # true to visualize
    one_hot=False
)

# Ad Hoc data does not typically require scaling
X_train_scaled = X_train
X_test_scaled = X_test

print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")
# Map labels from {0, 1} to class names for the report
target_names_adhoc = ["Class A", "Class B"]
print("-" * 50)


print("Running Classical SVM (SVC)...")

# 1. Initialize the CML model (using the standard RBF kernel)
cml_model = SVC(kernel='rbf')

# 2. Train the model
start_time = time.time()
cml_model.fit(X_train_scaled, y_train)
cml_train_time = time.time() - start_time

# 3. Test the model
start_time = time.time()
cml_predictions = cml_model.predict(X_test_scaled)
cml_predict_time = time.time() - start_time

# 4. Report CML Scores
cml_accuracy = accuracy_score(y_test, cml_predictions)

print("\n--- CML (SVC) RESULTS ---")
print(f"Training Time:    {cml_train_time:.4f} seconds")
print(f"Prediction Time:  {cml_predict_time:.4f} seconds")
print(f"Total CML Time:      {cml_train_time + cml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {cml_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, cml_predictions, target_names=target_names_adhoc, zero_division=0))
print("-" * 50)
print("Running Quantum SVM (QSVC)... (This may take a few seconds)")

# 1. Define the Quantum Backend, Fidelity, and Feature Map
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
# Use ZZFeatureMap as this dataset is designed for it
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')

# 2. Define the Quantum Kernel
qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 3. Initialize the QML model
qml_model = QSVC(quantum_kernel=qml_kernel)

# 4. Train the model
start_time = time.time()
qml_model.fit(X_train_scaled, y_train)
qml_train_time = time.time() - start_time

# 5. Test the model
start_time = time.time()
qml_predictions = qml_model.predict(X_test_scaled)
qml_predict_time = time.time() - start_time

# 6. Report QML Scores
qml_accuracy = accuracy_score(y_test, qml_predictions)

print("\n--- QML (QSVC) RESULTS ---")
print(f"Training Time:    {qml_train_time:.4f} seconds")
print(f"Prediction Time:  {qml_predict_time:.4f} seconds")
print(f"Total QML Time:   {qml_train_time + qml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {qml_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, qml_predictions, target_names=target_names_adhoc, zero_division=0))
print("-" * 50)

print("\n--- FINAL COMPARISON (AD HOC DATASET) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}  |  Total Time: {cml_train_time + cml_predict_time:.4f}s")
print(f"QML (QSVC) Accuracy:  {qml_accuracy:.4f}  |  Total Time: {qml_train_time + qml_predict_time:.4f}s")
print("-" * 50)