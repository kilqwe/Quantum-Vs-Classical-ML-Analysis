import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

n_qubits = 2

training_sizes = [20, 40, 60, 80, 100] 

cml_scores = []
zz_scores = []
z_scores = []



for size in training_sizes:
    print(f"\n--- Training Size: {size} ---")
 
    X, y = make_circles(n_samples=size+50, noise=0.1, factor=0.5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=42
    )
    
    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- CML (SVC - RBF) ---
    cml = SVC(kernel='rbf')
    cml.fit(X_train_scaled, y_train)
    cml_acc = accuracy_score(y_test, cml.predict(X_test_scaled))
    cml_scores.append(cml_acc)
    print(f"CML Accuracy: {cml_acc:.2f}")
    
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    
    # --- QML (ZZ Feature Map - Entangled) ---
    zz_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    zz_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=zz_map)
    qsvc_zz = QSVC(quantum_kernel=zz_kernel)
    
    qsvc_zz.fit(X_train_scaled, y_train)
    zz_acc = accuracy_score(y_test, qsvc_zz.predict(X_test_scaled))
    zz_scores.append(zz_acc)
    print(f"QML (ZZ) Accuracy: {zz_acc:.2f}")

    # --- QML (Z Feature Map - No Entanglement) ---
    z_map = ZFeatureMap(feature_dimension=n_qubits, reps=2)
    z_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=z_map)
    qsvc_z = QSVC(quantum_kernel=z_kernel)
    
    qsvc_z.fit(X_train_scaled, y_train)
    z_acc = accuracy_score(y_test, qsvc_z.predict(X_test_scaled))
    z_scores.append(z_acc)
    print(f"QML (Z) Accuracy: {z_acc:.2f}")

plt.figure(figsize=(10, 6))

plt.plot(training_sizes, cml_scores, marker='o', linestyle='-', color='blue', label='CML (SVC-RBF)')
plt.plot(training_sizes, zz_scores, marker='s', linestyle='--', color='green', label='QML (ZZ - Entangled)')
plt.plot(training_sizes, z_scores, marker='^', linestyle=':', color='red', label='QML (Z - Not Entangled)')

plt.title('CML vs QML Accuracy on Make Circles Dataset')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
plt.grid(True)
plt.legend()

plt.savefig('circles_comparison_plot.png')
plt.show()