import numpy as np
import time

# --- Scikit-Learn (CML) Imports ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# No scaling needed for this binary data

# --- Qiskit (QML) Imports ---
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# -------------------------------------------------------------------
#  STEP 1: BARS AND STRIPES DATASET GENERATOR
# -------------------------------------------------------------------

def generate_bars_and_stripes(n_samples, n_features_sqrt):
    """
    Generates 'n_samples' of bar-or-stripe images.
    n_features_sqrt: The width/height of the image (e.g., 3 for a 3x3=9 qubit/feature map).
    """
    n_features = n_features_sqrt * n_features_sqrt
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # 0 = Bars, 1 = Stripes
        label = np.random.randint(2)
        y[i] = label
        img = np.zeros((n_features_sqrt, n_features_sqrt))

        if label == 0:  # Bars (Vertical)
            col_pattern = np.random.randint(2, size=n_features_sqrt)
            if np.sum(col_pattern) == 0: # Ensure at least one bar
                col_pattern[np.random.randint(n_features_sqrt)] = 1
            for col in range(n_features_sqrt):
                if col_pattern[col] == 1:
                    img[:, col] = 1
        else:  # Stripes (Horizontal)
            row_pattern = np.random.randint(2, size=n_features_sqrt)
            if np.sum(row_pattern) == 0: # Ensure at least one stripe
                row_pattern[np.random.randint(n_features_sqrt)] = 1
            for row in range(n_features_sqrt):
                if row_pattern[row] == 1:
                    img[row, :] = 1
        
        # Add some noise (e.g., ~5% pixel flips)
        noise = np.random.rand(n_features_sqrt, n_features_sqrt) < 0.05
        img = np.abs(img - noise) # Flip bits where noise is True
        
        X[i, :] = img.flatten()

    return X, y

# -------------------------------------------------------------------
#  STEP 2: LOAD AND PREPARE THE DATA
# -------------------------------------------------------------------
print("Loading Bars and Stripes dataset...")

n_features_sqrt = 3 # 3x3 grid
n_qubits = n_features_sqrt * n_features_sqrt # 9 qubits

# --- FULL DATASET ---
training_size = 100
test_size = 50

# # --- QUICK TEST (Recommended first!) ---
# training_size = 20
# test_size = 10

X_train, y_train = generate_bars_and_stripes(training_size, n_features_sqrt)
X_test, y_test = generate_bars_and_stripes(test_size, n_features_sqrt)

# No scaling needed as data is already 0 or 1
X_train_scaled = X_train
X_test_scaled = X_test

print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")
print(f"Feature dimension (qubits): {n_qubits}")
target_names_bs = ["Bars", "Stripes"]
print("-" * 50)


# -------------------------------------------------------------------
#  STEP 3: CLASSICAL MACHINE LEARNING (CML) MODEL
# -------------------------------------------------------------------
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
print(f"Total CML Time:   {cml_train_time + cml_predict_time:.4f} seconds")
print(f"\nAccuracy Score:   {cml_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, cml_predictions, target_names=target_names_bs, zero_division=0))
print("-" * 50)

# -------------------------------------------------------------------
#  STEP 4: QUANTUM MACHINE LEARNING (QML) MODEL
# -------------------------------------------------------------------
print(f"Running Quantum SVM (QSVC) with {n_qubits} qubits...")


# 1. Define the Quantum Backend, Fidelity, and Feature Map
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
# Use ZZFeatureMap for all 9 qubits
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')

# 2. Define the Quantum Kernel
qml_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 3. Initialize the QML model
qml_model = QSVC(quantum_kernel=qml_kernel)

# 4. Train the model (THE SLOW PART)
start_time = time.time()
qml_model.fit(X_train_scaled, y_train)
qml_train_time = time.time() - start_time

# 5. Test the model (ALSO SLOW)
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
print(classification_report(y_test, qml_predictions, target_names=target_names_bs, zero_division=0))
print("-" * 50)

# -------------------------------------------------------------------
#  STEP 5: FINAL COMPARISON
# -------------------------------------------------------------------
print("\n--- FINAL COMPARISON (BARS & STRIPES DATASET) ---")
print(f"CML (SVC) Accuracy:   {cml_accuracy:.4f}  |  Total Time: {cml_train_time + cml_predict_time:.4f}s")
print(f"QML (QSVC) Accuracy:  {qml_accuracy:.4f}  |  Total Time: {qml_train_time + qml_predict_time:.4f}s")
print("-" * 50)