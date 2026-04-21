import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Use only first 2 features (important for quantum)
X = X[:, :2]

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Quantum device
dev = qml.device("default.qubit", wires=2)

# Quantum circuit
@qml.qnode(dev)
def circuit(inputs, weights):
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    
    qml.CNOT(wires=[0,1])
    
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    
    return qml.expval(qml.PauliZ(0))

# Prediction function
def predict(X, weights):
    preds = []
    for x in X:
        val = circuit(x, weights)
        preds.append(1 if val > 0 else 0)
    return np.array(preds)

# Training (very simple)
weights = np.random.randn(2)

for i in range(20):
    preds = predict(X_train, weights)
    loss = np.mean((preds - y_train) ** 2)
    weights -= 0.1 * np.random.randn(2)  # simple update

# Test
y_pred = predict(X_test, weights)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\n===== QML PERFORMANCE =====")
print("Accuracy:", acc)