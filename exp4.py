import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Create Rule-Based Synthetic Dataset
# ============================================================

np.random.seed(42)

n_samples = 1000

voltage = np.random.normal(230, 20, n_samples)
current = np.random.normal(10, 5, n_samples)
frequency = np.random.normal(50, 1, n_samples)
impedance = np.random.normal(10, 3, n_samples)


# Function to generate fault type
def generate_fault(v, c, f, z):

    if c > 15 and v < 220:
        return "LG"

    elif f < 49 and c > 12:
        return "LLG"

    else:
        return "No Fault"


# Generate fault labels
faults = [
    generate_fault(v, c, f, z)
    for v, c, f, z in zip(voltage, current, frequency, impedance)
]


# Create DataFrame
df = pd.DataFrame({
    'Voltage': voltage,
    'Current': current,
    'Frequency': frequency,
    'Impedance': impedance,
    'FaultType': faults
})


# ============================================================
# 2. Preprocessing
# ============================================================

X = df[['Voltage', 'Current', 'Frequency', 'Impedance']]
y = df['FaultType']


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Decision Tree Optimization
# ============================================================

dt_param_grid = {
    'max_depth': [3, 5, 10],
    'criterion': ['entropy', 'gini']
}


dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5
)


# Train Decision Tree
dt_grid.fit(X_train, y_train)


# ============================================================
# 4. KNN Optimization
# ============================================================

# KNN requires feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_param_grid = {
    'n_neighbors': range(1, 15),
    'weights': ['uniform', 'distance']
}


knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=5
)


# Train KNN
knn_grid.fit(X_train_scaled, y_train)


# ============================================================
# 5. Predictions
# ============================================================

dt_pred = dt_grid.predict(X_test)

knn_pred = knn_grid.predict(X_test_scaled)


# ============================================================
# 6. Accuracy
# ============================================================

print("Decision Tree Accuracy:",
      f"{accuracy_score(y_test, dt_pred):.4f}")

print("KNN Accuracy:",
      f"{accuracy_score(y_test, knn_pred):.4f}")


# ============================================================
# 7. Best Parameters
# ============================================================

print("\nBest Decision Tree Parameters:")
print(dt_grid.best_params_)

print("\nBest KNN Parameters:")
print(knn_grid.best_params_)


# ============================================================
# 8. Classification Reports
# ============================================================

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))