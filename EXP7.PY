# ============================================================
# MACHINE LEARNING LAB
# Experiment: Comparison of KNN Regression and RBF Regression
# Dataset: Housing.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

data = pd.read_csv("Housing.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ------------------------------------------------------------
# 2. CONVERT CATEGORICAL FEATURES INTO NUMBERS
# ------------------------------------------------------------

# Convert categorical columns into dummy/one-hot variables
data = pd.get_dummies(data, drop_first=True)


# ------------------------------------------------------------
# 3. SEPARATE INPUT FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------

X = data.drop("price", axis=1)
y = data["price"]


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 5. STANDARDIZE THE DATA
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. KNN REGRESSION
# ============================================================

k_values = range(3, 11)

knn_rmse = []
knn_models = []

for k in k_values:

    # Create KNN model
    model = KNeighborsRegressor(
        n_neighbors=k
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    pred = model.predict(X_test_scaled)

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    knn_rmse.append(rmse)
    knn_models.append(model)


# Find best K value
best_knn_index = np.argmin(knn_rmse)

best_k = list(k_values)[best_knn_index]
best_knn_rmse = knn_rmse[best_knn_index]

print("\n========================================")
print("KNN REGRESSION RESULTS")
print("========================================")

print("Best K:", best_k)
print("Best KNN RMSE:", round(best_knn_rmse, 2))


# ============================================================
# 7. RBF KERNEL FUNCTION
# ============================================================

def rbf_kernel(X, centers, gamma):
    """
    Calculate the RBF kernel transformation.
    """
    distances = cdist(X, centers)

    return np.exp(
        -gamma * (distances ** 2)
    )


# ------------------------------------------------------------
# 8. SELECT RBF CENTERS
# ------------------------------------------------------------

# Set a fixed random seed so results are reproducible
np.random.seed(42)

# Number of RBF centers
n_centers = min(20, X_train_scaled.shape[0])

centers = X_train_scaled[
    np.random.choice(
        X_train_scaled.shape[0],
        n_centers,
        replace=False
    )
]


# ------------------------------------------------------------
# 9. SET GAMMA
# ------------------------------------------------------------

gamma = 1.0 / (
    2 * X_train_scaled.shape[1]
)

print("\nRBF Gamma:", gamma)


# ------------------------------------------------------------
# 10. TRANSFORM DATA USING RBF KERNEL
# ------------------------------------------------------------

X_train_rbf = rbf_kernel(
    X_train_scaled,
    centers,
    gamma
)

X_test_rbf = rbf_kernel(
    X_test_scaled,
    centers,
    gamma
)


# ============================================================
# 11. RBF REGRESSION USING RIDGE
# ============================================================

c_values = range(10, 31, 2)

rbf_rmse = []
rbf_models = []

for c in c_values:

    # Ridge regression on RBF-transformed data
    model = Ridge(
        alpha=c
    )

    # Train model
    model.fit(
        X_train_rbf,
        y_train
    )

    # Make predictions
    pred = model.predict(
        X_test_rbf
    )

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    rbf_rmse.append(rmse)
    rbf_models.append(model)


# Find best C value
best_rbf_index = np.argmin(rbf_rmse)

best_c = list(c_values)[best_rbf_index]
best_rbf_rmse = rbf_rmse[best_rbf_index]


# ============================================================
# 12. DISPLAY RBF RESULTS
# ============================================================

print("\n========================================")
print("RBF REGRESSION RESULTS")
print("========================================")

print("Best C (Ridge alpha):", best_c)
print("Best RBF RMSE:", round(best_rbf_rmse, 2))


# ============================================================
# 13. FINAL COMPARISON
# ============================================================

print("\n========================================")
print("FINAL COMPARISON")
print("========================================")

print(
    "Best KNN RMSE:",
    round(best_knn_rmse, 2)
)

print(
    "Best RBF RMSE:",
    round(best_rbf_rmse, 2)
)


if best_knn_rmse < best_rbf_rmse:

    print("\nKNN performed better for this dataset.")

elif best_rbf_rmse < best_knn_rmse:

    print("\nRBF performed better for this dataset.")

else:

    print("\nBoth models performed equally.")


# ============================================================
# 14. PLOT KNN RMSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    knn_rmse,
    marker="o"
)

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("RMSE")
plt.title("KNN Regression: K vs RMSE")

plt.grid(True)

plt.show()


# ============================================================
# 15. PLOT RBF RMSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(c_values),
    rbf_rmse,
    marker="o"
)

plt.xlabel("Ridge Alpha (C)")
plt.ylabel("RMSE")
plt.title("RBF Regression: Alpha vs RMSE")

plt.grid(True)

plt.show()


# ============================================================
# 16. ACTUAL VS PREDICTED VALUES FOR BEST KNN
# ============================================================

best_knn_model = knn_models[best_knn_index]

best_knn_predictions = best_knn_model.predict(
    X_test_scaled
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_knn_predictions
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("KNN: Actual vs Predicted House Prices")

plt.grid(True)

plt.show()


# ============================================================
# 17. ACTUAL VS PREDICTED VALUES FOR BEST RBF
# ============================================================

best_rbf_model = rbf_models[best_rbf_index]

best_rbf_predictions = best_rbf_model.predict(
    X_test_rbf
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_rbf_predictions
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("RBF: Actual vs Predicted House Prices")

plt.grid(True)

plt.show()