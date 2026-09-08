# ============================================================
# MACHINE LEARNING LAB
# Experiment: DBSCAN Clustering with Automated Parameter Search
# Dataset: Mall_Customers.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

data = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())


# ------------------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------------------

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values

print("\nNumber of customers:", X.shape[0])


# ------------------------------------------------------------
# 3. STANDARDIZE THE DATA
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 4. AUTOMATICALLY FIND THE BEST DBSCAN PARAMETERS
# ------------------------------------------------------------

best_score = -1
best_params = None
best_labels = None

eps_values = np.arange(0.1, 1.0, 0.05)
min_samples_values = range(3, 15, 2)


for eps in eps_values:

    for min_samples in min_samples_values:

        # Create DBSCAN model
        db = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        # Predict cluster labels
        labels = db.fit_predict(X_scaled)

        # Count clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # We need at least 2 clusters for silhouette score
        if n_clusters > 1:

            score = silhouette_score(X_scaled, labels)

            # Check whether this is the best result
            if score > best_score:

                best_score = score
                best_params = (eps, min_samples)
                best_labels = labels


# ------------------------------------------------------------
# 5. DISPLAY BEST PARAMETERS
# ------------------------------------------------------------

if best_params is not None:

    best_eps = best_params[0]
    best_min_samples = best_params[1]

    print("\n========================================")
    print("BEST DBSCAN PARAMETERS")
    print("========================================")

    print("Best eps:", round(best_eps, 2))
    print("Best min_samples:", best_min_samples)
    print("Best Silhouette Score:", round(best_score, 4))


    # --------------------------------------------------------
    # 6. COUNT CLUSTERS AND NOISE POINTS
    # --------------------------------------------------------

    n_clusters = len(set(best_labels)) - (
        1 if -1 in best_labels else 0
    )

    n_noise = list(best_labels).count(-1)

    print("Number of clusters:", n_clusters)
    print("Number of noise points:", n_noise)


    # --------------------------------------------------------
    # 7. DISPLAY NUMBER OF CUSTOMERS IN EACH CLUSTER
    # --------------------------------------------------------

    print("\n========================================")
    print("CUSTOMERS IN EACH CLUSTER")
    print("========================================")

    unique_labels, counts = np.unique(best_labels, return_counts=True)

    for label, count in zip(unique_labels, counts):

        if label == -1:
            print("Noise:", count, "customers")
        else:
            print("Cluster", label, ":", count, "customers")


    # --------------------------------------------------------
    # 8. VISUALIZE THE DBSCAN CLUSTERS
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=best_labels,
        cmap="plasma",
        s=60
    )

    plt.title(
        f"DBSCAN Clustering "
        f"(eps={best_eps:.2f}, min_samples={best_min_samples})"
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")

    plt.colorbar(
        scatter,
        label="Cluster Label"
    )

    plt.grid(True, alpha=0.3)

    plt.show()


else:

    print("\nNo suitable DBSCAN parameters were found.")