import os
import cv2
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# =========================================================
# 1. LOAD IMAGE DATASET
# =========================================================

DATASET_PATH = r"brain_tumor_dataset"

IMG_SIZE = (64, 64)

X = []
y = []


# Class 0 = No tumor
# Class 1 = Tumor

class_map = {
    "no": 0,
    "yes": 1
}


for class_name, label in class_map.items():

    folder_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        continue

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        # Read image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, IMG_SIZE)

        # Normalize pixel values from 0-255 to 0-1
        image = image / 255.0

        # Flatten image
        image = image.flatten()

        X.append(image)
        y.append(label)


# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)


print("Dataset loaded successfully!")
print("Total samples:", len(X))
print("Number of features:", X.shape[1])


# =========================================================
# 2. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 3. RBF TRANSFORMER
# =========================================================

class RBFTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_centers=150,
        gamma=1.0,
        random_state=42
    ):
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):

        n_centers = min(
            self.n_centers,
            X.shape[0]
        )

        self.kmeans_ = KMeans(
            n_clusters=n_centers,
            random_state=self.random_state,
            n_init=10
        )

        self.kmeans_.fit(X)

        self.centers_ = self.kmeans_.cluster_centers_

        return self

    def transform(self, X):

        # Calculate squared norms of samples
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)

        # Calculate squared norms of centers
        centers_norm = np.sum(
            self.centers_ ** 2,
            axis=1,
            keepdims=True
        ).T

        # Calculate squared Euclidean distances
        distances = (
            X_norm
            + centers_norm
            - 2 * np.dot(X, self.centers_.T)
        )

        # Avoid small negative values due to floating-point errors
        distances = np.maximum(distances, 0)

        # Apply Gaussian RBF
        rbf_features = np.exp(
            -self.gamma * distances
        )

        return rbf_features


# =========================================================
# 4. GAUSSIAN NAIVE BAYES PIPELINE
# =========================================================

nb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GaussianNB())
])


# =========================================================
# 5. RBF NETWORK PIPELINE
# =========================================================

rbf_pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("rbf", RBFTransformer(
        n_centers=150,
        gamma=1.0
    )),

    ("clf", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])


# =========================================================
# 6. TRAINING AND TESTING
# =========================================================

models = [
    ("GaussianNB", nb_pipeline),
    ("RBF Network", rbf_pipeline)
]


for name, pipe in models:

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train
    pipe.fit(X_train, y_train)

    # Predict class
    y_pred = pipe.predict(X_test)

    # Predict probability
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auroc = roc_auc_score(
        y_test,
        y_prob
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")