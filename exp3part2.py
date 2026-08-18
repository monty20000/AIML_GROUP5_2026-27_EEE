from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Input: [Vin, R1, R2]
X = np.array([
    [1.0, 1000, 1000],
    [1.5, 1000, 2000],
    [2.0, 2000, 2000],
    [2.5, 1000, 1000],
    [1.0, 2000, 1000]
])

# 2. Output: Vout
y = np.array([
    2.0,
    4.5,
    4.0,
    5.0,
    1.5
])

# 3. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Define and train MLP
model = MLPRegressor(
    hidden_layer_sizes=(5, 5),
    activation='relu',
    max_iter=10000,
    random_state=1
)

model.fit(X_scaled, y)

# 5. Make predictions
predictions = model.predict(X_scaled)

# 6. Display results
print("Actual Target Values:", y)
print("Model Predictions:   ", np.round(predictions, 2))