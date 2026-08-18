import numpy as np

# 1. Configuration and Data
# XOR gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

epochs = 10000
lr = 0.1

# 2. Initialization
input_neurons = X.shape[1]
hidden_neurons = 2
output_neurons = 1

wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))

wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))


# 3. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 4. Training Loop
for i in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wout) + bout
    output = sigmoid(final_input)

    # Backward Propagation
    error = y - output

    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Updating Weights and Biases
    wout += hidden_output.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr

    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr


# 5. Final Predictions
print("Final Predictions after training:")
print(np.round(output, 2))