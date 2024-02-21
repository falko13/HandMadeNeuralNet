import numpy as np
import time

class CustomNormalization:
    def __init__(self):
        """
        Initialize with predefined mean and variance for testing without actual training data.
        This is only for demonstration purposes and should ideally be replaced with dynamic computation in fit method.
        """
        self.mean = [218.67, 13.43]
        self.variance = [1.60e+03, 1.27e+00]
        self.std = [v**0.5 for v in self.variance]  # Calculate standard deviation from variance

    def fit(self, X):
        """
        Computes the mean and standard deviation from the training data X.
        This method is designed to learn parameters from the training data.

        Args:
            X (ndarray): Training data, shape (m, n), with m examples and n features.
        """
        m, n = X.shape
        self.mean = [sum(X[:, j]) / m for j in range(n)]
        self.variance = [sum((x_ij - self.mean[j])**2 for x_ij in X[:, j]) / m for j in range(n)]
        self.std = [v**0.5 for v in self.variance]

    def transform(self, X):
        """
        Applies normalization to the data X based on learned parameters.

        Args:
            X (ndarray): Data to be normalized, shape (m, n), with m examples and n features.

        Returns:
            ndarray: The normalized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalization parameters are not initialized. Call 'fit' with training data first.")

        X_normalized = (X - np.array(self.mean)) / np.array(self.std)
        return X_normalized

    def fit_transform(self, X):
        """
        Fits the normalization parameters and transforms the data in a single step.

        Args:
            X (ndarray): Training data to learn from and to be normalized.

        Returns:
            ndarray: The normalized training data.
        """
        self.fit(X)
        return self.transform(X)


# Example usage
# Assuming X_train is training data and X_test is new data to be normalized
# X_train = np.array([...])  # Training data
# X_test = np.array([...])   # New data


def sigmoid(z):
    """
    The sigmoid activation function. Maps any real value to the (0, 1) interval, acting as an activation function.

    Args:
        z (float or ndarray): The input value(s) to the sigmoid function.

    Returns:
        float or ndarray: The computed sigmoid value.
    """
    return 1 / (1 + np.exp(-z))

def dense(a_in, W, b, g=sigmoid):
    """
    Represents a dense (fully connected) layer in a neural network. This function computes the layer's output.

    Args:
        a_in (ndarray): The input array to the layer, shape (n,).
        W (ndarray): The weight matrix of the layer, shape (n, j).
        b (ndarray): The bias vector of the layer, shape (j,).
        g (callable): The activation function to be applied.

    Returns:
        ndarray: The output of the dense layer, shape (j,).
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def dense_vectorized(AT, W, b, g=sigmoid):
    """
    Represents a dense (fully connected) layer in a neural network utilizing vectorized operations for efficiency.
    This function computes the outputs for multiple inputs in a batch.

    Args:
        AT (ndarray): The transpose of the input matrix to the layer, shape (n, m), where m is the number of input samples.
        W (ndarray): The weight matrix of the layer, shape (n, j).
        b (ndarray): The bias vector of the layer, shape (j,).
        g (callable): The activation function to be applied element-wise.

    Returns:
        ndarray: The output of the dense layer for all inputs, shape (m, j), where each row corresponds to the output for one input sample.

    Notes:
        - The function is optimized for handling multiple inputs at once, leveraging matrix multiplication for faster computation compared to iterating over inputs.
    """
    Z = np.matmul(AT,W)+b
    a_out = g(Z)
    return a_out

def sequential(x, W1, b1, W2, b2, vectorized=False):
    """
    Constructs and evaluates a simple 2-layer neural network model.
    Can operate in vectorized mode for efficiency.

    Args:
        x (ndarray): The input data. Shape (n,) for non-vectorized, shape (m, n) for vectorized.
        W1, W2 (ndarray): Weight matrices for the first and second layers.
        b1, b2 (ndarray): Bias vectors for the first and second layers.
        vectorized (bool): If True, uses the vectorized implementation.

    Returns:
        ndarray: The output of the neural network.
    """
    if vectorized:
        a1 = dense_vectorized(x, W1, b1, sigmoid)  # Note: x and following a1 is already transposed, so no additional x.T,a1.T required
        a2 = dense_vectorized(a1, W2, b2, sigmoid)
    else:
        a1 = dense(x, W1, b1, sigmoid)
        a2 = dense(a1, W2, b2, sigmoid)
    return a2

def predict(X, W1, b1, W2, b2):
    """
    Predicts the output for multiple inputs using the sequential neural network model.
    Compares the efficiency of non-vectorized and vectorized implementations.

    Args:
        X (ndarray): The input data, shape (m, n), for m examples.
        W1, W2 (ndarray): Weight matrices for the network.
        b1, b2 (ndarray): Bias vectors for the network.

    Returns:
        Prints predictions and efficiency comparison.
    """
    start_time = time.time()
    predictions_non_vectorized = np.array([sequential(x, W1, b1, W2, b2) for x in X])
    non_vectorized_time = time.time() - start_time

    start_time = time.time()
    predictions_vectorized = sequential(X, W1, b1, W2, b2, vectorized=True)
    vectorized_time = time.time() - start_time

    print("Predictions (Non-Vectorized):")
    print(predictions_non_vectorized)
    print("\nPredictions (Vectorized):")
    print(predictions_vectorized)

    print("\nEfficiency Comparison:")
    print(f"Non-Vectorized Time: {non_vectorized_time} seconds")
    print(f"Vectorized Time: {vectorized_time} seconds")

    # Assuming the vectorized implementation might produce a 2D array
    # Flatten if necessary for direct comparison
    if predictions_vectorized.ndim > 1:
        predictions_vectorized = predictions_vectorized.flatten()
    # Compare predictions (Optional, for demonstration)
    if np.allclose(predictions_non_vectorized, predictions_vectorized):
        print("Predictions are approximately equal.")
    else:
        print("Predictions differ significantly.")


# Example usage
# Define weights and biases (these should ideally be learned during training)
W1_tmp = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
b1_tmp = np.array([-9.82, -9.28, 0.96])
W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
b2_tmp = np.array([15.41])


# Example input data, already transposed
X_test = np.array([
    [200,13.9],  # Example 1
    [200, 17]  # Example 2
])

normalizer = CustomNormalization()
# Learn normalization parameters from training data
# normalizer.fit(X_train)
# Apply normalization to training data
# X_train_normalized = normalizer.fit_transform(X_train)
# Apply the same transformation to test data
X_test_normalized = normalizer.transform(X_test)

# Now, X_train_normalized and X_test_normalized are ready for use in training or prediction


# Predict using the model
predictions = predict(X_test_normalized, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
print(predictions)
# print("Normalized Test Data:", X_test_normalized)