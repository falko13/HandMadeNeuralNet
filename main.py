import numpy as np

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

def sequential(x, W1, b1, W2, b2):
    """
    Constructs and evaluates a simple 2-layer neural network model.

    Args:
        x (ndarray): The input data, shape (n,).
        W1, W2 (ndarray): Weight matrices for the first and second layers.
        b1, b2 (ndarray): Bias vectors for the first and second layers.

    Returns:
        ndarray: The output of the neural network.
    """
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    return a2

def predict(X, W1, b1, W2, b2):
    """
    Predicts the output for multiple inputs using the sequential neural network model.

    Args:
        X (ndarray): The input data, shape (m, n), for m examples.
        W1, W2 (ndarray): Weight matrices for the network.
        b1, b2 (ndarray): Bias vectors for the network.

    Returns:
        ndarray: The predicted outputs, shape (m, 1).
    """
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = sequential(X[i], W1, b1, W2, b2)
    return p


# Example usage
# Define weights and biases (these should ideally be learned during training)
W1_tmp = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
b1_tmp = np.array([-9.82, -9.28, 0.96])
W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
b2_tmp = np.array([15.41])

# Example input data
X_tst = np.array([
    [200, 13.9],  # Example 1
    [200, 17]  # Example 2
])



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