import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for the training set
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Add a column of ones to X for the bias term
X = np.concatenate((np.ones((100, 1)), X), axis=1)

# Calculate the weight vector using the normal equation
w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Make predictions using the weight vector
X_new = np.array([[1, 0], [1, 1]]) # Two new examples
y_pred = X_new.dot(w)

# Plot the data and the predictions
plt.scatter(X[:, 1], y)
plt.plot(X_new[:, 1], y_pred, 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
