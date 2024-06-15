import jax.numpy as np

def relu(X):
    return X * (X >= 0.0)


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))