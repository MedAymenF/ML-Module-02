import numpy as np


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
Return:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    m = x.shape[0]
    y_hat = np.full((m, 1), theta[0])
    for i in range(m):
        y_hat[i] = y_hat[i] + x[i] @ theta[1:, :]
    return y_hat
