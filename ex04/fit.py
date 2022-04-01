import numpy as np


def predict_(x, theta):
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return x @ theta


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array,\
 without any for-loop.
The three arrays must have the compatible shapes.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
Return:
    The gradient as a numpy.array, a vector of shapes (n + 1) * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of expected type.
Raises:
    This function should not raise any Exception."""
    X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return X.T @ (predict_(x, theta) - y) / x.shape[0]


def fit_(x, y, theta, alpha, max_iter):
    """Description:
    Fits the model to the training dataset contained in x and y.
Args:
    x: has to be a numpy.array, a matrix of shape m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of shape m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during\
 the gradient descent
Return:
    new_theta: numpy.array, a vector of shape (number of features + 1, 1).
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    if not isinstance(alpha, (float, int)):
        print("alpha has to be a float.")
        return None
    if alpha <= 0:
        print("The learning rate has to be strictly positive.")
        return None
    if not isinstance(max_iter, int):
        print("max_iter has to be an int.")
        return None
    if max_iter < 0:
        print("The number of iterations has to be positive.")
        return None
    for _ in range(max_iter):
        grad = gradient(x, y, theta)
        theta = theta - alpha * grad
    return theta
