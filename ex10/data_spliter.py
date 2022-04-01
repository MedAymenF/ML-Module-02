import numpy as np
from numpy.random import default_rng


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)\
 into a training and a test set,
while respecting the given proportion of examples to be kept\
 in the training set.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset\
 that will be assigned to the
    training set.
Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
Raises:
    This function should not raise any Exception.
"""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    if not isinstance(proportion, (int, float)):
        print('proportion has to be a float.')
        return None
    if proportion < 0 or proportion > 1:
        print('proportion has to be between 0 and 1.')
        return None
    rng = default_rng(1337)
    z = np.hstack((x, y))
    rng.shuffle(z)
    x, y = z[:, :-1].reshape(x.shape), z[:, -1].reshape(y.shape)
    idx = int((x.shape[0] * proportion))
    x_train, x_test = np.split(x, [idx])
    y_train, y_test = np.split(y, [idx])
    return (x_train, x_test, y_train, y_test)
