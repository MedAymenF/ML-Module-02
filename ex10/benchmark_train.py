#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

MAX_ITER = 10 ** 7
ALPHA = 1e-1


# Load the dataset
df = pd.read_csv("space_avocado.csv", index_col=0)
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape(-1, 1)


# Split the dataset into a training set and a test set
(x_train, x_test, y_train, y_test) = data_spliter(x, y, 0.8)


# Train four separate Linear Regression models with polynomial hypothesis
# with degrees ranging from 1 to 4
train_mse_list = []
test_mse_list = []

print('Training the first model with polynomial degree 1.')
my_lr_1 = MyLR(np.array([1, 1, 1, 0]).reshape(-1, 1),
               max_iter=MAX_ITER, alpha=ALPHA)
min1 = x_train.min(axis=0)
range1 = x_train.max(axis=0) - min1
x_train_normalized = (x_train - min1) / range1
my_lr_1.fit_(x_train_normalized, y_train)
predictions = my_lr_1.predict_(x_train_normalized)
mse = my_lr_1.mse_(y_train, predictions)
print(f'Training set MSE = {mse:e}')
train_mse_list.append(mse)

# Evaluate model on the test set
x_test_normalized = (x_test - min1) / range1
predictions = my_lr_1.predict_(x_test_normalized)
mse = my_lr_1.mse_(y_test, predictions)
print(f'Test set MSE = {mse:e}')
test_mse_list.append(mse)

print('\nTraining the second model with polynomial degree 2.')
my_lr_2 = MyLR(np.array([1, 1, 1, 1, 1, 0, 0]).reshape(-1, 1),
               max_iter=MAX_ITER, alpha=ALPHA)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)
                    for row in x_train.T])
min2 = x_poly.min(axis=0)
range2 = x_poly.max(axis=0) - min2
x_poly = (x_poly - min2) / range2
my_lr_2.fit_(x_poly, y_train)
predictions = my_lr_2.predict_(x_poly)
mse = my_lr_2.mse_(y_train, predictions)
print(f'Training set MSE = {mse:e}')
train_mse_list.append(mse)

# Evaluate model on the test set
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)
                    for row in x_test.T])
x_poly = (x_poly - min2) / range2
predictions = my_lr_2.predict_(x_poly)
mse = my_lr_2.mse_(y_test, predictions)
print(f'Test set MSE = {mse:e}')
test_mse_list.append(mse)

print('\nTraining the third model with polynomial degree 3.')
my_lr_3 = MyLR(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).reshape(-1, 1),
               max_iter=MAX_ITER, alpha=ALPHA)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 3)
                    for row in x_train.T])
min3 = x_poly.min(axis=0)
range3 = x_poly.max(axis=0) - min3
x_poly = (x_poly - min3) / range3
my_lr_3.fit_(x_poly, y_train)
predictions = my_lr_3.predict_(x_poly)
mse = my_lr_3.mse_(y_train, predictions)
print(f'Training set MSE = {mse:e}')
train_mse_list.append(mse)

# Evaluate model on the test set
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 3)
                    for row in x_test.T])
x_poly = (x_poly - min3) / range3
predictions = my_lr_3.predict_(x_poly)
mse = my_lr_3.mse_(y_test, predictions)
print(f'Test set MSE = {mse:e}')
test_mse_list.append(mse)

print('\nTraining the fourth model with polynomial degree 4.')
my_lr_4 = MyLR(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).reshape(-1,
                                                                         1),
               max_iter=MAX_ITER, alpha=ALPHA)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 4)
                    for row in x_train.T])
min4 = x_poly.min(axis=0)
range4 = x_poly.max(axis=0) - min4
x_poly = (x_poly - min4) / range4
my_lr_4.fit_(x_poly, y_train)
predictions = my_lr_4.predict_(x_poly)
mse = my_lr_4.mse_(y_train, predictions)
print(f'Training set MSE = {mse:e}')
train_mse_list.append(mse)

# Evaluate model on the test set
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 4)
                    for row in x_test.T])
x_poly = (x_poly - min4) / range4
predictions = my_lr_4.predict_(x_poly)
mse = my_lr_4.mse_(y_test, predictions)
print(f'Test set MSE = {mse:e}')
test_mse_list.append(mse)

# Save the parameters of the different models into a file
my_lr_1.thetas.resize(13, 1)
my_lr_2.thetas.resize(13, 1)
my_lr_3.thetas.resize(13, 1)
models = pd.DataFrame(
    np.hstack([my_lr_1.thetas, my_lr_2.thetas,
               my_lr_3.thetas, my_lr_4.thetas])).to_csv("models.csv")
