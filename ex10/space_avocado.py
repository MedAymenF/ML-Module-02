#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

# Load the dataset
df = pd.read_csv("space_avocado.csv", index_col=0)
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape(-1, 1)


# Split the dataset into a training set and a test set
(x_train, x_test, y_train, y_test) = data_spliter(x, y, 0.8)


# Train a model with a polynomial degree of 3
print('Training a model with a polynomial degree of 3.')
my_lr_3 = MyLR(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).reshape(-1, 1),
               max_iter=10 ** 7, alpha=1e-1)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 3)
                    for row in x_train.T])
min3 = x_poly.min(axis=0)
range3 = x_poly.max(axis=0) - min3
x_poly = (x_poly - min3) / range3
my_lr_3.fit_(x_poly, y_train)
train_predictions = my_lr_3.predict_(x_poly)
my_train_mse = my_lr_3.mse_(y_train, train_predictions)
print(f'Training set MSE = {my_train_mse:e}')


# Evaluate model on the test set
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 3)
                    for row in x_test.T])
x_poly = (x_poly - min3) / range3
test_predictions = my_lr_3.predict_(x_poly)
my_test_mse = my_lr_3.mse_(y_test, test_predictions)
print(f'Test set MSE = {my_test_mse:e}')


# Load models saved from benchmark_train.py
models = pd.read_csv("models.csv", index_col=0).to_numpy()
theta1 = models[:4, 0].reshape(-1, 1)
theta2 = models[:7, 1].reshape(-1, 1)
theta3 = models[:10, 2].reshape(-1, 1)
theta4 = models[:, 3].reshape(-1, 1)


# Evaluate three separate Linear Regression models with polynomial hypothesis
# with degrees 1, 2 and 4
train_mse_list = []
test_mse_list = []

print('\nEvaluating the first model with polynomial degree 1.')
my_lr_1 = MyLR(theta1)
min1 = x_train.min(axis=0)
range1 = x_train.max(axis=0) - min1
x_train_normalized = (x_train - min1) / range1
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

print('\nEvaluating the second model with polynomial degree 2.')
my_lr_2 = MyLR(theta2)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)
                    for row in x_train.T])
min2 = x_poly.min(axis=0)
range2 = x_poly.max(axis=0) - min2
x_poly = (x_poly - min2) / range2
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

print('\nEvaluating the third model with polynomial degree 4.')
my_lr_4 = MyLR(theta4)
x_poly = np.hstack([add_polynomial_features(row.reshape(-1, 1), 4)
                    for row in x_train.T])
min4 = x_poly.min(axis=0)
range4 = x_poly.max(axis=0) - min4
x_poly = (x_poly - min4) / range4
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


# Plot a bar plot showing the MSE score of the models in function of the
# polynomial degree of the hypothesis
train_mse_list.insert(2, my_train_mse)
test_mse_list.insert(2, my_test_mse)
x_axis_ticks = np.arange(1, 5)
plt.bar(x_axis_ticks - 0.1, train_mse_list,
        label='Training set MSE', width=0.2)
plt.bar(x_axis_ticks + 0.1, test_mse_list,
        label='Test set MSE', width=0.2)
plt.xticks(x_axis_ticks)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# Plot the true price and the predicted price obtained via the best model
# Plot `weight` on the x axis
data = np.vstack((x_train, x_test))
predictions = np.vstack((train_predictions, test_predictions))

x = np.array(df['weight']).reshape(-1, 1)
plt.scatter(x, y, label='True values')

x_ = data[:, 0].reshape(-1, 1)
plt.scatter(x_, predictions, label='Predictions')
plt.xlabel('weight (in ton)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()


# Plot `prod_distance` on the x axis
x = np.array(df['prod_distance']).reshape(-1, 1)
plt.scatter(x, y, label='True values')

x_ = data[:, 1].reshape(-1, 1)
plt.scatter(x_, predictions, label='Predictions')
plt.xlabel('prod_distance (in Mkm)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()


# Plot `time_delivery` on the x axis
x = np.array(df['time_delivery']).reshape(-1, 1)
plt.scatter(x, y, label='True values')

x_ = data[:, 2].reshape(-1, 1)
plt.scatter(x_, predictions, label='Predictions')
plt.xlabel('time_delivery (in days)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()
