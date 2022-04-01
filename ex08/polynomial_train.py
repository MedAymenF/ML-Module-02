#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features

# Load the dataset
df = pd.read_csv("are_blue_pills_magics.csv")
x = np.array(df['Micrograms']).reshape(-1, 1)
y = np.array(df['Score']).reshape(-1, 1)

# Train six separate Linear Regression models with polynomial hypothesis
# with degrees ranging from 1 to 6
mse_list = []
print('Training the first model with polynomial degree 1.')
my_lr_1 = MyLR([1, -1], max_iter=100000, alpha=0.01)
my_lr_1.fit_(x, y)
predictions = my_lr_1.predict_(x)
mse = my_lr_1.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

print('\nTraining the second model with polynomial degree 2.')
my_lr_2 = MyLR(np.ones(3).reshape(-1, 1), max_iter=100000, alpha=0.001)
x_poly = add_polynomial_features(x, 2)
my_lr_2.fit_(x_poly, y)
predictions = my_lr_2.predict_(x_poly)
mse = my_lr_2.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

print('\nTraining the third model with polynomial degree 3.')
my_lr_3 = MyLR(np.ones(4).reshape(-1, 1), max_iter=100000, alpha=0.00001)
x_poly = add_polynomial_features(x, 3)
my_lr_3.fit_(x_poly, y)
predictions = my_lr_3.predict_(x_poly)
mse = my_lr_3.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

print('\nTraining the fourth model with polynomial degree 4.')
theta4 = np.array([[-20], [160], [-80], [10], [-1]])
my_lr_4 = MyLR(theta4, max_iter=100000, alpha=0.000001)
x_poly = add_polynomial_features(x, 4)
my_lr_4.fit_(x_poly, y)
predictions = my_lr_4.predict_(x_poly)
mse = my_lr_4.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

print('\nTraining the fifth model with polynomial degree 5.')
theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]])
my_lr_5 = MyLR(theta5, max_iter=100000, alpha=0.00000001)
x_poly = add_polynomial_features(x, 5)
my_lr_5.fit_(x_poly, y)
predictions = my_lr_5.predict_(x_poly)
mse = my_lr_5.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

print('\nTraining the sixth model with polynomial degree 6.')
theta6 = np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]])
my_lr_6 = MyLR(theta6, max_iter=100000, alpha=0.000000001)
x_poly = add_polynomial_features(x, 6)
my_lr_6.fit_(x_poly, y)
predictions = my_lr_6.predict_(x_poly)
mse = my_lr_6.mse_(y, predictions)
print('MSE =', mse)
mse_list.append(mse)

# Plot a bar plot showing the MSE score of the models in function of the
# polynomial degree of the hypothesis
plt.bar(np.arange(1, 7), mse_list)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.show()

# Plot a scatter plot of the true values
plt.ylabel('Score')
plt.scatter(x, y, label='True values', color='black')
continuous_x = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)

# Plot the 6 models
predictions_1 = my_lr_1.predict_(continuous_x)
plt.plot(continuous_x, predictions_1, label='Model 1')

x_poly = add_polynomial_features(continuous_x, 2)
predictions_2 = my_lr_2.predict_(x_poly)
plt.plot(continuous_x, predictions_2, label='Model 2')

x_poly = add_polynomial_features(continuous_x, 3)
predictions_3 = my_lr_3.predict_(x_poly)
plt.plot(continuous_x, predictions_3, label='Model 3')

x_poly = add_polynomial_features(continuous_x, 4)
predictions_4 = my_lr_4.predict_(x_poly)
plt.plot(continuous_x, predictions_4, label='Model 4')

x_poly = add_polynomial_features(continuous_x, 5)
predictions_5 = my_lr_5.predict_(x_poly)
plt.plot(continuous_x, predictions_5, label='Model 5')

x_poly = add_polynomial_features(continuous_x, 6)
predictions_6 = my_lr_6.predict_(x_poly)
plt.plot(continuous_x, predictions_6, label='Model 6')

plt.xlabel('Micrograms')
plt.legend()
plt.show()
