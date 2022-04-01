#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("spacecraft_data.csv")
y = np.array(data[['Sell_price']])

# Univariate linear regression using the `Age` feature
feature = 'Age'
x1 = np.array(data[[feature]])
myLR_age = MyLR([[1000.0], [-1.0]], alpha=1e-3, max_iter=100000)
myLR_age.fit_(x1, y)
predictions = myLR_age.predict_(x1)

# Plot a scatter plot of the actual prices and another of our predictions
plt.grid()
plt.scatter(x1, y, label='Sell price', color='midnightblue')
plt.scatter(x1, predictions, label='Predicted sell price', s=6)
plt.xlabel(f'x1: {feature} (in years)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()

# Print the final value of theta and the mean squared error
print(f'{feature}:')
print('Final thetas =', myLR_age.thetas)
print('MSE=', myLR_age.mse_(y, predictions))


# Univariate linear regression using the `Thrust_power` feature
feature = 'Thrust_power'
x2 = np.array(data[[feature]])
myLR_age = MyLR([[0], [1.0]], alpha=1e-4, max_iter=100000)
myLR_age.fit_(x2, y)
predictions = myLR_age.predict_(x2)

# Plot a scatter plot of the actual prices and another of our predictions
plt.grid()
plt.scatter(x2, y, label='Sell price', color='green')
plt.scatter(x2, predictions, label='Predicted sell price', s=6)
plt.xlabel(f'x2: {feature} (in 10Km/s)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()

# Print the final value of theta and the mean squared error
print(f'{feature}:')
print('Final thetas =', myLR_age.thetas)
print('MSE=', myLR_age.mse_(y, predictions))


# Univariate linear regression using the `Terameters` feature
feature = 'Terameters'
x3 = np.array(data[[feature]])
myLR_age = MyLR([[1000], [-1.0]], alpha=1e-4, max_iter=200000)
myLR_age.fit_(x3, y)
predictions = myLR_age.predict_(x3)

# Plot a scatter plot of the actual prices and another of our predictions
plt.grid()
plt.scatter(x3, y, label='Sell price', color='purple')
plt.scatter(x3, predictions, label='Predicted sell price', s=6)
plt.xlabel('x3: Distance travelled (in Tmeters)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()

# Print the final value of theta and the mean squared error
print(f'{feature}:')
print('Final thetas =', myLR_age.thetas)
print('MSE=', myLR_age.mse_(y, predictions))


# Multivariate linear regression using all three features
X = np.hstack((x1, x2, x3))
my_lreg = MyLR([1.0, -1.0, 1.0, -1.0], alpha=1e-5, max_iter=1500000)
my_lreg.fit_(X, y)
predictions = my_lreg.predict_(X)

# Print the final value of theta and the mean squared error
print('\nMultivariate linear regression:')
print('Final thetas=', my_lreg.thetas)
print('MSE=', my_lreg.mse_(y, predictions))

# Plot a scatter plot of the actual prices and another of our predictions
# using the `Age` feature
plt.grid()
plt.scatter(x1, y, label='Sell price', color='midnightblue')
plt.scatter(x1, predictions, label='Predicted sell price', s=6)
plt.xlabel(f'x1: age (in years)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()

# Plot a scatter plot of the actual prices and another of our predictions
# using the `Thrust_power` feature
plt.grid()
plt.scatter(x2, y, label='Sell price', color='green')
plt.scatter(x2, predictions, label='Predicted sell price', s=6)
plt.xlabel(f'x2: thrust power (in 10Km/s)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()

# Plot a scatter plot of the actual prices and another of our predictions
# using the `Terameters` feature
plt.grid()
plt.scatter(x3, y, label='Sell price', color='purple')
plt.scatter(x3, predictions, label='Predicted sell price', s=6)
plt.xlabel('x3: Distance travelled (in Tmeters)')
plt.ylabel('y: sell price (in keuros)')
plt.legend()
plt.show()
