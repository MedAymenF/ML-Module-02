#!/usr/bin/env python3
import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
Y = np.array(data[['Sell_price']])
my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], alpha=1e-5, max_iter=1200000)

# Example 0:
predictions = my_lreg.predict_(X)
print(my_lreg.mse_(Y, predictions))
# Output:
# 144044.877...

# Example 1:
my_lreg.fit_(X, Y)
print(repr(my_lreg.thetas))
# Output:
# array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

# Example 2:
predictions = my_lreg.predict_(X)
print(my_lreg.mse_(Y, predictions))
# Output:
# 586.896999...
