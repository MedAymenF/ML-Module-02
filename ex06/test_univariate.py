#!/usr/bin/env python3
import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])

myLR_age = MyLR([[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
myLR_age.fit_(X, Y)
predictions = myLR_age.predict_(X)
print(myLR_age.mse_(Y, predictions))
# Output
# 57636.77729...
