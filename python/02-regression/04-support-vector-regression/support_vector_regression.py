# Support Vector Regression (SVR)

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
print(x)
print(y)
print()
y = y.reshape(len(y), 1)
print(y)
print()

# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
print(x)
print(y)
print()

# Training the SVR model on the whole dataset

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predicting a new result
# reshape used to avoid ValueError
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(1, 1))

# Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
# reshape used to avoid ValueError
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results (for higher resolution and smoother curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
# reshape used to avoid ValueError
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
