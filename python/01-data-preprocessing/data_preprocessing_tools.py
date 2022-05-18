# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print('---- x values ----')
print(x)
print('---- y values ----')
print(y)
print()

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print('---- x data inputed ----')
print(x)
print()

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print('---- Categorical x data encoded ----')
print(x)

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print('---- y labels encoded ----')
print(y)
print()

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print('---- Train x set ----')
print(x_train)
print('---- Test x set ----')
print(x_test)
print()
print('---- Train y set ----')
print(y_train)
print('---- Test y set ----')
print(y_test)
print()

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print('---- Train x scaled set ----')
print(x_train)
print('')
print('---- Train x scaled set ----')
print(x_test)
