# Data Preprocessing Template

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)