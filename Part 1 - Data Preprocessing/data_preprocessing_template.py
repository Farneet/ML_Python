# Data Preprocessing Template

# Importing the libraries
import numpy as np #used to include mathematics in code
import matplotlib.pyplot as plt # Used for plotting 
import pandas as pd  # import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Removed missing data as we doont need it for all the problems



#REmoved code for Encoding categorical data as we dont need for all the problems


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# commented code for Feature Scaling as we may need it further.
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''
