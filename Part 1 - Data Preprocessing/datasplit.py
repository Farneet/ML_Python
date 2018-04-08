\

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:03:18 2018

@author: INTEL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3:4].values

# Missing data

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values="NaN",strategy="mean", axis=0)
imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Catagorising  Data

# Independent variable
from sklearn.preprocessing import LabelEncoder
# will add values 0,1,2 for different counties that can lead to confusion
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])

# To avoid confusion ,will create 3 colums for diffrent countries
onehotencoder=OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#dependent variable
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)

# splitting in training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


