# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np #used to include mathematics in code
import matplotlib.pyplot as plt # Used for plotting 
import pandas as pd  # import and manage datasets

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# using logistic Regression

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predict by testing the test set
y_pred=classifier.predict(X_test)

# testing the predicted y and the actual y using confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)