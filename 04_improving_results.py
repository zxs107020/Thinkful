#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:28:32 2018

@author: zxs107020
"""

# Import the required libraries
import ml
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the variables
wd = '/Users/zansadiq/Documents/Code/local/predict_subscriptions'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
fn = 'bank-additional.zip'
path = 'staging/bank-additional'
header = 0
sep = ';'
target = 'bank-additional-full'

# Load the data
data = ml.dat_imp(wd, url, fn, path, header, sep, target)

# Select object columns
cols = data.columns[data.dtypes.eq('object')]

# Convert to categories
for col in cols:
    data[col] = data[col].astype('category')
    
# Initialize label encoder
le = LabelEncoder()

# Encode categories numerically
for col in cols:
    data[col] = le.fit_transform(data[col]) 

# Split the data into training, validation, and testing sets using `ml`
x_train, x_val, x_test, y_train, y_val, y_test = ml.separate(data, 'y', .3)
    
# Feature selection
clf = RandomForestClassifier()

rfe = RFECV(estimator = clf, step = 1, cv = 5, scoring = 'accuracy')

rfecv = rfe.fit(x_train, y_train)

# Results
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Eliminate the unnecessary variables
keep = x_train.columns[rfecv.support_]
x_train = x_train[keep]
x_val = x_val[keep]
x_test = x_test[keep]

# Test the performance

# Define the library and function
lib = 'from sklearn.linear_model import LogisticRegression'
func = 'LogisticRegression()'

# Perform the logistic regression using `ml`
preds, conf_mat, class_rep, acc = ml.machine_learning(lib, x_train, y_train, x_val, y_val, func)

# Check the results
print('Confusion Matrix for the Logistic Regression Model', '\n', conf_mat, '\n')
print('Accuracy for the Logistic Regression Model', acc, '\n')
print('Summary statistics for Logistic Regression Model', '\n', class_rep, '\n')

# Visualize the results with an ROC curve
ml.visualize(y_val, preds)

# Principal Component Analysis

# Resplit the data
x_train, x_val, x_test, y_train, y_val, y_test = ml.separate(data, 'y', .3)

# Initialize the model
pca = PCA(.95)

fit = pca.fit(x_train)

# Find the number of selected transformations
print('The PCA model chose', pca.n_components_, 'components', '\n')

# Apply the transformations to the data
x_train = pca.transform(x_train)
x_val = pca.transform(x_val)

# Run the logistic regression again
preds, conf_mat, class_rep, acc = ml.machine_learning(lib, x_train, y_train, x_val, y_val, func)

# Check the results
print('Confusion Matrix for the Logistic Regression Model', '\n', conf_mat, '\n')
print('Accuracy for the Logistic Regression Model', acc, '\n')
print('Summary statistics for Logistic Regression Model', '\n', class_rep, '\n')

# Visualize the results with an ROC curve
ml.visualize(y_val, preds)
