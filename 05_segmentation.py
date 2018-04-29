#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:16:35 2018

@author: zxs107020
"""

# Import the required libraries
import ml
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the variables
wd = '/path/to/files'
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
    
# Plot
sns.pairplot(x_vars = ['age'], y_vars = ['duration'], data = data, hue = 'y')

# Split the data into training, validation, and testing sets using `ml`
x_train, x_val, x_test, y_train, y_val, y_test = ml.separate(data, 'y', .3)

# Initialize the scaler
s = StandardScaler()
s.fit(x_train)

# Transform the data
x_train_scaled = s.transform(x_train)
x_val_scaled = s.transform(x_val)
x_test_scaled = s.transform(x_test)

# Setup the KNN
lib = 'from sklearn.neighbors import KNeighborsClassifier'
func = 'KNeighborsClassifier(n_neighbors = 2)'

# Run the algorithm
preds, conf_mat, class_rep, acc = ml.machine_learning(lib, x_train_scaled, y_train, x_val_scaled, y_val, func)

# Initialize a list to store results
error = []

# Vary the K values
for i in range(1, 10):
    string = ''.join(['KNeighborsClassifier(n_neighbors =', str(i), ')'])
    
    preds_i, conf_mat, class_rep, acc = ml.machine_learning(lib, x_train_scaled, y_train, x_val_scaled, y_val, string)
    
    error.append(np.mean(preds_i != y_val))

# Plot the errors
plt.plot(range(1, 10), error, color = 'blue', marker = 'o', markerfacecolor = 'blue')
plt.title('Error Rate for Different Values of K')
plt.xlabel('N')
plt.ylabel('Error')

# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 8)

# Re-scale the entire dataset
s.fit(data.loc[:, data.columns != 'y'])

# Transform
data_scaled = s.transform(data.loc[:, data.columns != 'y'])

# Run the algorithm
kmeans = kmeans.fit(data_scaled)

# Grab the labels
clusters = kmeans.predict(data_scaled)

# Assign values
data['cluster'] = clusters

# Plot
sns.pairplot(x_vars = ['age'], y_vars = ['duration'], data = pd.DataFrame(data), hue = 'cluster')
