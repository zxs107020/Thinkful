#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:41:48 2018

@author: zxs107020
"""

# Import the required libraries
import os
import requests
import zipfile
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ml

# Check the working directory
os.getcwd()

# Set path to new directory
path = '/Users/zansadiq/Documents/Code/local/predict_subscriptions'

# Change wd
os.chdir(path)

# File location
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'

r = requests.get(url)

# Create the staging directory
# staging_dir = "staging"
# os.mkdir(staging_dir)

# Machine independent path to create files
zip_file = os.path.join(staging_dir, 'bank-additional.zip')

# Write the file to the computer
zf = open(zip_file,"wb")
zf.write(r.content)
zf.close()

# Unzip the files
z = zipfile.ZipFile(zip_file,"r")
z.extractall(staging_dir)
z.close()

# Extract the .csv's
files = glob.glob(os.path.join("staging/bank-additional" + "/*.csv"))

# Create an empty dictionary to hold the dataframes from csvs
dict_ = {}

# Write the files into the dictionary
for file in files:
    fname = os.path.basename(file)
    fname = fname.replace('.csv', '')
    dict_[fname] = pd.read_csv(file, header = 0, sep = ';')
    
# Extract the relevant dataframe
data = dict_['bank-additional-full']

# Check for missing values
data.isnull().sum()

# Check column types
data.dtypes

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