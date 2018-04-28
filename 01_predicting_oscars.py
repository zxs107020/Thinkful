#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:49:50 2018

@author: zxs107020
"""

# Import the required libraries
import requests
import os
import zipfile
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# File location
url = 'https://www.dropbox.com/s/shg31hm4voydqnl/Thinkful%20Workshops%20-%20Predicting%20the%20Oscars.zip?dl=1'

r = requests.get(url)

# Create the staging directory
staging_dir = "staging"
os.mkdir(staging_dir)

# Confirm the staging directory path
os.path.isdir(staging_dir)

# Machine independent path to create files
zip_file = os.path.join(staging_dir, "Thinkful Workshops - Predicting the Oscars.zip")

# Write the file to the computer
zf = open(zip_file,"wb")
zf.write(r.content)
zf.close()

# Unzip the files
z = zipfile.ZipFile(zip_file,"r")
z.extractall(staging_dir)
z.close()

# Extract the .csv's
files = glob.glob(os.path.join("staging/oscars" + "/*.csv"))

# Create an empty dictionary to hold the dataframes from csvs
dict_ = {}

# Write the files into the dictionary
for file in files:
    fname = os.path.basename(file)
    fname = fname.replace('.csv', '')
    dict_[fname] = pd.read_csv(file, header = 0).fillna('')
    
# Extract the dataframes
train = dict_['train']
test = dict_['test']

# Convert target variable to factor
train['Won?'] = train['Won?'].astype('category')
test['Won?'] = test['Won?'].astype('category')

# Fix the genres
#train['Genres'] = train['Genres'].str.split(",")
#test['Genres'] = test['Genres'].str.split(",")

# Fix the ratings
train.loc[train["Rate"] == "G", "Rate"] = 1
train.loc[train["Rate"] == "PG", "Rate"] = 2
train.loc[train["Rate"] == "PG-13", "Rate"] = 3
train.loc[train["Rate"] == "R", "Rate"] = 4

test.loc[test["Rate"] == "G", "Rate"] = 1
test.loc[test["Rate"] == "PG", "Rate"] = 2
test.loc[test["Rate"] == "PG-13", "Rate"] = 3
test.loc[test["Rate"] == "R", "Rate"] = 4

# Handle missing values
train.isnull().sum()
test.isnull().sum()

train = train.drop(columns = ['Opening Weekend'])
test = test.drop(columns = ['Opening Weekend'])

# List all of the column headers
train_vars = train.columns.values.tolist()
test_vars = test.columns.values.tolist()

# Select independent variables
x_train = [i for i in train_vars if i not in ['Won?']]
x_test = [i for i in test_vars if i not in ['Won?']]

# Fill the values and select the dependent variable
x = train[x_train]
y = train['Won?']

x_test = test[x_test]
y_test = test['Won?']

# Inspect and convert variables
x.dtypes

# Fix the genres
lb = LabelEncoder()
x['Genres'] = lb.fit_transform(x['Genres'])
x_test['Genres'] = lb.fit_transform(x_test['Genres'])

# Fix the movie titles
x['Movie'] = lb.fit_transform(x['Movie'])
x_test['Movie'] = lb.fit_transform(x_test['Movie'])

# Remove the IMDB id
x = x.drop(columns = 'IMdB id')
x_test = x_test.drop(columns = 'IMdB id')

# Fill blank cells with 0's
x.replace(r'^\s*$', 0, regex = True, inplace = True)
x_test.replace(r'^\s*$', 0, regex = True, inplace = True)

# Convert remaining object variables
cols = x.columns[x.dtypes.eq('object')]
test_cols = x_test.columns[x_test.dtypes.eq('object')]

x[cols] = x[cols].apply(pd.to_numeric)
x_test[test_cols] = x_test[test_cols].apply(pd.to_numeric)

x.isnull().sum()

# Create a validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 100)

# Decision Tree
d_tree = DecisionTreeClassifier()
tree_fit = d_tree.fit(x_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf_fit = rf.fit(x_train, y_train)

# Predict
tree_val_preds = tree_fit.predict(x_val)

rf_val_preds = rf_fit.predict(x_val)

# Accuracy
print("Validation decision tree accuracy:", accuracy_score(y_val, tree_val_preds))
print("Validation random forest accuracy:", accuracy_score(y_val, rf_val_preds))

# ROC Curve

# Decision Tree
fpr, tpr, _ = roc_curve(y_val, tree_val_preds)
tree_roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC Curve (area = %0.2f)' % tree_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Decision Tree')
plt.legend(loc="lower right")
plt.show()

# Random Forest
fpr, tpr, _ = roc_curve(y_val, rf_val_preds)
tree_roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC Curve (area = %0.2f)' % tree_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Random Forest')
plt.legend(loc="lower right")
plt.show()

# Final Predictions
tree_test_preds = tree_fit.predict(x_test)

# Add predictions to data
x_test['Won_Preds'] = tree_test_preds

# Save output
x_test.to_csv('oscar_test_preds.csv')

