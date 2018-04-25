#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:17:03 2018

@author: zxs107020
"""

# Import the required libraries
import os
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve, auc

# Find the working directory
print(os.getcwd())

# Define a new directory
path = 'path/to/files'

# Change the working directory
# os.chdir(path)

# Create an empty list to store the data
train = list()

# Load the data from a local file using the csv module
with open('train.csv') as titanic_train:
    csvReader = csv.reader(titanic_train)
    for row in csvReader:
        train.append(row)
        
# Convert the training list to a dataframe
train = pd.DataFrame(train)

# Load the test data directly from a url
test = pd.read_csv('https://www.kaggle.com/c/3136/download/test.csv', error_bad_lines = False)

# Reload the test data using csv
test = list()

with open('test.csv') as titanic_test:
    csvReader = csv.reader(titanic_test)
    for row in csvReader:
        test.append(row)
        
# Convert to data frame
test = pd.DataFrame(test)

# Fix the headers
train.columns = train.iloc[0]

test.columns = test.iloc[0]

# Delete the row
train = train.reindex(train.index.drop(0))

test = test.reindex(test.index.drop(0))

# Drop unnecessary columns
train = train.drop(columns = ['Name', 'Ticket', 'Cabin'], axis = 1)

test = test.drop(columns = ['Name', 'Ticket', 'Cabin'], axis = 1)

# Convert factor variables
train['Survived'] = train['Survived'].astype('category')
train['Pclass'] = train['Pclass'].astype('category')
train['Sex'] = train['Sex'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')

test['Pclass'] = test['Pclass'].astype('category')
test['Sex'] = test['Sex'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')

# Convert remaining variables to numeric
train['PassengerId'] = pd.to_numeric(train['PassengerId'])
train['Age'] = pd.to_numeric(train['Age'])
train['SibSp'] = pd.to_numeric(train['SibSp'])
train['Parch'] = pd.to_numeric(train['Parch'])
train['Fare'] = pd.to_numeric(train['Fare'])

test['PassengerId'] = pd.to_numeric(test['PassengerId'])
test['Age'] = pd.to_numeric(test['Age'])
test['SibSp'] = pd.to_numeric(test['SibSp'])
test['Parch'] = pd.to_numeric(test['Parch'])
test['Fare'] = pd.to_numeric(test['Fare'])

# Handle missing values

# Calculate average fare
test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean()

# Fill the missing age cells with the average
train['Age'].fillna(train['Age'].mean(), inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

# Convert the embarked column
lb = LabelEncoder()

train['Embarked'] = lb.fit_transform(train['Embarked'])
test['Embarked'] = lb.fit_transform(test['Embarked'])

train['Sex'] = lb.fit_transform(train['Sex'])
test['Sex'] = lb.fit_transform(test['Sex'])

# List all of the column headers
train_vars = train.columns.values.tolist()

# Select independent variables
x_train = [i for i in train_vars if i not in ['Survived']]

# Fill the values and select the dependent variable
x = train[x_train]
y = train['Survived']

# Convert everything to numbers
x = x.apply(pd.to_numeric)
y = y.apply(pd.to_numeric)

# Split the data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 100)

# Create a Decision Tree
my_tree = DecisionTreeClassifier()
tree_fit = my_tree.fit(x_train, y_train)

# Create a Random Forest
rf = RandomForestClassifier()
rf_fit = rf.fit(x_train, y_train)

# Create a Logistic Regression
lr = LogisticRegression()
lr_fit = lr.fit(x_train, y_train)

# Predict
tree_preds = tree_fit.predict(x_val)
rf_preds = rf_fit.predict(x_val)
lr_preds = lr.predict(x_val)

# Accuracy
print("Validation decision tree accuracy:", accuracy_score(y_val, tree_preds))
print("Validation random forest accuracy:", accuracy_score(y_val, rf_preds))
print("Validation logistic regression accuracy:", accuracy_score(y_val, lr_preds))

# ROC Curve: Decision Tree
fpr, tpr, _ = roc_curve(y_val, tree_preds)
tree_roc_auc = auc(fpr, tpr)
print("The AUC for the Decision Tree is", tree_roc_auc)

# ROC Curve: Random Forest
fpr, tpr, _ = roc_curve(y_val, rf_preds)
rf_roc_auc = auc(fpr, tpr)
print("The AUC for the Random Forest is", rf_roc_auc)

# ROC Curve: Logistic Regression
fpr, tpr, _ = roc_curve(y_val, lr_preds)
lr_roc_auc = auc(fpr, tpr)
print("The AUC for the Logistic Regression is", lr_roc_auc)

# Fill a new column in the test data for the predictions
test['Survived'] = rf_fit.predict(test)

# Create a new df for the results
out = pd.DataFrame(test['PassengerId'])
out['Survived'] = test['Survived']

# Use pandas to create the file
out.to_csv('kaggle_submission.csv')

# Define the file name and header
csv_header = 'PassengerId, Survived' 
file_name = 'kaggle_submission.csv'

# Function to create the csv
def print_results(file_name, csv_header, data): 
    with open(file_name,'wt') as f:
        print(csv_header, file = f) 
        for s in data:
            print(','.join(s), file = f)

# Call function
print_results(file_name, csv_header, out)
