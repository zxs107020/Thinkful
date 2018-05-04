#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:01:16 2018

@author: zxs107020
"""

# Import the global libraries
import os
import pandas as pd

# Function for importing data from the internet
def dat_imp(wd, url, fn, path, header, sep, target):
    
    # Import the required libraries
    import requests
    import zipfile
    import glob
    
    # Grab the data
    r = requests.get(url)
    
    # Set the working directory
    os.chdir(wd)
    
    # Create a file for data storage
    staging_dir = "staging"
    
    # Conditional execution of directory creation
    try:
        os.mkdir(staging_dir)
    except OSError:
        pass
    
    # Machine independent path to create files
    zip_file = os.path.join(staging_dir, fn)

    # Write the file to the computer
    zf = open(zip_file,"wb")
    zf.write(r.content)
    zf.close()

    # Unzip the files
    z = zipfile.ZipFile(zip_file,"r")
    z.extractall(staging_dir)
    z.close()

    # Extract the .csv's
    files = glob.glob(os.path.join(path + "/*.csv"))

    # Create an empty dictionary to hold the dataframes from csvs
    dict_ = {}

    # Write the files into the dictionary
    for file in files:
        fname = os.path.basename(file)
        fname = fname.replace('.csv', '')
        dict_[fname] = pd.read_csv(file, header = header, sep = sep)
    
    # Extract the dataframes
    data = dict_[target]
    
    return data

# Function for loading a csv that exists locally
def local_import(wd, fn):

    # Import library
    import csv
    
    # Set the directory
    os.chdir(wd)
    
    # Initialize a list to store the data
    data = list() 
    
    # Load the data
    with open(fn) as dat:
        csvReader = csv.reader(dat)
        for row in csvReader:
            data.append(row)
    
    # Convert to Pandas
    data = pd.DataFrame(data)
    
    # Fix the headers
    data.columns = data.iloc[0]
    
    # Delete the initial header row
    data = data.reindex(data.index.drop(0))
    
    # Output 
    return data
    
# Function for splitting data into training, validation, and testing sets
def separate(data, target, size):    

    # Import the required libraries
    from sklearn.cross_validation import train_test_split
    
    # List all of the column headers
    variables = data.columns.values.tolist()

    # Select independent variables
    x = [i for i in variables if i not in [target]]

    # Fill the values and select the dependent variable
    x = data[x]
    y = data[target]

    # Split the data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = size, random_state = 100)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.5, random_state = 100)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

# Function to run any machine learning algorithm
def machine_learning(package, x, y, x1, y1, model):
    
    # Import required libraries
    exec(package)
    from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
    
    # Model
    mod = eval(model)
    
    mod_fit = mod.fit(x, y)
    
    # Predict
    preds = mod_fit.predict(x1)
    
    # Results
    conf_mat = confusion_matrix(y1, preds)

    class_rep = classification_report(y1, preds)

    acc = accuracy_score(y1, preds)
    
    # Output
    return preds, conf_mat, class_rep, acc

# Function to visualize the results
def visualize(target, preds):

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # Visualize ROC
    fpr, tpr, _ = roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Predictions')
    plt.legend(loc = "lower right")
    plt.show()