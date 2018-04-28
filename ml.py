#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:01:16 2018

@author: zxs107020
"""

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