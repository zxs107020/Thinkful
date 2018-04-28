#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:01:24 2018

@author: zxs107020
"""

# Import the required libraries
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics, naive_bayes
from sklearn.ensemble import RandomForestClassifier

# Check the working directory
os.getcwd()

# Set the directory
path = '/Users/zansadiq/Documents/Work/Thinkful/Code'

os.chdir(path)

# Access the data
data = pd.read_csv('https://github.com/Thinkful-Ed/data-201-resources/raw/master/hotel-reviews.csv')

# Convert to lower-case
data['reviews.text'] = data['reviews.text'].str.lower()

# Remove punctuation
data["words"] = data['reviews.text'].str.replace('[^\w\s]','')

# Tokenize words
data['words'] = data['reviews.text'].apply(str)

data['words'] = data['words'].apply(lambda row: word_tokenize(row))

# Filter out stop words
stop_words = set(stopwords.words('english'))

data['words'] = data['words'].apply(lambda x: [item for item in x if item not in stop_words])

# 'Stemming' the words
ps = PorterStemmer()

data['stemmed'] = data['words'].apply(lambda x: [ps.stem(y) for y in x])

# Rejoin the words
data['stemmed'] = data['stemmed'].apply(lambda x: ' '.join(x))

# Separate features and target
df = pd.DataFrame()

df['x'] = data['stemmed']

df['y'] = data['name']

# Convert the hotel names to numbers for model
encoder = preprocessing.LabelEncoder()

df['y1'] = encoder.fit_transform(df['y'])

# Create count vectors for each review
cv = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')

# Apply counter to data
x = cv.fit_transform(df['x'])

x = pd.DataFrame(x.toarray(), columns = cv.get_feature_names())

# TF-IDF
tf_idf = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', max_features = 5000)

x_tf_idf = tf_idf.fit_transform(df['x'])

x_tf_idf = pd.DataFrame(x_tf_idf.toarray(), columns = tf_idf.get_feature_names())

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = model_selection.train_test_split(x, df['y1'])

tf_train_x, tf_test_x, tf_train_y, tf_test_y = model_selection.train_test_split(x_tf_idf, df['y1'])

# Build the model
def build_model(model, x_training, y_training, x_testing, y_testing):
    
    # Fit the model on the training data
    model.fit(x_training, y_training)
    
    # Predictions
    preds = model.predict(x_testing)
    
    # Output 
    return(metrics.accuracy_score(preds, y_testing))
    
# Run the function to create a Random Forest
rf_count_acc = build_model(RandomForestClassifier(), train_x, train_y, test_x, test_y)

rf_tf_acc = build_model(RandomForestClassifier(), tf_train_x, tf_train_y, tf_test_x, tf_test_y)

# Run the function to create a Naive Bayes 
nb_count_acc = build_model(naive_bayes.MultinomialNB(), train_x, train_y, test_x, test_y)

nb_tf_acc = build_model(naive_bayes.MultinomialNB(), tf_train_x, tf_train_y, tf_test_x, tf_test_y)
