#!/usr/bin/env python

"""
Author: Srdjan Santic
Date: 18-Feb-2020

This script is used for evaluating an already trained model on the test data,
both provided by the train.py script. It loads the pickled model, and outputs
a number of accuracy metrics for the model.

The correct usage of the script is:

$ python test.py test_data_features.tsv test_data_labels.tsv model_file.pickle

"""


import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report

# Prepare the argument parser
parser = argparse.ArgumentParser(description='Script for training a Random Forest classifier')
parser.add_argument('test_features', type=str, help='Input TSV file with test features.')
parser.add_argument('test_labels', type=str, help='Input TSV file with test labels.')
parser.add_argument('model', type=str, help='Input pickled model for scoring.')
args = parser.parse_args()

# Import the model
print("Loading model...")
rfc = pickle.load(open(args.model, 'rb'))

# Load the features for testing
print("Loading test dataset features...")
X_test = pd.read_csv(args.test_features, sep='\t', low_memory=False)

# Load the labels for testing
print("Loading test dataset labels...")
y_test = pd.read_csv(args.test_labels, sep='\t', low_memory=False)

# Generate the predictions
print("Generating predictions...")
rfc_preds = rfc.predict(X_test)

# Generate the accuracy metrics
print("Calculating accuracy metrics...")
accuracy = accuracy_score(y_test, rfc_preds)
balanced_accuracy = balanced_accuracy_score(y_test, rfc_preds)
precision = precision_score(y_test, rfc_preds, average='weighted')
recall = recall_score(y_test, rfc_preds, average='weighted')
f1 = f1_score(y_test, rfc_preds, average='weighted')

# Output the accuracy metrics
print("")
print("")
print("Overall Accuracy is {}".format(accuracy))
print("Overall Balanced Accuracy is {}".format(balanced_accuracy))
print("Overall Precision (weighted) is {}".format(precision))
print("Overall Recall (wighted) is {}".format(recall))
print("Overall F1 Score (weighted) is {}". format(f1))
print("")
print("")

# Output the classification report
print(classification_report(y_test, rfc_preds))
