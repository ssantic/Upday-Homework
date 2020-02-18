#!/usr/bin/env python

"""
Author: Srdjan Santic
Date: 18-Feb-2020

This script is used for evaluating an already trained model on the test data,
both provided by the train.py script. It loads the pickled model, and outputs
a number of accuracy metrics for the model.

The correct usage of the script is:

$ python test.py model_file.pickle test_data_file.tsv

"""


import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report

# Prepare the argument parser
parser = argparse.ArgumentParser(description='Script for training a Random Forest classifier')
parser.add_argument('test_file', type=str, help='Input TSV file with test data.')
parser.add_argument('model', type=str, help='Input pickled model for scoring.')
args = parser.parse_args()

# Import the model
rfc = pickle.load(open(args.model, 'rb'))

# Load the dataset for testing
test_df = pd.read_csv(args.test_file, sep='\t')

# Split the dataset into features and labels
y_test = test_df['test_labels']
X_test = test_df.drop('test_labels', axis=1)

# Generate the predictions
rfc_preds = rfc.predict(X_test)

# Generate the accuracy metrics
accuracy = accuracy_score(y_test, rfc_preds)
balanced_accuracy = balanced_accuracy_score(y_test, rfc_preds)
precision = precision_score(y_test, rfc_preds, average='weighted')
recall = recall_score(y_test, rfc_preds, average='weighted')
f1 = f1_score(y_test, rfc_preds, average='weighted')

# Output the accuracy metrics
print("Overall Accuracy is {}".format(accuracy))
print("Overall Balanced Accuracy is {}".format(balanced_accuracy))
print("Overall Precision (weighted) is {}".format(precision))
print("Overall Recall (wighted) is {}".format(recall))
print("Overall F1 Score (weighted) is {}". format(f1))

# Output the classification report
print(classification_report(y_test, rfc_preds))
