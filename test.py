import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import the model
rfc = pickle.loads("rfc.pickle")

# Load the dataset for testing
test_df = pd.read_csv("test_data.tsv", sep='\t')

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