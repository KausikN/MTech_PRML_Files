'''
Bayes Classifier
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from Dataset import *

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Main Functions
# Bayes Functions
def BayesClassifier_Run(train_data_pts, train_targets, test_data_pts, curCO=0):
    Predictions = []
    for i in tqdm(range(train_targets.shape[0])):
        curCO += 1
        print("For C" + str(curCO) + ": ")

        # X_train, X_test, y_train, y_test = train_test_split(train_data_pts, train_targets[i], test_size=0.1, random_state=0)
        X_train, X_test, y_train, y_test = train_data_pts, train_data_pts, train_targets[i], train_targets[i]

        classifier = GaussianNB()
        trained_classifier = classifier.fit(X_train, y_train)
        y_pred = trained_classifier.predict(X_test)

        # Print Results
        missclassified_count = (y_test != y_pred).sum()

        print("Missclassified:", missclassified_count, "/", X_test.shape[0])
        print("\n")
        
        # Predict on Test Dataset
        y_pred = trained_classifier.predict(test_data_pts)
        Predictions.append(y_pred)
        
    return Predictions, curCO


# Driver Code
# Params
train_dataset_path_1 = "Datasets/Dataset_1_Training.csv"
test_dataset_path_1 = "Datasets/Dataset_1_Testing.csv"
train_dataset_path_2 = "Datasets/Dataset_2_Training.csv"
test_dataset_path_2 = "Datasets/Dataset_2_Testing.csv"

preds_save_path = "Submissions/BayesClassifier/submissions_BayesClassifier.csv"
# Params

# RunCode
# Load Datasets
train_data_pts_1, train_targets_1, test_data_pts_1, train_data_pts_2, train_targets_2, test_data_pts_2 = LoadContestDatasets(train_dataset_path_1, test_dataset_path_1, train_dataset_path_2, test_dataset_path_2)

# Train
Predictions = []
curCO = 0

#### DATASET 1 ####
# Train Classifier
print("DATASET 1")
Predictions_1, curCO = BayesClassifier_Run(train_data_pts_1, train_targets_1, test_data_pts_1, curCO=curCO)
print()

#### DATASET 2 ####
# Train Classifier
print("DATASET 2")
Predictions_2, curCO = BayesClassifier_Run(train_data_pts_2, train_targets_2, test_data_pts_2, curCO=curCO)
print()
print()

Predictions = list(Predictions_1)
Predictions.extend(Predictions_2)

# Save Predictions
SaveContestPredictions(Predictions, preds_save_path)

# Check Difference with previous submission
old_preds_save_path = "Submissions/SVM/submission_SVM_L1Reg.csv"

print("Difference between", preds_save_path, "and", old_preds_save_path, "...")
CheckSubmissions_Difference(old_preds_save_path, preds_save_path)