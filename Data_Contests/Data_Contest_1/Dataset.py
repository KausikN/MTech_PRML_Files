'''
Read and Visualise the given Dataset
'''

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Main Functions
# Dataset Functions
def ReadDataset(path):
    '''
    Read the given dataset
    '''
    dataset = pd.read_csv(path)
    return dataset

def SaveDataset(dataset, path):
    '''
    Save the given dataset
    '''
    dataset.to_csv(path, index=False)

def FormatDataset(dataset, targetRowCount=0):
    '''
    Format the given dataset
    '''

    data_pts = []
    targets = []
    if targetRowCount == 0:
        data_pts = dataset.iloc[:, 1:].values.T
        targets = np.array([])
    else:
        data_pts = dataset.iloc[:-targetRowCount, 1:].values.T
        targets = dataset.iloc[-targetRowCount:, 1:].values

    print("Data points: ", data_pts.shape)
    print("Targets: ", targets.shape)

    return data_pts, targets

# Specific Functions
def LoadContestDatasets(train_dataset_path_1, test_dataset_path_1, train_dataset_path_2, test_dataset_path_2):
    '''
    Load Contest Datasets
    '''
    # Dataset 1
    print("Train Dataset 1")
    train_dataset_1 = ReadDataset(train_dataset_path_1)
    train_data_pts_1, train_targets_1 = FormatDataset(train_dataset_1, targetRowCount=2)
    print("Test Dataset 1")
    test_dataset_1 = ReadDataset(test_dataset_path_1)
    test_data_pts_1, _ = FormatDataset(test_dataset_1, targetRowCount=0)

    # Dataset 2
    print("Train Dataset 2")
    train_dataset_2 = ReadDataset(train_dataset_path_2)
    train_data_pts_2, train_targets_2 = FormatDataset(train_dataset_2, targetRowCount=4)
    print("Test Dataset 2")
    test_dataset_2 = ReadDataset(test_dataset_path_2)
    test_data_pts_2, _ = FormatDataset(test_dataset_2, targetRowCount=0)

    return train_data_pts_1, train_targets_1, test_data_pts_1, train_data_pts_2, train_targets_2, test_data_pts_2

def SaveContestPredictions(Predictions, preds_save_path):
    '''
    Save Predictions
    '''
    # Append all predictions to single list
    Predictions_Appended = []
    for preds in Predictions:
        Predictions_Appended.extend(preds)
    Predictions_Appended = [int(round(pred, 0)) for pred in Predictions_Appended]
    print("Predictions Count:", len(Predictions_Appended))

    # Form a dataframe
    indices = list(range(len(Predictions_Appended)))
    preds_data = np.dstack((indices, Predictions_Appended))[0]
    Predictions_df = pd.DataFrame(data=preds_data, columns=["Id", "Predicted"])
    SaveDataset(Predictions_df, preds_save_path)

# Visualisation Functions
def CheckSubmissions_Difference(pred_path_1, pred_path_2):
    '''
    Check the difference between two submissions
    '''
    # Load Predictions
    preds_1 = ReadDataset(pred_path_1)
    preds_2 = ReadDataset(pred_path_2)

    # Check the difference
    preds_1_ids = preds_1["Id"].values
    preds_2_ids = preds_2["Id"].values
    preds_1_preds = preds_1["Predicted"].values
    preds_2_preds = preds_2["Predicted"].values
    preds_1_preds = [int(round(pred, 0)) for pred in preds_1_preds]
    preds_2_preds = [int(round(pred, 0)) for pred in preds_2_preds]
    preds_1_preds = np.array(preds_1_preds)
    preds_2_preds = np.array(preds_2_preds)

    # Check the difference
    print("Difference:", np.sum(preds_1_preds != preds_2_preds))

def CheckSubmissions_DifferenceSeparate(pred_path_1, pred_path_2):
    '''
    Check the difference between two submissions
    '''
    # Load Predictions
    preds_1 = ReadDataset(pred_path_1)
    preds_2 = ReadDataset(pred_path_2)

    # Check the difference
    preds_1_ids = preds_1["Id"].values
    preds_2_ids = preds_2["Id"].values
    preds_1_preds = preds_1["Predicted"].values
    preds_2_preds = preds_2["Predicted"].values
    preds_1_preds = [int(round(pred, 0)) for pred in preds_1_preds]
    preds_2_preds = [int(round(pred, 0)) for pred in preds_2_preds]
    preds_1_preds = np.array(preds_1_preds)
    preds_2_preds = np.array(preds_2_preds)

    testCounts = [100, 100, 214, 214, 214, 214]
    curInd = 0
    subDiffs = []
    for i in range(len(testCounts)):
        subDiffs.append(np.sum(preds_1_preds[curInd:curInd+testCounts[i]] != preds_2_preds[curInd:curInd+testCounts[i]]))
        print((i+1), "Difference:", subDiffs[-1])
        curInd += testCounts[i]

    # Check the difference
    mainDiff = np.sum(preds_1_preds != preds_2_preds)
    print("Overall Difference:", mainDiff)

    return subDiffs, mainDiff



# Driver Code
# # Params
# dataset_path = "Datasets/Dataset_1_Training.csv"
# # Params

# # RunCode
# dataset = ReadDataset(dataset_path)
# data_pts, targets = FormatDataset(dataset)