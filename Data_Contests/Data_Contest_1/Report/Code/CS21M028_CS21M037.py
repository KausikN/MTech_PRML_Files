# Imports
import numpy as np
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

"""# Dataset Functions"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd

# Main Functions
# Dataset Functions
def ReadDataset(path):
    dataset = pd.read_csv(path)
    return dataset

def SaveDataset(dataset, path):
    dataset.to_csv(path, index=False)

def FormatDataset(dataset, targetRowCount=0):
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

# Dataset Paths
train_dataset_path_1 = "./Dataset_1_Training.csv"
test_dataset_path_1 = "./Dataset_1_Testing.csv"
train_dataset_path_2 = "./Dataset_2_Training.csv"
test_dataset_path_2 = "./Dataset_2_Testing.csv"

# Load Datasets
train_data_pts_1_original, train_targets_1, test_data_pts_1_original, train_data_pts_2_original, train_targets_2, test_data_pts_2_original = \
    LoadContestDatasets(train_dataset_path_1, test_dataset_path_1, train_dataset_path_2, test_dataset_path_2)

"""# Dataset Visualisations"""

# Visualisation Functions
def PlotDatasetCovariance(data_pts, title=""):
    # Visualise the covariance
    cov_matrix = np.cov(data_pts)
    plt.imshow(cov_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def PlotDatasetMean(data_pts, title=""):
    # Visualise the mean
    mean_data_pt = np.mean(data_pts, axis=0)
    size = mean_data_pt.shape[0]

    # Plot
    plt.plot(list(range(size)), mean_data_pt)
    plt.title(title)
    plt.show()

def PlotDatasetVariance(data_pts, title=""):
    # Visualise the variance
    variance = np.var(data_pts, axis=0)
    size = variance.shape[0]

    # Plot
    plt.plot(list(range(size)), variance)
    plt.title(title)
    plt.show()
    
def ShowClassImbalanceDataset(targets, title=''):
    # Calculate the class imbalance
    targets = np.array(targets, dtype=int)
    class_counts = np.bincount(targets)
    # class_counts = class_counts[1:]
    # class_counts = class_counts / np.sum(class_counts)

    # Show the class imbalance
    plt.bar(list(range(len(class_counts))), class_counts)
    plt.title(title)
    plt.show()
    
def PlotDatasetMeanBins(data_pts, bins=100, title=""):
    # Visualise the mean
    mean = np.mean(data_pts, axis=0)
    size = mean.shape[0]

    # Plot
    plt.hist(mean, bins=bins)
    plt.title(title)
    plt.show()

def PlotDatasetVarianceBins(data_pts, bins=100, title=""):
    # Visualise the variance
    variance = np.var(data_pts, axis=0)
    size = variance.shape[0]
    
    # Plot
    plt.hist(variance, bins=bins)
    plt.title(title)
    plt.show()

print("TRAIN DATA VISUALISATION - ORIGINAL")
# Train Dataset 1
# Visualise Covariance
PlotDatasetCovariance(train_data_pts_1_original, "Original Train Dataset 1 Covariance")

# Visualise Mean
PlotDatasetMean(train_data_pts_1_original, "Original Train Dataset 1 Mean")
PlotDatasetMeanBins(train_data_pts_1_original, 100, "Original Train Dataset 1 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(train_data_pts_1_original, "Original Train Dataset 1 Variance")
PlotDatasetVarianceBins(train_data_pts_1_original, 100, "Original Train Dataset 1 Variance Histogram")

# Train Dataset 2
# Visualise Covariance
PlotDatasetCovariance(train_data_pts_2_original, "Original Train Dataset 2 Covariance")

# Visualise Mean
PlotDatasetMean(train_data_pts_2_original, "Original Train Dataset 2 Mean")
PlotDatasetMeanBins(train_data_pts_2_original, 100, "Original Train Dataset 2 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(train_data_pts_2_original, "Original Train Dataset 2 Variance")
PlotDatasetVarianceBins(train_data_pts_2_original, 100, "Original Train Dataset 2 Variance Histogram")

print("TEST DATA VISUALISATION - ORIGINAL")
# Test Dataset 1
# Visualise Covariance
PlotDatasetCovariance(test_data_pts_1_original, "Original Test Dataset 1 Covariance")

# Visualise Mean
PlotDatasetMean(test_data_pts_1_original, "Original Test Dataset 1 Mean")
PlotDatasetMeanBins(test_data_pts_1_original, 100, "Original Test Dataset 1 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(test_data_pts_1_original, "Original Test Dataset 1 Variance")
PlotDatasetVarianceBins(test_data_pts_1_original, 100, "Original Test Dataset 1 Variance Histogram")

# Test Dataset 2
# Visualise Covariance
PlotDatasetCovariance(test_data_pts_2_original, "Original Test Dataset 2 Covariance")

# Visualise Mean
PlotDatasetMean(test_data_pts_2_original, "Original Test Dataset 2 Mean")
PlotDatasetMeanBins(test_data_pts_2_original, 100, "Original Test Dataset 2 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(test_data_pts_2_original, "Original Test Dataset 2 Variance")
PlotDatasetVarianceBins(test_data_pts_2_original, 100, "Original Test Dataset 2 Variance Histogram")

"""# Dataset Preprocessing"""

train_data_pts_features = []
test_data_pts_features = []

train_targets = list(train_targets_1)
train_targets.extend(train_targets_2)

for i in range(2):
    train_data_pts_features.append(np.copy(train_data_pts_1_original))
    test_data_pts_features.append(np.copy(test_data_pts_1_original))
for i in range(4):
    train_data_pts_features.append(np.copy(train_data_pts_2_original))
    test_data_pts_features.append(np.copy(test_data_pts_2_original))

# Print Dataset Shapes
print("TRAIN DATASETS: ")
for i in range(len(train_data_pts_features)):
    print("CO_" + str(i+1) + " Data points: ", train_data_pts_features[i].shape)

print("TEST DATASETS: ")
for i in range(len(test_data_pts_features)):
    print("CO_" + str(i+1) + " Data points: ", test_data_pts_features[i].shape)

# Imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Preprocessing Functions
def MinMaxScaleDataset(dataset_pts):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataset_pts)
    return scaled

def StandardScaleDataset(dataset_pts):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset_pts)
    return scaled

def CombinedStandardScaleDataset(datasets_pts):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(datasets_pts[0])
    scaled_test = scaler.transform(datasets_pts[1])
    return scaled_train, scaled_test

train_data_pts = []
test_data_pts = []

# # Original Datasets
# for pts in train_data_pts_features:
#     train_data_pts.append(np.copy(pts))
# for pts in test_data_pts_features:
#     test_data_pts.append(np.copy(pts))

# # MinMaxScale Datasets Separately
# for pts in train_data_pts_features:
#     pts_scaled = MinMaxScaleDataset(pts)
#     train_data_pts.append(pts_scaled)
# for pts in test_data_pts_features:
#     pts_scaled = MinMaxScaleDataset(pts)
#     test_data_pts.append(pts_scaled)
    
# StandardScale Datasets Separately
for pts in train_data_pts_features:
    pts_scaled = StandardScaleDataset(pts)
    train_data_pts.append(pts_scaled)
for pts in test_data_pts_features:
    pts_scaled = StandardScaleDataset(pts)
    test_data_pts.append(pts_scaled)

"""# Preprocessed Dataset Visualisations"""

print("TRAIN DATA CLASSES")
# Visualise Class Imbalance
co = 0
for i in range(train_targets_1.shape[0]):
    co += 1
    ShowClassImbalanceDataset(train_targets_1[i], "Train Dataset 1 CO_" + str(co) + " Classes")
for i in range(train_targets_2.shape[0]):
    co += 1
    ShowClassImbalanceDataset(train_targets_2[i], "Train Dataset 2 CO_" + str(co) + " Classes")

print("TRAIN DATA VISUALISATION - SCALED")
# Train Dataset 1
# Visualise Covariance
PlotDatasetCovariance(train_data_pts[0], "Train Dataset 1 Covariance")

# Visualise Mean
PlotDatasetMean(train_data_pts[0], "Train Dataset 1 Mean")
PlotDatasetMeanBins(train_data_pts[0], 100, "Train Dataset 1 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(train_data_pts[0], "Train Dataset 1 Variance")
PlotDatasetVarianceBins(train_data_pts[0], 100, "Train Dataset 1 Variance Histogram")

# Train Dataset 2
# Visualise Covariance
PlotDatasetCovariance(train_data_pts[-1], "Train Dataset 2 Covariance")

# Visualise Mean
PlotDatasetMean(train_data_pts[-1], "Train Dataset 2 Mean")
PlotDatasetMeanBins(train_data_pts[-1], 100, "Train Dataset 2 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(train_data_pts[-1], "Train Dataset 2 Variance")
PlotDatasetVarianceBins(train_data_pts[-1], 100, "Train Dataset 2 Variance Histogram")

print("TEST DATA VISUALISATION - SCALED")
# Test Dataset 1
# Visualise Covariance
PlotDatasetCovariance(test_data_pts[0], "Test Dataset 1 Covariance")

# Visualise Mean
PlotDatasetMean(test_data_pts[0], "Test Dataset 1 Mean")
PlotDatasetMeanBins(test_data_pts[0], 100, "Test Dataset 1 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(test_data_pts[0], "Test Dataset 1 Variance")
PlotDatasetVarianceBins(test_data_pts[0], 100, "Test Dataset 1 Variance Histogram")

# Test Dataset 2
# Visualise Covariance
PlotDatasetCovariance(test_data_pts[-1], "Test Dataset 2 Covariance")

# Visualise Mean
PlotDatasetMean(test_data_pts[-1], "Test Dataset 2 Mean")
PlotDatasetMeanBins(test_data_pts[-1], 100, "Test Dataset 2 Mean Histogram")

# Visualise Variance
PlotDatasetVariance(test_data_pts[-1], "Test Dataset 2 Variance")
PlotDatasetVarianceBins(test_data_pts[-1], 100, "Test Dataset 2 Variance Histogram")

"""# Classifiers

## AdaBoost
"""

# Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Main Functions
# AdaBoost Functions
def AdaBoost_Run(train_data_pts, train_targets, colsample_bytree=0.3, learning_rate=0.1, max_depth=10, alpha=10, n_estimators=200, seed=0):
    X_train, X_test, y_train, y_test = train_data_pts, train_data_pts, train_targets, train_targets

    classifier = AdaBoostClassifier(
#         RandomForestClassifier( n_estimators=5,
        DecisionTreeClassifier(
            max_depth=int(max_depth), random_state=int(seed)
        ), 
        n_estimators=int(n_estimators), learning_rate=learning_rate, random_state=int(seed))
    trained_classifier = classifier.fit(X_train, y_train)

    # Print Results
    y_pred = trained_classifier.predict(X_test)
    missclassified_count = (y_test != y_pred).sum()
    print("Missclassified:", missclassified_count, "/", X_test.shape[0])
    print("\n")
        
    return trained_classifier, missclassified_count

"""# Train"""

def RunContestTraining(Xs, Ys, 
                       params):
    ALGO = params[0]
    colsample_bytree=params[1]
    learning_rate=params[2]
    max_depth=params[3]
    alpha=params[4]
    n_estimators=params[5]
    seed=params[6]
    trained_classifiers = []
    missclassified_counts = []
    curCOs = []

    #### DATASET 1 ####
    # Train Classifier
    print("DATASET 1")

    # CO_1
    curCO = 1
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[0](Xs[0], Ys[0], # MODEL CHANGE
        colsample_bytree=colsample_bytree[0], learning_rate=learning_rate[0], max_depth=max_depth[0], 
                                      alpha=alpha[0], n_estimators=n_estimators[0], seed=seed[0])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()

    # CO_2
    curCO = 2
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[1](Xs[1], Ys[1], # MODEL CHANGE
        colsample_bytree=colsample_bytree[1], learning_rate=learning_rate[1], max_depth=max_depth[1], 
                                      alpha=alpha[1], n_estimators=n_estimators[1], seed=seed[1])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()


    #### DATASET 2 ####
    # Train Classifier
    print("DATASET 2")

    # CO_3
    curCO = 3
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[2](Xs[2], Ys[2], # MODEL CHANGE
        colsample_bytree=colsample_bytree[2], learning_rate=learning_rate[2], max_depth=max_depth[2], 
                                      alpha=alpha[2], n_estimators=n_estimators[2], seed=seed[2])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()

    # CO_4
    curCO = 4
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[3](Xs[3], Ys[3], # MODEL CHANGE
        colsample_bytree=colsample_bytree[3], learning_rate=learning_rate[3], max_depth=max_depth[3], 
                                      alpha=alpha[3], n_estimators=n_estimators[3], seed=seed[3])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()

    # CO_5
    curCO = 5
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[4](Xs[4], Ys[4], # MODEL CHANGE
        colsample_bytree=colsample_bytree[4], learning_rate=learning_rate[4], max_depth=max_depth[4], 
                                      alpha=alpha[4], n_estimators=n_estimators[4], seed=seed[4])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()

    # CO_6
    curCO = 6
    print("For C" + str(curCO) + ": ")
    trained_classifier, missclassified_count = ALGO[5](Xs[5], Ys[5], # MODEL CHANGE
        colsample_bytree=colsample_bytree[5], learning_rate=learning_rate[5], max_depth=max_depth[5], 
                                      alpha=alpha[5], n_estimators=n_estimators[5], seed=seed[5])
    trained_classifiers.append(trained_classifier)
    missclassified_counts.append(missclassified_count)
    curCOs.append(curCO)
    print()
    
    return trained_classifiers, missclassified_counts, curCOs

def RunContestPredict(Xs, Ys, trained_classifiers, curCOs):
    # Predict on Validation Dataset
    missclassified_counts = []
    for i in range(len(trained_classifiers)):
        trained_classifier = trained_classifiers[i]
        y_pred = trained_classifier.predict(Xs[i])
        missclassified_count = (Ys[i] != y_pred).sum()
        print("For C" + str(curCOs[i]) + ": ")
        print("Missclassified:", missclassified_count, "/", Xs[i].shape[0])
        print("\n")
        missclassified_counts.append(missclassified_count)
    
    return missclassified_counts

"""## HyperParameter Tuning and Validation"""

# HyperParamsTests = [
#     [
#         [AdaBoost_Run, 0.75, 0.2, 1, 10, 200, 0],
#         [AdaBoost_Run, 0.75, 0.2, 1, 10, 200, 0],
#         [AdaBoost_Run, 0.75, 0.2, 1, 10, 200, 0],
#         [AdaBoost_Run, 0.75, 0.2, 1, 10, 200, 0],
#         [AdaBoost_Run, 0.75, 0.25, 1, 10, 30, 0],
#         [AdaBoost_Run, 0.75, 0.25, 1, 10, 30, 0],
#     ]
# ]

# # Dataset Split for Validation
# X_trains = []
# Y_trains = []
# X_vals = []
# Y_vals = []

# for i in range(len(train_data_pts)):
#     X_train, X_test, y_train, y_test = train_test_split(train_data_pts[i], train_targets[i].T, 
#                 test_size=0.2, random_state=0) # SPLIT SEED -- CHANGE
#     X_trains.append(X_train)
#     Y_trains.append(y_train.T)
#     X_vals.append(X_test)
#     Y_vals.append(y_test.T)

#     print("DATASET CO_" + str(i+1))
#     print("Train: X:", X_train.shape, "Y:", y_train.shape)
#     print("Val: X:", X_test.shape, "Y:", y_test.shape)

# # Run Contest Training and Prediction on Grid of Hyperparams
# train_errors = []
# val_errors = []
# runs_trained_classifiers = []
# for i in tqdm(range(len(HyperParamsTests))):
#     params = list(np.array(HyperParamsTests[i]).T)
#     print("RUNNING TEST:", i+1)
#     print("PARAMS: colsample_bytree, learning_rate, max_depth, alpha, n_estimators, seed")
#     print("VALUES:", params)
#     print()
#     print("Training...")
#     trained_classifiers, train_missclassified_counts, curCOs = RunContestTraining(X_trains, Y_trains, 
#                            params)
#     print("Validating...")
#     val_missclassified_counts = RunContestPredict(X_vals, Y_vals, trained_classifiers, curCOs)
# #     train_missclassified_counts = [0, 0, 0, 0, 0, 0]
# #     val_missclassified_counts = [0, 0, 0, 0, 0, 0]
# #     trained_classifiers = [None, None, None, None, None]
#     train_errors.append(train_missclassified_counts)
#     val_errors.append(val_missclassified_counts)
#     runs_trained_classifiers.append(trained_classifiers)
#     print()

# # Save Test Run Errors and Data
# run_save_path = "./run_data.csv"

# HyperParams = []
# for hparams in HyperParamsTests:
#     hparams = np.array(hparams).T
#     hparams_joined = [", ".join([f.__name__ for f in hparams[0]])]
#     for p in hparams[1:]:
#         hparams_joined.append(", ".join(list(map(str, p))))
#     HyperParams.append(hparams_joined)
# HyperParams = np.array(HyperParams).T
# HyperParamsColumnNames = ["algo", "colsample_bytree", "learning_rate", "max_depth", "alpha", "n_estimators", "seed"]

# # Form a dataframe
# train_n_pts = []
# val_n_pts = []
# for i in range(len(X_trains)):
#     train_n_pts.append(X_trains[i].shape[0])
#     val_n_pts.append(X_vals[i].shape[0])
# train_n_pts_total = sum(train_n_pts)
# val_n_pts_total = sum(val_n_pts)
# train_n_pts_array = np.array(train_n_pts)
# val_n_pts_array = np.array(val_n_pts)

# total_train_errors_stacked = list((np.sum(train_errors, axis=1) / (train_n_pts_total)).T)
# total_val_errors_stacked = list((np.sum(val_errors, axis=1) / (val_n_pts_total)).T)
# train_errors_stacked = list((np.array(train_errors) / (train_n_pts_array)).T)
# val_errors_stacked = list((np.array(val_errors) / (val_n_pts_array)).T)
# indices = list(range(len(train_errors)))
# columns_data = ([indices]
#                 + list(HyperParams)
#                 + [total_train_errors_stacked, total_val_errors_stacked]
#                 + train_errors_stacked + val_errors_stacked)
# # columns_data = np.array(columns_data)
# columns_names = (["id"] + HyperParamsColumnNames
#                 + ["total_train_error_" + str(train_n_pts_total), "total_val_error_" + str(val_n_pts_total)]
#                 + ["train_error_CO_" + str(i+1) for i in range(len(train_errors[0]))]
#                 + ["val_error_CO_" + str(i+1) for i in range(len(val_errors[0]))])
# run_data = np.dstack(tuple(columns_data))[0]
# RunData_df = pd.DataFrame(data=run_data, columns=columns_names)
# SaveDataset(RunData_df, run_save_path)

# ReadDataset(run_save_path)

"""## Final Training

colsample_bytree and alpha ARE NOT USED by AdaBoost - was used in other algos

Given here so that same code can be used for training other algorithms

### Final Tuned HyperParams

 - CO_1 - AdaBoost with Decision Tree base_estimator, 0.2 Learning Rate, 1 Max Depth, 200 Estimators, Seed 0
 - CO_2 - AdaBoost with Decision Tree base_estimator, 0.2 Learning Rate, 1 Max Depth, 200 Estimators, Seed 0
 - CO_3 - AdaBoost with Decision Tree base_estimator, 0.2 Learning Rate, 1 Max Depth, 200 Estimators, Seed 0
 - CO_4 - AdaBoost with Decision Tree base_estimator, 0.2 Learning Rate, 1 Max Depth, 200 Estimators, Seed 0
 - CO_5 - AdaBoost with Decision Tree base_estimator, 0.25 Learning Rate, 1 Max Depth, 30 Estimators, Seed 0
 - CO_6 - AdaBoost with Decision Tree base_estimator, 0.25 Learning Rate, 1 Max Depth, 30 Estimators, Seed 0
"""

# Choose Best Classifier HyperParams from Tuning
# print(np.array(train_errors).shape, np.array(val_errors).shape)
# total_errors = np.sum(train_errors, axis=1) + np.sum(val_errors, axis=1)
# best_classifier_index = np.argmin(total_errors)
# Train on Whole Train Dataset with best HyperParams
# best_params = HyperParamsTests[best_classifier_index]

# Use Original Dataset for CO_1 to CO_4, Scaled Dataset for CO_5 and CO_6
# No Scaling for first 4
for i in range(4):
    train_data_pts[i] = np.copy(train_data_pts_features[i])
    test_data_pts[i] = np.copy(test_data_pts_features[i])

# Set HyperParams
best_params = [
        [AdaBoost_Run, 0.0, 0.2, 1, 10, 200, 0],
        [AdaBoost_Run, 0.0, 0.2, 1, 10, 200, 0],
        [AdaBoost_Run, 0.0, 0.2, 1, 10, 200, 0],
        [AdaBoost_Run, 0.0, 0.2, 1, 10, 200, 0],
        [AdaBoost_Run, 0.0, 0.25, 1, 10, 30, 0],
        [AdaBoost_Run, 0.0, 0.25, 1, 10, 30, 0],
    ]
best_params = list(np.array(best_params).T)
print("RUNNING FINAL TRAIN:", i+1)
print("PARAMS: colsample_bytree, learning_rate, max_depth, alpha, n_estimators, seed")
print("VALUES:", best_params)
trained_classifiers, train_missclassified_counts, curCOs = RunContestTraining(
        train_data_pts, train_targets, 
       best_params)

"""# Save Predictions"""

# Predict on Test Dataset
Predictions = []
for i in range(len(trained_classifiers)):
    trained_classifier = trained_classifiers[i]
    y_pred = trained_classifier.predict(test_data_pts[i])
    Predictions.append([int(round(yp, 0)) for yp in list(y_pred)])

# Save Predictions
preds_save_path = "./submission.csv"
SaveContestPredictions(Predictions, preds_save_path)

