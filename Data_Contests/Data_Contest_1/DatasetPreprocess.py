'''
Dataset Preprocessor and Visualiser
'''

# Imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from Dataset import *

# Main Functions
# Visualisation Functions
def PlotDatasetCovariance(data_pts, title=""):
    '''
    Visualise the covariance of the dataset points
    '''
    # Visualise the covariance
    cov_matrix = np.cov(data_pts)
    plt.imshow(cov_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def PlotDatasetMean(data_pts, title=""):
    '''
    Visualise the mean of the dataset points
    '''
    # Visualise the mean
    mean_data_pt = np.mean(data_pts, axis=0)
    size = mean_data_pt.shape[0]

    # # Image
    # I_mean = np.ones((int(size/2), size)) * mean_data_pt
    # plt.imshow(I_mean, cmap='viridis')
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

    # Plot
    plt.plot(list(range(size)), mean_data_pt)
    plt.title(title)
    plt.show()

def PlotDatasetVariance(data_pts, title=""):
    '''
    Visualise the variance of the dataset points
    '''
    # Visualise the variance
    variance = np.var(data_pts, axis=0)
    size = variance.shape[0]

    # # Image
    # I_variance = np.ones((int(size/2), size)) * variance
    # plt.imshow(I_variance, cmap='viridis')
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

    # Plot
    plt.plot(list(range(size)), variance)
    plt.title(title)
    plt.show()

def ShowClassImbalanceDataset(data_pts, targets, title=''):
    '''
    Show the class imbalance of the dataset
    '''
    # Calculate the class imbalance
    class_counts = np.bincount(targets)
    # class_counts = class_counts[1:]
    # class_counts = class_counts / np.sum(class_counts)

    # Show the class imbalance
    plt.bar(list(range(len(class_counts))), class_counts)
    plt.title(title)
    plt.show()

def PlotDatasetMeanBins(data_pts, bins=100, title=""):
    '''
    Visualise the mean histogram of the dataset points
    '''
    # Visualise the mean
    mean = np.mean(data_pts, axis=0)
    size = mean.shape[0]

    # # Image
    # I_mean = np.ones((int(size/2), size)) * mean
    # plt.imshow(I_mean, cmap='viridis')
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

    # Plot
    plt.hist(mean, bins=bins)
    plt.title(title)
    plt.show()

def PlotDatasetVarianceBins(data_pts, bins=100, title=""):
    '''
    Visualise the variance histogram of the dataset points
    '''
    # Visualise the variance
    variance = np.var(data_pts, axis=0)
    size = variance.shape[0]

    # # Image
    # I_variance = np.ones((int(size/2), size)) * variance
    # plt.imshow(I_variance, cmap='viridis')
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

    # Plot
    plt.hist(variance, bins=bins)
    plt.title(title)
    plt.show()

# Preprocessing Functions
def DatasetMeanStd(data_pts):
    '''
    Calculate the mean and std of the dataset points
    '''
    # Calculate the mean and std
    mean_data_pt = np.mean(data_pts, axis=0)
    std = np.std(data_pts, axis=0)

    return mean_data_pt, std

def NormaliseDataset(data_pts, mean=None, std=None):
    '''
    Normalise the dataset points
    '''
    # Calculate the mean and variance if not provided
    if mean is None:
        mean = np.mean(data_pts, axis=0)
    if std is None:
        std = np.std(data_pts, axis=0)

    # Normalise the dataset
    data_pts = (data_pts - mean) / std

    return data_pts, mean, std

def CombinedNormaliseDatasets(datasets_pts):
    '''
    Normalise the dataset points
    '''
    # Combine datasets points
    combined_pts = np.concatenate(datasets_pts, axis=0)

    # Calculate the mean and variance if not provided
    mean_data_pts = np.mean(combined_pts, axis=1)
    std_data_pts = np.std(combined_pts, axis=1)

    # Normalise the dataset
    combined_pts = (combined_pts - mean_data_pts) / std_data_pts

    # Split Combined Dataset
    datasets_pts_norm = []
    curIndex = 0
    for i in range(len(datasets_pts)):
        curSize = len(list(datasets_pts[i]))
        dataset_pts_norm = combined_pts[curIndex:curIndex+curSize, :]
        datasets_pts_norm.append(dataset_pts_norm)
        curIndex += curSize

    return datasets_pts_norm, mean_data_pts, std_data_pts

def FeatureExtraction_LDA(X, Y, n_components=2):
    '''
    Feature extraction using LDA
    '''
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    X_lda = lda.fit(X, Y).transform(X)
    print('Original number of features:', X.shape[1])
    print('Reduced number of features:', X_lda.shape[1])

    return X_lda

def FeatureExtraction_PCA(X, Y, n_components=2):
    '''
    Feature extraction using PCA
    '''
    pca = PCA(n_components=n_components)

    fitted_model = pca.fit(X)
    X_features = fitted_model.transform(X)
    print('Original number of features:', X.shape[1])
    print('Reduced number of features:', X_features.shape[1])

    return X_features, fitted_model

def FeatureExtraction_ICA(X, Y, n_components=2):
    '''
    Feature extraction using ICA
    '''
    ica = FastICA(n_components=n_components)

    fitted_model = ica.fit(X)
    X_features = fitted_model.transform(X)
    print('Original number of features:', X.shape[1])
    print('Reduced number of features:', X_features.shape[1])

    return X_features, fitted_model

# Driver Code
# # Params
# train_dataset_path_1 = "Datasets/Dataset_1_Training.csv"
# test_dataset_path_1 = "Datasets/Dataset_1_Testing.csv"
# train_dataset_path_2 = "Datasets/Dataset_2_Training.csv"
# test_dataset_path_2 = "Datasets/Dataset_2_Testing.csv"
# # Params

# # RunCode
# # Load Datasetss
# train_data_pts_1, train_targets_1, test_data_pts_1, train_data_pts_2, train_targets_2, test_data_pts_2 = LoadContestDatasets(train_dataset_path_1, test_dataset_path_1, train_dataset_path_2, test_dataset_path_2)

# # Before Normalisation
# # Visualise Covariance
# PlotDatasetCovariance(train_data_pts_1, "TD 1 Cov - Before Norm")
# PlotDatasetCovariance(train_data_pts_2, "TD 2 Cov - Before Norm")

# # Visualise Mean
# PlotDatasetMean(train_data_pts_1, "TD 1 Mean - Before Norm")
# PlotDatasetMean(train_data_pts_2, "TD 2 Mean - Before Norm")

# # Visualise Variance
# PlotDatasetVariance(train_data_pts_1, "TD 1 Var - Before Norm")
# PlotDatasetVariance(train_data_pts_2, "TD 2 Var - Before Norm")

# # Normalise Dataset
# train_data_pts_1_norm = NormaliseDataset(train_data_pts_1)
# train_data_pts_2_norm = NormaliseDataset(train_data_pts_2)

# # After Normalisation
# # Visualise Covariance
# PlotDatasetCovariance(train_data_pts_1_norm, "TD 1 Cov - After Norm")
# PlotDatasetCovariance(train_data_pts_2_norm, "TD 2 Cov - After Norm")

# # Visualise Mean
# PlotDatasetMean(train_data_pts_1_norm, "TD 1 Mean - After Norm")
# PlotDatasetMean(train_data_pts_2_norm, "TD 2 Mean - After Norm")

# # Visualise Variance
# PlotDatasetVariance(train_data_pts_1_norm, "TD 1 Var - After Norm")
# PlotDatasetVariance(train_data_pts_2_norm, "TD 2 Var - After Norm")

# # LDA
# print("Dataset 1")
# CO = 1
# for i in range(len(train_targets_1)):
#     print("CO_" + str(CO))
#     train_data_pts_1_lda = FeatureExtraction_LDA(train_data_pts_1, train_targets_1[i])
#     CO += 1

# print("Dataset 2")
# for i in range(len(train_targets_2)):
#     print("CO_" + str(CO))
#     train_data_pts_2_lda = FeatureExtraction_LDA(train_data_pts_2, train_targets_2[i])
#     CO += 1