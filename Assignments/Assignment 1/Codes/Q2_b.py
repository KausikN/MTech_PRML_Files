'''
Q2 (b)
'''

# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Main Functions
# Dataset Functions
def LoadDataset(path):
    return pd.read_csv(path)

def SaveDataset(dataset, savePath):
    dataset.to_csv(savePath, index=False)

def ConfusionMatrix(y_target, y_pred, classes):
    dim = len(classes)
    Conf_Matrix = np.zeros((dim, dim))

    # row => Target
    # column => Prediction
    for target, pred in zip(y_target, y_pred):
        target_index = classes.index(str(target))
        pred_index = classes.index(str(pred))
        Conf_Matrix[target_index][pred_index] += 1
    Conf_Matrix /= y_target.shape[0]

    columns = []
    indices = []
    for c in classes:
        columns.append("Predicted " + str(c))
        indices.append("Actual " + str(c))

    ConfMatrix_df = pd.DataFrame(Conf_Matrix, columns=columns, index=indices)

    return ConfMatrix_df

def PlotLabelledDataset(X, y, classes, title=''):
    classPts = []
    for i in range(len(classes)):
        classPts.append([])
    for pt, cl in zip(X, y):
        mapped_class = int(round(cl, 0))
        classPts[mapped_class].append(pt)

    for i in range(len(classes)):
        pts = np.array(classPts[i])
        plt.scatter(pts[:, 0], pts[:, 1], label=classes[i])
    plt.legend()
    plt.title(title)
    plt.show()

def PlotClassificationBoundary(W, B, X, Y, title=''):
    # Plot Points
    classPts = []
    for i in range(len(classes)):
        classPts.append([])
    for pt, cl in zip(X, Y):
        mapped_class = int(round(cl, 0))
        classPts[mapped_class].append(pt)
    for i in range(len(classes)):
        pts = np.array(classPts[i])
        plt.scatter(pts[:, 0], pts[:, 1], label="Class " + classes[i])

    # Plot Decision Boundary
    X = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    Y = -(W[0]*X + B)/W[1]
    plt.plot(X, Y, c="black", label="Decision Boundary")

    plt.legend()
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    
    plt.show()

# Discriminant Functions
def LinearDiscriminant_Case1(X, Y, classes):
    # Calculate Means
    classMeans = []
    for i in range(len(classes)):
        classMeans.append([])
    for i in range(len(classes)):
        classMeans[i] = np.mean(X[Y == classes[i]], axis=0)
    classMeans = np.array(classMeans)

    # Calculate Within-Class Scatter Matrix
    W = []
    for i in range(len(classes)):
        W.append([])
    for i in range(len(classes)):
        W[i] = np.zeros((2, 2))
    for i in range(len(classes)):
        for pt in X[Y == classes[i]]:
            diff = pt - classMeans[i]
            W[i] += np.outer(diff, diff)
    W = np.array(W)

    # Calculate Between-Class Scatter Matrix
    B = np.zeros((2, 2))
    for i in range(len(classes)):
        diff = classMeans[i] - np.mean(classMeans, axis=0)
        B += np.outer(diff, diff)
    B /= len(classes)

    # Calculate Covariance Matrix
    C = W + B

    # Calculate Eigenvalues and Eigenvectors
    eigVals, eigVecs = np.linalg.eig(C)
    idx = eigVals.argsort()[::-1]
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    # Project Data
    X_proj = np.dot(X, eigVecs[:, 0])

    return X_proj, eigVals, eigVecs