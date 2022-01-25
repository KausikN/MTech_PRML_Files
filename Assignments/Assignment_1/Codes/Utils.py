'''
Utils for loading datasets
'''

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Main Functions
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

# Plot Functions
def PlotPoints(X, Y, title=''):
    plt.scatter(X, Y)
    plt.title(title)
    plt.show()

def PlotLabelledDataset_2D(X, y, classes, title=''):
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

def PlotLabelledDataset_3D(X, y, classes, title=''):
    classPts = []
    for i in range(len(classes)):
        classPts.append([])
    for pt, cl in zip(X, y):
        mapped_class = int(round(cl, 0))
        classPts[mapped_class].append(pt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(classes)):
        pts = np.array(classPts[i])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=classes[i])
    ax.legend()
    ax.set_title(title)
    plt.show()

# Driver Code
# X = np.linspace(-1, 1, 10)
# Y = np.linspace(-1, 1, 10)
# Pts = []
# labels = []
# for x in tqdm(X):
#     for y in Y:
#         z = x**2 + y**2
#         Pts.append([x, y, z])
#         label = 0 if z < 0.5 else 1
#         labels.append(label)
# PlotDatasetPoints_3D(Pts, labels, [0, 1])