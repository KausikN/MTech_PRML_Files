'''
Q2 (a)
'''

# Imports
import sys
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

# Perceptron Functions    
def GetAccuracy(y_pred, y_target):
    return np.mean(y_pred == y_target)


def Activation(x, w, b):
    return np.dot(w, x) + b

def Predict(x, w, b):
    activations = Activation(x, w, b)
    y_pred = activations >= 0
    return y_pred

def TrainPerceptron(x, y, w, b, lr=1.0):
    error_total = 0.0

    for x_row, y_target in zip(x, y):
        # Predict
        y_pred = Predict(x_row, w, b)

        # Calculate Error
        error = y_target - y_pred
        error_total += error**2

        # Update Bias
        # b = b + lr*error
        b = b + lr * error

        # Update Weights
        # w = w + lr*error*x
        w = w + lr * error * x_row

    return w, b, error_total

def Perceptron(x, y, lr=1.0, epochs=1, w_init=None, b_init=None):
    # Init Weights and Bias
    w = np.zeros(x.shape[1]) if w_init is None else np.array(w_init)
    b = 0.0 if b_init is None else b_init
    
    # Train Perceptron
    for epoch in (range(epochs)):
        w_old = np.array(w)
        b_old = b
        w, b, error_total = TrainPerceptron(x, y, w, b, lr=lr)

        # Print
        print(f' => EPOCH={epoch}, LR={lr}, ERROR={error_total}')

        # Check if we have converged
        if np.array_equal(w, w_old) and b == b_old:
            print(' => Converged!')
            break

    return w, b

# Driver Code
# Params
train_dataset_Path = 'classification_train_data.csv' if len(sys.argv) < 2 else sys.argv[1]
test_dataset_Path = 'classification_test_data.csv' if len(sys.argv) < 3 else sys.argv[2]
classes = ["0.0", "1.0"]

lr = 1.0
epochs = 10

w_init = [0.0, 1.0]
b_init = 0.0

pred_savePath = 'Q2_a.csv' if len(sys.argv) < 4 else sys.argv[3]
# Params

# RunCode
# Load Train Dataset
dataset_train = LoadDataset(train_dataset_Path)
X_train = np.array(dataset_train.iloc[:, :-1].values)
Y_train = np.array(dataset_train.iloc[:, -1].values)

# PlotLabelledDataset(X_train, Y_train, classes, 'Train Dataset Points')

# Load Test Dataset
dataset_test = LoadDataset(test_dataset_Path)
X_test = np.array(dataset_test.iloc[:, :-1].values)
Y_test = np.array(dataset_test.iloc[:, -1].values)

# PlotLabelledDataset(X_test, Y_test, classes, 'Test Dataset Points')

print("Train Dataset:", X_train.shape)
print("Test Dataset:", X_test.shape)

# Train on train dataset
print("Training...")
W, B = Perceptron(X_train, Y_train, lr=lr, epochs=epochs, w_init=w_init, b_init=b_init)
print("Completed Training.")

# Print Weights and Bias
print()
print("Trained Perceptron:")
print(f' => Bias: {B}')
print(f' => Weights:\n => {W}')
print()

# Predict on test dataset
Y_pred_test = np.array([(np.round(Predict(x, W, B), 0)) for x in X_test])
accuracy_test = GetAccuracy(Y_pred_test, Y_test)
confusion_matrix_test = ConfusionMatrix(Y_test, Y_pred_test, classes)
# Print Test Results
print()
print("Test Dataset Results:")
print(f' => Accuracy: {accuracy_test*100}%')
print(f' => Confusion Matrix:\n{confusion_matrix_test}')
print()

# Predict on train dataset
Y_pred_train = np.array([(np.round(Predict(x, W, B), 0)) for x in X_train])
accuracy_train = GetAccuracy(Y_pred_train, Y_train)
confusion_matrix_train = ConfusionMatrix(Y_train, Y_pred_train, classes)
# Print Train Results
print()
print("Train Dataset Results:")
print(f' => Accuracy: {accuracy_train*100}%')
print(f' => Confusion Matrix:\n{confusion_matrix_train}')
print()

# Plot Classification Boundary
PlotClassificationBoundary(W, B, X_test, Y_test, 'Decision Boundary for Test Data')

# Save Predictions
BestPredictions_df = pd.DataFrame(Y_pred_test, columns=['y_pred'])
SaveDataset(BestPredictions_df, pred_savePath)