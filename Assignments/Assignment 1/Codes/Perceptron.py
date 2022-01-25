'''
Perceptron
'''

# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm

import Utils

# Main Functions
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
train_dataset_Path = 'Datasets/Q2/classification_train_data.csv'
test_dataset_path = 'Datasets/Q2/classification_test_data.csv'
classes = ["0.0", "1.0"]

lr = 1.0
epochs = 10

w_init = [0.0, 1.0]
b_init = 0.0
# Params

# RunCode
# Load Train Dataset
dataset_train = Utils.LoadDataset(train_dataset_Path)
X_train = np.array(dataset_train.iloc[:, :-1].values)
Y_train = np.array(dataset_train.iloc[:, -1].values)

# Utils.PlotLabelledDataset_2D(X_train, Y_train, classes, 'Train Dataset Points')

# Load Test Dataset
dataset_test = Utils.LoadDataset(test_dataset_path)
X_test = np.array(dataset_test.iloc[:, :-1].values)
Y_test = np.array(dataset_test.iloc[:, -1].values)

# Utils.PlotLabelledDataset_2D(X_test, Y_test, classes, 'Test Dataset Points')

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
confusion_matrix_test = Utils.ConfusionMatrix(Y_test, Y_pred_test, classes)
# Print Test Results
print()
print("Test Dataset Results:")
print(f' => Accuracy: {accuracy_test*100}%')
print(f' => Confusion Matrix:\n{confusion_matrix_test}')
print()

# Predict on train dataset
Y_pred_train = np.array([(np.round(Predict(x, W, B), 0)) for x in X_train])
accuracy_train = GetAccuracy(Y_pred_train, Y_train)
confusion_matrix_train = Utils.ConfusionMatrix(Y_train, Y_pred_train, classes)
# Print Train Results
print()
print("Train Dataset Results:")
print(f' => Accuracy: {accuracy_train*100}%')
print(f' => Confusion Matrix:\n{confusion_matrix_train}')
print()