'''
Linear Regression
'''

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Utils

# Main Functions
def PolynomialRegression(x, y, degree):
    # In Polynomial Regression we have,
    # y = b0 + b1*x + b2*x^2 + ... bn*x^n

    # Transform X
    X_transformed = []
    for x_val in x:
        x_poly = [1]
        for p in range(1, degree+1):
            x_poly.append(x_val**p)
        X_transformed.append(x_poly)
    X_transformed = np.array(X_transformed)

    # Calculate Regression Parameters
    phi = X_transformed
    phiTphi = np.matmul(phi.T, phi)
    phiTphi_inv = np.linalg.inv(phiTphi)
    phiTy = np.matmul(phi.T, y)
    b = np.dot(phiTphi_inv, phiTy)

    return b

def LinearRegression(x, y):
    # In Linear Regression we have,
    # y = b0 + b1*x
    # where,
    # b0 = y_mean - b1*x_mean
    # b1 = covariance_xy/variance_xx

    N = x.shape[0]
 
    # Mean of X and Y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
 
    # Variance and Covariance
    covariance_xy = np.sum(x * y) - N * y_mean * x_mean
    variance_xx = np.sum(x * x) - N * x_mean * x_mean
 
    # Calculate Regression parameters
    b_1 = covariance_xy / variance_xx
    b_0 = y_mean - b_1 * x_mean
 
    return (b_0, b_1)

# Predict and Error Functions
def Predict(b, x):
    y_pred = np.array([b[0]] * len(x))
    for i in range(1, len(b)):
        y_pred = y_pred + (b[i] * (x ** i))
    return y_pred

def Error_SumSquares(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.sum((y - y_pred) ** 2)

# Display Functions
def DisplayPolynomial(coeffs, degree):
    print("Polynomial Degree:", degree)
    print("Curve Equation:")
    polyStr = f'y = {coeffs[0]}'
    for i in range(1, len(coeffs)):
        polyStr += f' + {coeffs[i]}*x^{i}'
    print(polyStr)
    print()

# Plot Functions
def PlotRegressionCurves(x, y, bs, degrees, title=''):
    # Order the Points
    xy = sorted(zip(x, y))
    x = np.array([i[0] for i in xy])
    y = np.array([i[1] for i in xy])

    # Plot Actual Points
    plt.scatter(x, y, label="Points")
 
    for i in range(len(bs)):
        b = bs[i]
        deg = degrees[i]
        # Predict
        y_pred = Predict(b, x)
        # Plot Regression Line
        plt.plot(x, y_pred, label="Degree " + str(deg))

    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def PlotPredictions(x, y, y_pred, degree, title=''):
    # Order the Points
    xy = sorted(zip(x, y, y_pred))
    x = np.array([i[0] for i in xy])
    y = np.array([i[1] for i in xy])
    y_pred = np.array([i[2] for i in xy])

    # Plot Actual Points
    plt.scatter(x, y, label="Actual Points")
    # Plot Regression Line
    plt.scatter(x, y_pred, label="Predicted Points for Degree " + str(degree))

    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def PlotErrors(errors, degrees, title=''):
    plt.plot(degrees, errors)

    plt.legend()
    plt.title(title)
    plt.xlabel('Degrees')
    plt.ylabel('Error')
    plt.show()

    return errors

# Driver Code
# Params
train_dataset_Path = 'Datasets/Q1/linear_reg_train_data.csv'
test_dataset_path = 'Datasets/Q1/linear_reg_test_data.csv'

degrees = list(range(1, 81))

pred_savePath = 'Q1_a.csv'
# Params

# RunCode
# Load Train Dataset
dataset_train = Utils.LoadDataset(train_dataset_Path)
X_train = np.reshape(dataset_train.iloc[:, :-1].values, (-1))
Y_train = np.reshape(dataset_train.iloc[:, -1].values, (-1))

# Utils.PlotPoints(X_train, Y_train, 'Train Dataset Points')

# Load Test Dataset
dataset_test = Utils.LoadDataset(test_dataset_path)
X_test = np.reshape(dataset_test.iloc[:, :-1].values, (-1))
Y_test = np.reshape(dataset_test.iloc[:, -1].values, (-1))

# Utils.PlotPoints(X_test, Y_test, 'Test Dataset Points')

print("Train Dataset:", X_train.shape)
print("Test Dataset:", X_test.shape)

# Linear Regression
degrees_coeffs = []
print("Performing Regression...")
for degree in degrees:
    coeffs = PolynomialRegression(X_train, Y_train, degree)
    degrees_coeffs.append(coeffs)
print("Completed Regression.")

# Find Optimal Degree
y_preds_test = []
for i in range(len(degrees)):
    y_preds_test.append(Predict(degrees_coeffs[i], X_test))
errors_test = []
for i in range(len(degrees)):
    errors_test.append(Error_SumSquares(Y_test, y_preds_test[i]))
minError_index = np.argmin(errors_test)
optimalDegree = degrees[minError_index]

# Display Optimal Degree Equation
DisplayPolynomial(degrees_coeffs[minError_index], optimalDegree)
print()

# Predict and Get Errors on Train Data
y_preds_train = []
for i in range(len(degrees)):
    y_preds_train.append(Predict(degrees_coeffs[i], X_train))
errors_train = []
for i in range(len(degrees)):
    errors_train.append(Error_SumSquares(Y_train, y_preds_train[i]))

# Plot on train dataset
print("Error for Train Data for optimal degree: ", errors_train[minError_index])
PlotRegressionCurves(X_train, Y_train, [degrees_coeffs[minError_index]], [optimalDegree], "Best Fit Curve on Train Data")
PlotPredictions(X_train, Y_train, y_preds_train[minError_index], optimalDegree, "Predicted Points on Train Data")

# Plot on test dataset
print("Optimal Degree (Least Error) for Test Data: ", optimalDegree)
print("Least Error for Test Data: ", errors_test[minError_index])
PlotRegressionCurves(X_test, Y_test, [degrees_coeffs[minError_index]], [optimalDegree], "Best Fit Curve on Test Data")
PlotPredictions(X_test, Y_test, y_preds_test[minError_index], optimalDegree, "Predicted Points on Test Data")

# Save Best Predictions
BestPredictions_df = pd.DataFrame(y_preds_test[minError_index])
Utils.SaveDataset(y_preds_test[minError_index], 'Datasets/Q1/linear_reg_test_predictions.csv')