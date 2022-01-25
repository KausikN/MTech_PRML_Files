'''
Q1 (c)
'''

# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Main Functions
# Dataset Functions
def LoadDataset(path):
    return pd.read_csv(path)

def SaveDataset(dataset, savePath):
    dataset.to_csv(savePath, index=False)

# Regression Functions
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

def RidgeRegression(x, y, degree, lam):
    # In Ridge Regression we have,
    # y = b0 + b1*x + b2*x^2 + ... bn*x^n + λ*(b0 + b1*x + b2*x^2 + ... bn*x^n)

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
    phiTphi_inv = np.linalg.inv(phiTphi + (np.identity(phiTphi.shape[0]) * lam))
    phiTy = np.matmul(phi.T, y)
    b = np.dot(phiTphi_inv, phiTy)

    return b

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
    polyStr = f'y = \t{coeffs[0]}'
    for i in range(1, len(coeffs)):
        polyStr += f'\n\t + ({coeffs[i]}) x^{i}'
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
 
    x = np.linspace(np.min(x), np.max(x), 1000)
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

# Driver Code
# Params
train_dataset_Path = 'linear_reg_train_data.csv' if len(sys.argv) < 2 else sys.argv[1]
test_dataset_Path = 'linear_reg_test_data.csv' if len(sys.argv) < 3 else sys.argv[2]

degree = 10
lambdas = np.linspace(0, 1, 1000)

pred_savePath = 'Q1_c.csv' if len(sys.argv) < 4 else sys.argv[3]
# Params

# RunCode
# Load Train Dataset
dataset_train = LoadDataset(train_dataset_Path)
X_train = np.reshape(dataset_train.iloc[:, :-1].values, (-1))
Y_train = np.reshape(dataset_train.iloc[:, -1].values, (-1))

# Utils.PlotPoints(X_train, Y_train, 'Train Dataset Points')

# Load Test Dataset
dataset_test = LoadDataset(test_dataset_Path)
X_test = np.reshape(dataset_test.iloc[:, :-1].values, (-1))
Y_test = np.reshape(dataset_test.iloc[:, -1].values, (-1))

# Utils.PlotPoints(X_test, Y_test, 'Test Dataset Points')

print("Train Dataset:", X_train.shape)
print("Test Dataset:", X_test.shape)

# Linear Regression
lambdas_coeffs = []
for lam in lambdas:
    coeffs = RidgeRegression(X_train, Y_train, degree, lam)
    lambdas_coeffs.append(coeffs)

# Predict and Get Errors on Test Data
y_preds_test = []
for i in range(len(lambdas)):
    y_preds_test.append(Predict(lambdas_coeffs[i], X_test))
errors_test = []
for i in range(len(lambdas)):
    errors_test.append(Error_SumSquares(Y_test, y_preds_test[i]))

# Predict and Get Errors on Train Data
y_preds_train = []
for i in range(len(lambdas)):
    y_preds_train.append(Predict(lambdas_coeffs[i], X_train))
errors_train = []
for i in range(len(lambdas)):
    errors_train.append(Error_SumSquares(Y_train, y_preds_train[i]))

# plt.plot(lambdas, errors_train, label="Train")
# plt.plot(lambdas, errors_test, label="Test")
# plt.legend()
# plt.show()

# Overfit Degree is taken as 10
optimal_index = np.argmin(errors_test)
optimal_lambda = lambdas[optimal_index]

# Report Minimum Error
print("Overfit Degree (Less Error on Train Data but High Error on Test Data):", degree)
print("Optimal Lambda:", optimal_lambda)
print("Least Sum of Squares Error for Test Data:", errors_test[optimal_index])
print("Sum of Squares Error for Train Data:", errors_train[optimal_index])

# Display Optimal Degree Equation
print("Best Fit Curve Equation:")
DisplayPolynomial(lambdas_coeffs[optimal_index], degree)
print()

# Plot on train dataset
PlotRegressionCurves(X_train, Y_train, [lambdas_coeffs[optimal_index]], [degree], "Best Fit Curve on Train Data")
PlotPredictions(X_train, Y_train, y_preds_train[optimal_index], degree, "Predicted Points on Train Data")

# Plot on test dataset
PlotRegressionCurves(X_test, Y_test, [lambdas_coeffs[optimal_index]], [degree], "Best Fit Curve on Test Data")
PlotPredictions(X_test, Y_test, y_preds_test[optimal_index], degree, "Predicted Points on Test Data")

# Save Best Predictions
BestPredictions_df = pd.DataFrame(y_preds_test[optimal_index], columns=['y_pred'])
SaveDataset(BestPredictions_df, pred_savePath)