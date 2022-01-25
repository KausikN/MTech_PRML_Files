'''
Single Layer ANN
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Main Functions
def ANN_SingleLayer(n_hidden, X, Y, funcs, learning_rate=0.01, epochs=1000):
    '''
    Single Layer ANN
    '''
    act_func, act_func_deriv = funcs["act_func"], funcs["act_func_deriv"]
    loss_func, loss_func_deriv = funcs["loss_func"], funcs["loss_func_deriv"]
    accuracy_func = funcs["accuracy_func"]
    # Initialize weights
    w1 = np.random.randn(n_hidden, X.shape[1])
    w2 = np.random.randn(1, n_hidden)
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((1, 1))
    # Initialize lists
    losses = []
    accuracies = []
    # Loop over epochs
    for i in range(epochs):
        # Forward Propagation
        z1 = np.dot(w1, X) + b1
        a1 = act_func(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = act_func(z2)
        # Backward Propagation
        dz2 = loss_func_deriv(a2, Y)
        dw2 = np.dot(dz2, a1.T)
        db2 = dz2
        dz1 = np.dot(w2.T, dz2) * act_func_deriv(z1)
        dw1 = np.dot(dz1, X.T)
        db1 = dz1
        # Update weights
        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2
        b1 = b1 - learning_rate * db1
        b2 = b2 - learning_rate * db2
        # Compute loss
        loss = loss_func(a2, Y)
        # Compute accuracy
        accuracy = accuracy_func(a2, Y)
        # Append to lists
        losses.append(loss)
        accuracies.append(accuracy)
        # Print progress
        print(f'Epoch: {i}, Loss: {loss}, Accuracy: {accuracy}')
    # Return weights, losses, accuracies
    return w1, w2, b1, b2, losses, accuracies

# Activation Functions
def sigmoid(z):
    '''
    Sigmoid Activation Function
    '''
    return 1 / (1 + np.exp(-z))

# Activation Function Derivatives
def sigmoid_deriv(z):
    '''
    Sigmoid Activation Function Derivative
    '''
    return sigmoid(z) * (1 - sigmoid(z))

# Loss Functions
def mse(y, y_hat):
    '''
    Mean Squared Error
    '''
    return np.mean((y - y_hat) ** 2)

# Loss Function Derivatives
def mse_deriv(y, y_hat):
    '''
    Mean Squared Error Derivative
    '''
    return 2 * (y - y_hat) / y.shape[0]

# Accuracy Functions
def accuracy(y, y_hat):
    '''
    Accuracy
    '''
    return np.mean(y == y_hat)

# Plot Functions
def PlotData(vals, xlabel, ylabel, title):
    '''
    Plot Data
    '''
    plt.plot(vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Predict Functions
def Predict(X, w1, w2, b1, b2):
    '''
    Predict
    '''
    z1 = np.dot(w1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    # a2 = sigmoid(z2)
    return z2

# Driver Code
# Params

# Params

# RunCode