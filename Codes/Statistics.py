'''
Statistics Codes
'''

# Imports
import numpy as np

# Main Functions
def mean(x):
    '''
    Mean of a list
    '''
    return np.sum(x) / x.shape[0]

def sd(x):
    '''
    Standard Deviation of a list
    '''
    return np.sqrt(np.sum((x - mean(x))**2) / x.shape[0])

def scatter(x):
    '''
    Scatter of a list
    '''
    return np.sum((x - mean(x))**2)

def cov(x, y):
    '''
    Covariance of two lists
    '''
    return np.sum((x - mean(x)) * (y - mean(y))) / x.shape[0]

# Driver Code
# Params
X1 = [1, 2, 2, 5]
X2 = [1, 3, 4, 3]
# Params

# RunCode
X1 = np.array(X1)
X2 = np.array(X2)

print("Mean of X1:", mean(X1))
print("Standard Deviation of X1:", sd(X1))
print("Variance of X1:", sd(X1)**2)
print("Scatter of X1:", scatter(X1))

print("Mean of X1:", mean(X2))
print("Standard Deviation of X1:", sd(X2))
print("Variance of X1:", sd(X2)**2)
print("Scatter of X1:", scatter(X2))


print("Cov of X1:", cov(X1, X2))

print(1/(1 + np.exp(-0.25)))