'''
Functions
'''

# Imports
import numpy as np

# Main Functions
def Sigmoid(x):
    '''
    Sigmoid Function
    '''
    return 1 / (1 + np.exp(-x))

def Sigmoid_Deriv(x):
    '''
    Sigmoid Derivative
    '''
    return Sigmoid(x) * (1 - Sigmoid(x))

def Tanh(x):
    '''
    Tanh Function
    '''
    return np.tanh(x)

def Tanh_Deriv(x):
    '''
    Tanh Derivative
    '''
    return 1 - np.tanh(x)**2

def Normal(x, mean, var):
    return ((2 * np.pi * var) ** (-0.5)) * np.exp(-((x - mean) ** 2) / (2 * var))

# Driver Code
# Params
Funcs = [Sigmoid, Tanh, Sigmoid_Deriv, Tanh_Deriv, Normal]

Func = 1

x = 1
mean = 0
var = 1
# Params

# RunCode
# Func = Funcs[Func]

# if Func == Normal:
#     print(Func(x, mean, var))
# else:
#     print(Func(x))
# print(np.exp(1))

