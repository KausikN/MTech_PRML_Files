
# Imports
import math

# Main Functions
def Gaussian(x, mean, std):
    return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-0.5 * ((x - mean) / std) ** 2)

# Driver Code
# Params
x = 66
mean = 74.6
std = 7.89
# Params

# RunCode
print(Gaussian(x, mean, std))