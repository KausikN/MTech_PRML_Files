import numpy as np

# a = 2.4
# sig = 1 / (1 + np.math.exp(-a))

# print(sig)

a = np.array([2.61, 5.99])
softmax = np.exp(a) / np.sum(np.exp(a))
print(softmax)

labels = np.array([0, 1])
crossentropyloss = -np.sum(labels*np.log2(softmax))
print(crossentropyloss)