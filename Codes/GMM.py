'''
GMM
'''

# Imports
import numpy as np

# Main Functions
def Normal(x, mean, var):
    return ((2 * np.pi * var) ** (-0.5)) * np.exp(-((x - mean) ** 2) / (2 * var))

def GetGammas(x, pis, means, variances):
    gammas = np.zeros((x.shape[0], pis.shape[0]))
    for i in range(pis.shape[0]):
        gammas[:, i] = (pis[i] * Normal(x, means[i], variances[i]))
    return gammas / np.sum(gammas, axis=1, keepdims=True)

def GetPis(gammas, N):
    return np.sum(gammas, axis=0) / N

def GetMeans(x, gammas, N):
    means = np.zeros(gammas.shape[1])
    for i in range(gammas.shape[1]):
        means[i] = np.sum(gammas[:, i] * x, axis=0) / np.sum(gammas[:, i])
    return means

def GetVariances(x, gammas, means, N):
    variances = np.zeros(gammas.shape[1])
    for i in range(gammas.shape[1]):
        variances[i] = np.sum(gammas[:, i] * ((x - means[i]) ** 2), axis=0) / np.sum(gammas[:, i])
    return variances

def GMM(x, pis, means, variances, max_iterations=100):
    N = x.shape[0]
    for i in range(max_iterations):
        gammas = GetGammas(x, pis, means, variances)
        pis = GetPis(gammas, N)
        means = GetMeans(x, gammas, N)
        variances = GetVariances(x, gammas, means, N)

        # print("IT " + str(i) + ":")
        # for j in range(pis.shape[0]):
        #     print(j, pis[j], means[j], variances[j])
        # print()

    return pis, means, variances

# Plot Functions
def PlotHistogram(x, pis, means, variances, bins=50):
    import matplotlib.pyplot as plt
    plt.hist(x, bins=bins, density=True)
    for i in range(pis.shape[0]):
        p_x = np.linspace(np.min(x), np.max(x), bins)
        p_y = pis[i] * Normal(p_x, means[i], variances[i])
        plt.plot(p_x, p_y, label=str(i))
    plt.legend()
    plt.show()

# Driver Code
# Params
Means = [
    1,
    2
]

Variances = [
    1,
    1
]

X = [
    1,
    2,
    3,
    5,
    10,
]
# Params

# RunCode
Means = np.array(Means)
Variances = np.array(Variances)
X = np.array(X)
K = Means.shape[0]
Pis = np.ones(K) / K

# Before
# PlotHistogram(X, Pis, Means, Variances)

# Run
Pis, Means, Variances = GMM(X, Pis, Means, Variances)

# After
# PlotHistogram(X, Pis, Means, Variances)

# Print
print("FINAL")
for j in range(Pis.shape[0]):
    print(j, Pis[j], Means[j], Variances[j])
print()