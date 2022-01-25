# Imports
import numpy as np

# Main Functions
def Entropy(data):
    """
    Calculates the entropy of a dataset.
    """
    # Calculate the probabilities of each class
    labels = np.unique(data, return_counts=True)
    counts = labels[1]
    p = counts / data.shape[0]
    # print(p)
    print(data.shape[0])
    print(labels[0])
    print(counts)
    # Calculate the entropy
    return -np.sum(p * np.log2(p))

def InformationGain(data, split_index, out_index):
    """
    Calculates the information gain of a dataset.
    """
    # Calculate the entropy of the dataset before split
    H = Entropy(data[:, out_index])
    # Split Dataset
    datasets_split, values = SplitDatasetOnColumn(data, split_index)
    # Calculate the entropy of each split dataset
    H_split = np.array([Entropy(dataset[:, out_index]) for dataset in datasets_split])
    datasets_split_weights = np.array([dataset.shape[0] / data.shape[0] for dataset in datasets_split])
    # Calculate the information gain
    IG = H - np.sum(H_split * datasets_split_weights)
    return IG

def SplitDatasetOnColumn(data, column):
    """
    Splits a dataset into two parts based on the values of a column.
    """
    # Create a list of values for the column
    values = np.unique(data[:, column])
    # Create a list of datasets
    datasets = []
    # For each value in the column
    for value in values:
        # Create a new dataset
        dataset = np.empty((0, data.shape[1]))
        # For each row in the dataset
        for row in data:
            # If the value of the column is equal to the value
            if row[column] == value:
                # Add the row to the dataset
                dataset = np.vstack((dataset, row))
        # Add the dataset to the list of datasets
        datasets.append(np.array(dataset))
    # Return the list of datasets
    return datasets, values

# Driver Code
# Q1
Dataset = [
    [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2], # Age
    [2, 2, 2, 1, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1], # Income
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], # Student
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], # Credit
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0], # Buys_Computer
]
Columns = ["Age", "Income", "Student", "Credit", "Buys_Computer"]

Dataset = np.array(Dataset).T

# Q1, a
print("# ENTROPIES ##############################")
print("Entropies")
Entropies = []
for i in range(len(Columns)):
    OUT_INDEX = i
    print(f"Entropy of '{Columns[OUT_INDEX]}':")
    Data = Dataset[:, OUT_INDEX]
    entr = Entropy(Data)
    print(entr)
    print()
    Entropies.append(entr)
Entropies = np.array(Entropies)
print()
print("# ENTROPIES ##############################")

print("# AVERAGE ENTROPIES ##############################")
print("Average Entropies")
AvgEntropies = []
for i in range(len(Columns)):
    SPLIT_INDEX = i
    Dataset_Split, Values = SplitDatasetOnColumn(Dataset, SPLIT_INDEX)
    print(f"Average Entropy when split on '{Columns[SPLIT_INDEX]}':")
    Entropies_Split = []
    for i in range(len(Dataset_Split)):
        Entropies_Split.append(Entropy(Dataset_Split[i][:, OUT_INDEX]))
    Entropies_Split = np.array(Entropies_Split)
    # print(Entropies_Split)
    datasets_split_weights = np.array([dataset.shape[0] / Dataset.shape[0] for dataset in Dataset_Split])
    AvgEntrop_Split = np.mean(Entropies_Split * datasets_split_weights)
    print(AvgEntrop_Split)
    print()
    AvgEntropies.append(AvgEntrop_Split)
print("# AVERAGE ENTROPIES ##############################")
print()

# Q1, c
print("# INFO GAINS ##############################")
print("INFO GAINS")
InfoGains = []
for i in range(len(Columns)):
    SPLIT_INDEX = i
    InfoGain = InformationGain(Dataset, SPLIT_INDEX, OUT_INDEX)
    print(f"Information Gain when split on '{Columns[SPLIT_INDEX]}':")
    print(InfoGain)
    print()
    InfoGains.append(InfoGain)
InfoGains = np.array(InfoGains)

print("Maximum Information gain is for ")
MaxInfoGainIndex = np.argmax(InfoGains)
MaxInfoGain = InfoGains[MaxInfoGainIndex]
print(Columns[MaxInfoGainIndex], "with a value of", MaxInfoGain)