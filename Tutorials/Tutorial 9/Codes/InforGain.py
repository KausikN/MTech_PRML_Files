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
OUT_INDEX = 4

# Q1, a
print("### Q1 (A) ###############################")
print(f"Entropy of '{Columns[OUT_INDEX]}':")
BuysComputerData = Dataset[:, OUT_INDEX]
print(Entropy(BuysComputerData))

print("##########################################")
print()

# Q1, b
print("### Q1 (B) ###############################")
SPLIT_INDEX = 0
Dataset_AgeSplit, AgeValues = SplitDatasetOnColumn(Dataset, SPLIT_INDEX)
# print("Dataset Split on Age:")
# for i in range(len(Dataset_AgeSplit)):
#     print("Age " + str(AgeValues[i]) + ":")
#     print(Dataset_AgeSplit[i])
print(f"Average Entropy when split on '{Columns[SPLIT_INDEX]}':")
Entropies_AgeSplit = []
for i in range(len(Dataset_AgeSplit)):
    Entropies_AgeSplit.append(Entropy(Dataset_AgeSplit[i][:, OUT_INDEX]))
Entropies_AgeSplit = np.array(Entropies_AgeSplit)
# print(Entropies_AgeSplit)
datasets_split_weights = np.array([dataset.shape[0] / Dataset.shape[0] for dataset in Dataset_AgeSplit])
AvgEntrop_AgeSplit = np.sum(Entropies_AgeSplit * datasets_split_weights)
print(AvgEntrop_AgeSplit)

print("##########################################")
print()

# Q1, c
print("### Q1 (C) ###############################")
SPLIT_INDEX = 0
InfoGain_Age = InformationGain(Dataset, SPLIT_INDEX, OUT_INDEX)
print(f"Information Gain when split on '{Columns[SPLIT_INDEX]}':")
print(InfoGain_Age)

print("##########################################")
print()

# Q1, d
print("### Q1 (D) ###############################")
SPLIT_INDEX = 1
InfoGain_Income = InformationGain(Dataset, SPLIT_INDEX, OUT_INDEX)
print(f"Information Gain when split on '{Columns[SPLIT_INDEX]}':")
print(InfoGain_Income)

SPLIT_INDEX = 2
InfoGain_Student = InformationGain(Dataset, SPLIT_INDEX, OUT_INDEX)
print(f"Information Gain when split on '{Columns[SPLIT_INDEX]}':")
print(InfoGain_Student)

SPLIT_INDEX = 3
InfoGain_Credit = InformationGain(Dataset, SPLIT_INDEX, OUT_INDEX)
print(f"Information Gain when split on '{Columns[SPLIT_INDEX]}':")
print(InfoGain_Credit)

print("Maximum Information gain is for ")
InfoGains = [InfoGain_Age, InfoGain_Income, InfoGain_Student, InfoGain_Credit]
MaxInfoGainIndex = np.argmax(InfoGains)
MaxInfoGain = InfoGains[MaxInfoGainIndex]
print(Columns[MaxInfoGainIndex], "with a value of", MaxInfoGain)

print("##########################################")
print()

# Q1, e
print("### Q1 (E) ###############################")
SPLIT_INDEX = MaxInfoGainIndex

print("##########################################")
print()

# Q1, f
print("### Q1 (F) ###############################")
print("##########################################")