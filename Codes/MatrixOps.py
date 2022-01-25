'''
Matrix Operations
'''

# Imports
import prettytable as pt
import numpy as np

# Main Functions
def MatrixMultiply(A, B):
    '''
    Matrix Multiplication
    '''
    return np.dot(A, B)

def MatrixChainMultiply(Ms):
    '''
    Matrix Chain Multiplication
    '''
    Solution = Ms[0]
    for i in range(1, len(Ms)):
        Solution = MatrixMultiply(Solution, Ms[i])
    return Solution

def MatrixInverse(M):
    '''
    Matrix Inverse
    '''
    return np.linalg.inv(M)

def MatrixDeterminant(M):
    '''
    Matrix Determinant
    '''
    return np.linalg.det(M)

# Display Functions
def DisplayMatrix(M):
    '''
    Display Matrix
    '''
    Table = pt.PrettyTable()
    for i in range(len(M)):
        Table.add_row(M[i])
    print(Table)

def DisplayMatrix_LaTeX(M):
    '''
    Display Matrix in LaTeX Code
    '''
    code = "\\begin{bmatrix}"
    lines = []
    for i in range(len(M)):
        row = ' & '.join(list(map(str, M[i])))
        lines.append(row)
    code += ' \\\\ '.join(lines)
    code += "\\end{bmatrix}"
    print(code)


# Driver Code
# Params
Matrix = [
    [13, 9, -1],
    [9, 7, -3],
    [-1, -3, 7]
]
# Params

# RunCode
Det = MatrixDeterminant(Matrix)
print("Determinant:", Det)
Solution = MatrixInverse(Matrix)
DisplayMatrix(Solution)