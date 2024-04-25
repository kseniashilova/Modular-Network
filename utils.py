import numpy as np


def save_matrix(matrix, path):
    np.save(path, matrix)


def load_matrix(path):
    # Load the matrix
    loaded_matrix = np.load(path)
    print(loaded_matrix)
