import numpy as np
import itertools

def error_covariance_R():
    R = np.zeros((10, 10))
    np.fill_diagonal(R, 0.1)
    R = np.power(R, 2)
    return R

def unique_R(R_original):
    numbers = []
    for i in range(len(R_original)):
        numbers.append(i)

    all_combinations = []

    # Generate combinations
    for r in range(1, len(numbers) + 1):
        combs = list(itertools.combinations(numbers, r))
        all_combinations.extend(combs)

    R_modified = []

    for r in range(1, 5):
        for combo in itertools.combinations(range(len(R_original)), r):
            matrix = R_original.copy()
            
            # Set large number for diagonals NOT in the combo
            for i in range(len(R_original)):
                if i not in combo:
                    matrix[i, i] = 1e20
            
            R_modified.append(matrix)
    return R_modified  