import numpy as np

# file for testing random stuff


C = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [2,0,0,0],
    [0,2,0,0],
    [0,0,2,0],
    [0,0,0,2],
    [1,0,1,0],
    [0,1,0,1],
    [1,1,0,0]])

C2 = np.array([ C[0], C[1] ])

C3 = np.array([C[1], C[2]])

Cgather = [C2, C3]
print(Cgather[0])
