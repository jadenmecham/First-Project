import numpy as np

C = np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]])
C1 = np.array([C[0],C[1],C[2]])
C2 = np.array([C[1],C[2],C[3]])
C3 = np.array([C[0],C[1],C[3]])
C4 = np.array([C[0],C[2],C[3]])
print(C1)
print(C2)
print(C3)
print(C4)
