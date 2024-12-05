import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import control

m = 1
M = 5
L = 2
g = -10
d = 1
s = -1

A = np.array([[0, 1, 0, 0], 
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B = np.array([[0],
             [1/M],
             [0],
             [s*1/(M*L)]])

C = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [2,0,0,0],
              [0,2,0,0]
              [0,0,2,0]
              [0,0,0,2]
              [1,0,1,0],
              [0,1,0,1]])

D = np.array[0]

