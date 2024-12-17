import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

# Define system parameters
m = 1   # mass of the pendulum
M = 5   # mass of the cart
L = 2   # length of the pendulum
g = -9.81  # gravity
d = 1   # damping
s = -1  # starting position

# State-space matrices
A = np.array([[0, 1, 0, 0], 
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B = np.array([[0],
              [1/M],
              [0],
              [s*1/(M*L)]])

C = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0],  
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

D = np.zeros((4, 1))  # Correct dimensions for D

# Create the state-space system
system = StateSpace(A, B, C, D)

# Time vector and input signal
t = np.linspace(0, 10, 1000)  # 0 to 10 seconds
u = np.ones_like(t)          # Zero input

# Initial state
x0 = np.array([0, 0, np.pi/4, 0])  # Pendulum starts at 45 degrees (in radians)

# Simulate the system
t_out, y_out, x_out = lsim(system, U=u, T=t, X0=x0)

# Plot the states on separate subplots
num_states = y_out.shape[1]  # Number of states
fig, axes = plt.subplots(num_states, 1, figsize=(8, 2 * num_states), sharex=True)

state = ['Position (x)', 'Velocity (dx)', 'Angle (theta)', 'Angular Velocity (dtheta)']
for i, ax in enumerate(axes):
    ax.plot(t_out, y_out[:, i], label='True Value')
    ax.set_ylabel(state[i])
    ax.grid()
    ax.legend()

axes[-1].set_xlabel('Time (s)')
fig.suptitle('State Responses Over Time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
plt.show()

