import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import control
from scipy.integrate import solve_ivp

m = 1 # mass of the pend
M = 5 # mass of the cart
L = 2 # length of the pendulum 
g = -9.81 # gravity 
d = 1 # damping
s = -1 # starting position

# state space matrices
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
              [0,2,0,0],
              [0,0,2,0],
              [0,0,0,2],
              [1,0,1,0],
              [0,1,0,1]])

D = np.zeros((4,1))

# define the system 
def state_space(t, x, u):
    xdot = A @ x + B.flatten() * u
    return xdot

# Simulation parameters
t_span = (0, 10)  # Time range for simulation
x0 =  [0,0,0,0]#[0, 0, np.pi / 4, 0]  # Initial state: [x, dx, theta, dtheta]
u = 1  # input force

# Solve the system using ODE solver
sol = solve_ivp(state_space, t_span, x0, args=(u,), t_eval=np.linspace(0, 10, 100))

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], label="Cart Position (x)")
plt.plot(sol.t, sol.y[2], label="Pendulum Angle (theta)")
plt.xlabel("Time (s)")
plt.ylabel("Position/Angle")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[1], label="Cart Velocity (dx)")
plt.plot(sol.t, sol.y[3], label="Pendulum Angular Velocity (dtheta)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()