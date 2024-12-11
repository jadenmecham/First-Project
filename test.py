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

C = np.array([
    [1, 0, 0, 0],  # Measure cart position
    [0, 1, 0, 0],  # Measure cart velocity
    [0, 0, 1, 0]   # Measure pendulum angle
])

# C = np.array([[1,0,0,0],
#              [0,1,0,0],
#              [0,0,1,0],
#              [0,0,0,1],
#              [2,0,0,0],
#              [0,2,0,0],
#              [0,0,2,0],
#              [0,0,0,2],
#              [1,0,1,0],
#              [0,1,0,1]])

D = np.zeros((4,1))

# define the system 
def state_space(t, x, u):
    xdot = A @ x + B.flatten() * u
    return xdot

# Kalman filter update
def kalman_filter(y, x_hat, u):
    global P
    # Predict
    x_hat_minus = A @ x_hat + B.flatten() * u
    P_minus = A @ P @ A.T + Q

    # Update
    K = P_minus @ C.T @ np.linalg.inv(C @ P_minus @ C.T + R)
    y_tilde = y - C @ x_hat_minus
    x_hat = x_hat_minus + K @ y_tilde
    P = (np.eye(len(P)) - K @ C) @ P_minus
    return x_hat

# Simulation parameters
t_span = (0, 10)  # Time range for simulation
x0 =  [0, 0, np.pi / 4, 0]  # Initial state: [x, dx, theta, dtheta]
u = 1  # input force

# Kalman filter parameters
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])  # Process noise covariance
R = np.diag([1e-3, 1e-3, 1e-3])  # Measurement noise covariance
P = np.eye(4)  # Initial error covariance
x_hat = np.zeros(4)  # Initial state estimate

# Measurement noise
np.random.seed(42)  # For reproducibility
measurement_noise = lambda: np.random.multivariate_normal([0, 0, 0], R)

# Solve system
t_eval = np.linspace(0, 10, 1000)
sol = solve_ivp(state_space, t_span, x0, args=(u,), t_eval=t_eval)

# Kalman filter estimation
x_hat_history = []
for t, x_true in zip(sol.t, sol.y.T):
    # Simulate noisy measurements
    y = C @ x_true + measurement_noise()

    # Apply Kalman filter
    x_hat = kalman_filter(y, x_hat, u)
    x_hat_history.append(x_hat)

x_hat_history = np.array(x_hat_history)

# Plot results
plt.figure(figsize=(12, 8))
state_labels = ['Cart Position (x)', 'Cart Velocity (dx)', 'Pendulum Angle (theta)', 'Pendulum Angular Velocity (dtheta)']

for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(sol.t, sol.y[i], label=f"True {state_labels[i]}")
    plt.plot(sol.t, x_hat_history[:, i], '--', label=f"Estimated {state_labels[i]}")
    plt.xlabel("Time (s)")
    plt.ylabel(state_labels[i])
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()