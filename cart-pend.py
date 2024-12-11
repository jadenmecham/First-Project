import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system parameters
m = 1 # mass of the pend
M = 5 # mass of the cart
L = 2 # length of the pendulum 
g = -9.81 # gravity 
d = 1 # damping
s = -1 # starting position

# State-space matrices
A = np.array([[0, 1, 0, 0], 
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B = np.array([[0],
             [1/M],
             [0],
             [s*1/(M*L)]])

# Define 5 different C matrices 
C_matrices = [
    np.array([
        [1, 0, 0, 0],  # Measure cart position
        [0, 1, 0, 0],  # Measure cart velocity
        [0, 0, 1, 0]   # Measure pendulum angle
    ]),
    np.array([
        [0, 0, 1, 0],  # Measure cart position
        [0, 0, 0, 1],  # Measure pendulum angle
        [2, 0, 0, 0]   # Measure pendulum angular velocity
    ]),
    np.array([
        [2, 0, 0, 0],  # Measure cart position
        [0, 2, 0, 0],  # Measure cart velocity
        [0, 0, 2, 0]   # Measure pendulum angular velocity
    ]),
    np.array([
        [0, 0, 2, 0],  # Measure cart velocity
        [0, 0, 0, 2],  # Measure pendulum angle
        [1, 0, 1, 0]   # Measure pendulum angular velocity
    ]),
    np.array([
        [1, 0, 1, 0],  # Measure cart position
        [0, 1, 0, 1],  # Measure cart velocity
        [1, 1, 0, 0]   # Measure pendulum angle
    ])
]

# State-space function
def state_space(t, x, u):
    dxdt = A @ x + B.flatten() * u
    return dxdt

# Kalman filter function
def kalman_filter(y, x_hat, P, C, u, R):
    # Predict
    x_hat_minus = A @ x_hat + B.flatten() * u
    P_minus = A @ P @ A.T + Q

    # Update
    K = P_minus @ C.T @ np.linalg.inv(C @ P_minus @ C.T + R)
    y_tilde = y - C @ x_hat_minus
    x_hat = x_hat_minus + K @ y_tilde
    P = (np.eye(len(P)) - K @ C) @ P_minus
    return x_hat, P

# Process and measurement noise covariances
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])  # Process noise covariance
R = np.eye(3) * 1e-2                  # Measurement noise covariance 

# Simulation parameters
t_span = (0, 10)
x0 = [0.1, 0, np.pi / 4, 0.1]  # Initial state: [x, dx, theta, dtheta]
u = 1

# Solve the system
t_eval = np.linspace(0, 10, 10000)
sol = solve_ivp(state_space, t_span, x0, args=(u,), t_eval=t_eval)

# Simulate measurements and estimate states for each C matrix
results = []
for C in C_matrices:
    x_hat = np.zeros(4)  # Initial state estimate
    P = np.eye(4)        # Initial error covariance
    x_hat_history = []

    for t, x_true in zip(sol.t, sol.y.T):
        # Generate noisy measurements
        measurement_noise = np.random.multivariate_normal(np.zeros(3), R)
        y = C @ x_true + measurement_noise

        # Apply Kalman filter
        x_hat, P = kalman_filter(y, x_hat, P, C, u, R)
        x_hat_history.append(x_hat)

    results.append(np.array(x_hat_history))

# Plot the results
plt.figure(figsize=(16, 12))
state_labels = ['Cart Position (x)', 'Cart Velocity (dx)', 'Pendulum Angle (theta)', 'Pendulum Angular Velocity (dtheta)']

for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(sol.t, sol.y[i], label="True State", linewidth=2)

    for j, x_hat_history in enumerate(results):
        plt.plot(sol.t, x_hat_history[:, i], '--', label=f"Estimated State (C{j+1})")

    plt.xlabel("Time (s)")
    plt.ylabel(state_labels[i])
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
