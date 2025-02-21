import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm

# State of this file: Simple spring mass damper system to test that the Kalman filter works for an easy system before adding too many other things. 

# Spring-mass-damper parameters
m = 1.0  # Mass (kg)
c = 0.5  # Damping coefficient (Ns/m)
k = 2.0  # Spring constant (N/m)

# Continuous-time state-space matrices
A_c = np.array([[0, 1],
                [-k/m, -c/m]])
B_c = np.array([[0],
                [1/m]])
C = np.array([[1, 0],   # Measuring position
              [0, 1],   # Measuring velocity
              [1, 1]])  # New measurement (sum of position and velocity)

# Process and measurement noise
Q = np.array([[1e-4, 0], 
              [0, 1e-4]])  # Process noise covariance

# Discrete time system for kalman filter
dt = 0.1  # Time step
A_d = expm(A_c * dt)
B_d = np.linalg.solve(A_c, (A_d - np.eye(2))) @ B_c

# Initial conditions
x0 = np.array([1.0, 0.0])  # Initial displacement and velocity

# Kalman Filter setup
kf = KalmanFilter(dim_x=2, dim_z=3)  # Changed dim_z to 3
kf.F = A_d  # Discretized A matrix
kf.H = C  # Measurement matrix C
kf.Q = Q  # Process noise covariance Q
R = np.array([[0.05, 0, 0], 
              [0, 0.05, 0],
              [0, 0, 0.05]])  # New row and column for extra measurement noise
kf.R = R

kf.P = np.eye(2) * 500  # Initial state covariance P
kf.x = x0  # Initial state estimate

# Simulate the system
np.random.seed(42)
n_steps = 100 
true_states = []
measurements = []
for _ in range(n_steps):
    # Simulate true dynamics
    x_true = A_d @ x0 + np.random.multivariate_normal([0, 0], Q).T
    true_states.append(x_true)

    # Simulate noisy measurements
    z = C @ x_true + np.random.multivariate_normal([0, 0, 0], R).T
    measurements.append(z)

    # Update for the next time step
    x0 = x_true

true_states = np.array(true_states)
measurements = np.array(measurements)

# Run Kalman Filter and collect uncertainties
filtered_states = []
state_uncertainties = []  # To store standard deviations for each state
for z in measurements:
    kf.predict()
    kf.update(z)
    filtered_states.append(kf.x.copy())
    state_uncertainties.append(3*np.sqrt(np.diag(kf.P)))  # 3 sigma bound 

filtered_states = np.array(filtered_states)
state_uncertainties = np.array(state_uncertainties)

#Create a time vector based on the number of steps and time step size
time = np.arange(0, n_steps * dt, dt)  # Total simulation time

# Plot results with confidence intervals and time in seconds
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Position
axs[0].plot(time, true_states[:, 0], label="True Position", color="blue")
axs[0].plot(time, filtered_states[:, 0], label="Estimated Position", linestyle="--", color="green")
axs[0].fill_between(time, 
                    filtered_states[:, 0] - state_uncertainties[:, 0],
                    filtered_states[:, 0] + state_uncertainties[:, 0],
                    color="green", alpha=0.2, label="95% CI (Position)")
axs[0].set_ylabel("Position (m)")
axs[0].set_title("Spring-Mass-Damper System: Kalman Filter with Confidence Intervals")
axs[0].legend()
axs[0].grid()

# Velocity
axs[1].plot(time, true_states[:, 1], label="True Velocity", color="blue")
axs[1].plot(time, filtered_states[:, 1], label="Estimated Velocity", linestyle="--", color="green")
axs[1].fill_between(time, 
                    filtered_states[:, 1] - state_uncertainties[:, 1],
                    filtered_states[:, 1] + state_uncertainties[:, 1],
                    color="green", alpha=0.2, label="95% CI (Velocity)")
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_xlabel("Time (s)")  # Label the x-axis with time
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()