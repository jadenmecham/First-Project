import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm
from scipy.signal import cont2discrete

# State of this file: More complex cart pend example where all states are measured and used to estimate all states. No C groups. Works well. 

# Define system parameters
m = 1 # mass of the pend
M = 5 # mass of the cart
L = 2 # length of the pendulum 
g = -9.81 # gravity 
d = 1 # damping
s = -1 # starting position

# State-space matrices
A_c = np.array([[0, 1, 0, 0], 
              [0, -d/M, -m*g/M, 0],
              [0, 0, 0, 1],
              [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]])

B_c = np.array([[0],
             [1/M],
             [0],
             [s*1/(M*L)]])

# measurements
C = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0],  
        [0, 0, 1, 0],
        [0, 0, 0, 1]   
    ])

# Process and measurement noise
Q = np.eye(4) * 1e-4
R = np.eye(4) * 0.05

# discrete time
dt = 0.1  # Time step
system = (A_c, B_c, np.eye(4), 0)
A_d, B_d, _, _, _ = cont2discrete(system, dt)

# Initial conditions
x0 = [0.1, 0, np.pi / 4, 0.1] # Initial state: [x, dx, theta, dtheta] # Initial displacement and velocity

# Kalman Filter setup
kf = KalmanFilter(dim_x=4, dim_z=4)
kf.F = A_d  # Discretized A matrix
kf.H = C  # Measurement matrix C
kf.Q = Q  # Process noise covariance Q
kf.R = R  # Measurement noise covariance R
kf.P = np.eye(4) * 500  # Initial state covariance P
kf.x = x0  # Initial state estimate

# Simulate the system
np.random.seed(42)
n_steps = 100 
true_states = []
measurements = []
for _ in range(n_steps):
    # Simulate true dynamics
    x_true = A_d @ x0 + np.random.multivariate_normal([0, 0, 0, 0], Q).T
    true_states.append(x_true)

    # Simulate noisy measurements
    z = C @ x_true + np.random.multivariate_normal([0, 0, 0, 0], R).T
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
    state_uncertainties.append(np.sqrt(np.diag(kf.P)))  # Standard deviation (sqrt of variance)

filtered_states = np.array(filtered_states)
state_uncertainties = np.array(state_uncertainties)

#Create a time vector based on the number of steps and time step size
time = np.arange(0, n_steps * dt, dt)  # Total simulation time

# Plot results with confidence intervals and time in seconds
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Position
axs[0].plot(time, true_states[:, 0], label="True Position", color="blue")
axs[0].plot(time, filtered_states[:, 0], label="Estimated Position", linestyle="--", color="green")
axs[0].fill_between(time, 
                    filtered_states[:, 0] - 2 * state_uncertainties[:, 0],
                    filtered_states[:, 0] + 2 * state_uncertainties[:, 0],
                    color="green", alpha=0.2, label="95% CI (Position)")
axs[0].set_ylabel("Position (m)")
axs[0].set_title("Spring-Mass-Damper System: Kalman Filter with Confidence Intervals")
axs[0].legend()
axs[0].grid()

# Velocity
axs[1].plot(time, true_states[:, 1], label="True Velocity", color="blue")
axs[1].plot(time, filtered_states[:, 1], label="Estimated Velocity", linestyle="--", color="green")
axs[1].fill_between(time, 
                    filtered_states[:, 1] - 2 * state_uncertainties[:, 1],
                    filtered_states[:, 1] + 2 * state_uncertainties[:, 1],
                    color="green", alpha=0.2, label="95% CI (Velocity)")
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_xlabel("Time (s)")  # Label the x-axis with time
axs[1].legend()
axs[1].grid()

# Anglular position
axs[2].plot(time, true_states[:, 2], label="True Angular Position", color="blue")
axs[2].plot(time, filtered_states[:, 2], label="Estimated Angular Position", linestyle="--", color="green")
axs[2].fill_between(time, 
                    filtered_states[:, 2] - 2 * state_uncertainties[:, 2],
                    filtered_states[:, 2] + 2 * state_uncertainties[:, 2],
                    color="green", alpha=0.2, label="95% CI (Velocity)")
axs[2].set_ylabel("Angular Position")
axs[2].set_xlabel("Time (s)")  # Label the x-axis with time
axs[2].legend()
axs[2].grid()

# Anglular velocity
axs[3].plot(time, true_states[:, 3], label="True Angular Velocity", color="blue")
axs[3].plot(time, filtered_states[:, 3], label="Estimated Angular Velocity", linestyle="--", color="green")
axs[3].fill_between(time, 
                    filtered_states[:, 3] - 2 * state_uncertainties[:, 3],
                    filtered_states[:, 3] + 2 * state_uncertainties[:, 3],
                    color="green", alpha=0.2, label="95% CI (Velocity)")
axs[3].set_ylabel("Angular Velocity")
axs[3].set_xlabel("Time (s)")  # Label the x-axis with time
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()