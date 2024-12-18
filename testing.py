import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm
from scipy.signal import cont2discrete

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

# Process and measurement noise
Q = np.eye(4) * 1e-4
R = np.eye(4) * 0.05

# discrete time
dt = 0.1  # Time step
system = (A_c, B_c, np.eye(4), 0)
A_d, B_d, _, _, _ = cont2discrete(system, dt)

# Initial conditions
x0 = [0.1, 0, np.pi / 4, 0.1] # Initial state: [x, dx, theta, dtheta] # Initial displacement and velocity

# Simulate the system
np.random.seed(42)
n_steps = 100 
true_states = []
measurements = []

# Simulate the system to compute true states
true_states = []
x0 = [0.1, 0, np.pi / 4, 0.1]  # Initial state
for _ in range(n_steps):
    x_true = A_d @ x0 + np.random.multivariate_normal([0, 0, 0, 0], Q).T
    true_states.append(x_true)
    x0 = x_true  # Update state for next iteration

# Convert true states to numpy array
true_states = np.array(true_states)

# Create a time vector
time = np.arange(0, n_steps * dt, dt)  # Total simulation time

# Define multiple C matrices
C_matrices = [
    np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),  # Observes position and velocity
    np.array([[0, 0, 1, 0], [0, 0, 0, 1]]),  # Observes angular position and velocity
    np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),  # Observes position and angular position
]

# Store results for all C matrices
results = []

for C in C_matrices:
    # Reinitialize the Kalman Filter for each C matrix
    kf = KalmanFilter(dim_x=4, dim_z=C.shape[0])
    kf.F = A_d  # Discretized A matrix
    kf.H = C  # Measurement matrix C
    kf.Q = Q  # Process noise covariance Q
    kf.R = R[:C.shape[0], :C.shape[0]]  # Adjust R size to match measurement dimension
    kf.P = np.eye(4) * 500  # Initial state covariance P
    kf.x = [0.1, 0, np.pi / 4, 0.1]  # Initial state estimate

    # Simulate measurements and filtering
    measurements = []
    filtered_states = []
    state_uncertainties = []
    
    x0 = [0.1, 0, np.pi / 4, 0.1]  # Reset initial state for each run
    for _ in range(n_steps):
        x_true = A_d @ x0 + np.random.multivariate_normal([0, 0, 0, 0], Q).T
        z = C @ x_true + np.random.multivariate_normal([0] * C.shape[0], R[:C.shape[0], :C.shape[0]]).T
        measurements.append(z)
        
        kf.predict()
        kf.update(z)
        filtered_states.append(kf.x.copy())
        state_uncertainties.append(np.sqrt(np.diag(kf.P)))
        
        x0 = x_true

    # Convert results to numpy arrays
    filtered_states = np.array(filtered_states)
    state_uncertainties = np.array(state_uncertainties)
    measurements = np.array(measurements)

    results.append({
        'C': C,
        'filtered_states': filtered_states,
        'state_uncertainties': state_uncertainties,
        'measurements': measurements
    })

# Plotting results
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
colors = ['blue', 'orange', 'green']
for i, res in enumerate(results):
    filtered_states = res['filtered_states']
    state_uncertainties = res['state_uncertainties']
    label_prefix = f"C Matrix {i+1}"
    
    for state_idx, state_label in enumerate(["Position", "Velocity", "Angular Position", "Angular Velocity"]):
        # Plot the true state
        axs[state_idx].plot(
            time, true_states[:, state_idx], label=f"True {state_label}", color="black", linestyle="-")
        
        # Plot the filtered state estimate
        axs[state_idx].plot(
            time, filtered_states[:, state_idx], label=f"{label_prefix} Estimate", linestyle="--", color=colors[i])
        
        # Plot the confidence intervals for the filtered state
        axs[state_idx].fill_between(
            time, 
            filtered_states[:, state_idx] - 2 * state_uncertainties[:, state_idx],
            filtered_states[:, state_idx] + 2 * state_uncertainties[:, state_idx],
            color=colors[i], alpha=0.2)

        axs[state_idx].set_ylabel(state_label)
        axs[state_idx].grid()

# Add labels and legends
axs[-1].set_xlabel("Time (s)")
axs[0].set_title("Comparison of True States and Kalman Filter Estimates with Different C Matrices")
axs[0].legend()
plt.tight_layout()
plt.show()
