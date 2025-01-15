import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm
from scipy.signal import cont2discrete

# Current state of this file: 4 C groups. 1 sensor is messed up on purpose to show that 2 C groups will have bad data from this and don't agree with the other sensors. 

n_steps = 100
dt = 0.1  # Time step

# Create state-space matrices
def createStateSpace():
    m = 1  # Mass of the pendulum
    M = 5  # Mass of the cart
    L = 2  # Length of the pendulum
    g = -9.81  # Gravity
    d = 1  # Damping
    s = -1  # Starting position

    # State-space matrices
    A = np.array([[0, 1, 0, 0],
                  [0, -d / M, -m * g / M, 0],
                  [0, 0, 0, 1],
                  [0, -s * d / (M * L), -s * (m + M) * g / (M * L), 0]])

    B = np.array([[0],
                  [1 / M],
                  [0],
                  [s * 1 / (M * L)]])

    # Define multiple C matrices
    C = [
        np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),  # Measuring position and velocity of the cart
        np.array([[3, 10, 7, 5], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),  # Measuring angle and angular velocity of the pendulum
        np.array([[3, 10, 7, 5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),  # Measuring velocity of the cart and angle of the pendulum
        np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),  # Measuring position of the cart and angular velocity of the pendulum
    ]

    # Define true C matrix
    Ctrue = np.eye(4)
    return A, B, C, Ctrue

# Simulate the real system
def simulateSS(A_c, B_c, C, x0):
    system = (A_c, B_c, np.eye(4), 0)
    A_d, B_d, _, _, _ = cont2discrete(system, dt)
    Q = np.eye(4) * 1e-4
    R = np.eye(4) * 0.05
    np.random.seed(42)
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

    return np.array(true_states), np.array(measurements)

# Kalman Filter Estimation
def kfEstimation(C, measurements, A_c, B_c):
    filtered_states = []
    state_uncertainties = []

    for i in range(len(C)):  # Iterate over every different C
        # Process and measurement noise
        system = (A_c, B_c, np.eye(4), 0)
        A_d, B_d, _, _, _ = cont2discrete(system, dt)
        Q = np.eye(4) * 1e-4
        R = np.eye(C[i].shape[0]) * 0.05

        # Initialize Kalman Filter
        kf = KalmanFilter(dim_x=4, dim_z=C[i].shape[0])
        kf.F = A_d  # Discretized A matrix
        kf.H = C[i]  # Measurement matrix C
        kf.Q = Q  # Process noise covariance Q
        kf.R = R  # Measurement noise covariance R
        kf.P = np.eye(4) * 500  # Initial state covariance P
        kf.x = x0  # Initial state estimate

        filstate = []
        stateunc = []

        for z in measurements:
            kf.predict()
            kf.update(z)
            filstate.append(kf.x.copy())
            stateunc.append(np.sqrt(np.diag(kf.P)))  # Standard deviation (sqrt of variance)

        filtered_states.append(filstate)
        state_uncertainties.append(stateunc)

    return filtered_states, state_uncertainties


def plotResults(true_states, filtered_states):
    state_labels = ["Position (x)", "Velocity (dx)", "Angle (theta)", "Angular Velocity (dtheta)"]
    colors = ['red', 'blue', 'orange', 'purple']
    time = np.arange(true_states.shape[0]) * 0.1

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time, true_states[:, i], label="True State", linewidth=2)

        for j, est in enumerate(filtered_states):
            est_array = np.array(est)
            ax.plot(time, est_array[:, i], label=f"C{j+1}", linestyle="--", color = colors[j])

        ax.set_title(f"State {i+1}: {state_labels[i]}")
        ax.set_ylabel(state_labels[i])
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# Initial conditions
x0 = np.array([0.1, 0, np.pi / 4, 0.1])  # Initial state: [x, dx, theta, dtheta]

# Run simulation and estimation
A_c, B_c, C_matrices, C_true = createStateSpace()
true_states, measurements = simulateSS(A_c, B_c, C_true, x0)
kf_states, state_unc = kfEstimation(C_matrices, measurements, A_c, B_c)

# Plot results
plotResults(true_states, kf_states)


