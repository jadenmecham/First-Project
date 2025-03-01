import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dt, k, m, c, process_var, measurement_var):
        self.dt = dt  # Time step
        
        # Continuous-time system dynamics
        A_c = np.array([[0, 1],
                        [-k/m, -c/m]])
        B_c = np.array([[0],
                        [1/m]])
        
        # Discretization using Euler method
        self.A = np.eye(2) + A_c * dt
        self.B = B_c * dt
        
        # Measurement matrix (measuring both position and velocity)
        self.H = np.array([[1, 0],
                           [0, 1]])
        
        # Process noise covariance
        self.Q = process_var * np.array([[dt**4/4, dt**3/2],
                                         [dt**3/2, dt**2]])
        
        # Measurement noise covariance (assuming independent noise for position and velocity)
        self.R = np.array([[measurement_var, 0],
                           [0, measurement_var]])
        
        # Initial state estimate and covariance matrix
        self.x = np.zeros((2, 1))  # Initial state [position, velocity]
        self.P = np.eye(2)  # Initial uncertainty
    
    def predict(self):
        # Predict state
        self.x = self.A @ self.x
        
        # Predict covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, z):
        # Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        y = z - self.H @ self.x  # Measurement residual
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self):
        return self.x.flatten()

# Simulate spring-mass system
np.random.seed(42)
dt = 0.1  # Time step
T = 10    # Total time
num_steps = int(T / dt)
k = 1.0   # Spring constant (arbitrary units)
m = 1.0   # Mass (arbitrary units)
c = 0.2   # Damping coefficient (arbitrary units)
process_var = 0.01  # Process noise
measurement_var = 0.1  # Measurement noise

# True states and measurements
x_true = np.zeros((2, num_steps))  # True position and velocity
z_meas = np.zeros((2, num_steps))  # Measured positions and velocities
x_true[:, 0] = [1.0, 0.0]  # Initial conditions

for i in range(1, num_steps):
    a = (-k / m * x_true[0, i-1]) - (c / m * x_true[1, i-1])  # Acceleration
    x_true[1, i] = x_true[1, i-1] + a * dt  # Update velocity
    x_true[0, i] = x_true[0, i-1] + x_true[1, i] * dt  # Update position
    
    # Simulate noisy measurements for both position and velocity
    z_meas[:, i] = x_true[:, i] + np.random.normal(0, np.sqrt(measurement_var), size=2)

# Apply Kalman Filter
kf = KalmanFilter(dt, k, m, c, process_var, measurement_var)
x_est = np.zeros((2, num_steps))

for i in range(num_steps):
    kf.predict()
    kf.update(z_meas[:, i].reshape(-1, 1))
    x_est[:, i] = kf.get_state()

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(np.arange(num_steps) * dt, x_true[0, :], label='True Position', linestyle='dashed')
plt.plot(np.arange(num_steps) * dt, z_meas[0, :], label='Measured Position', linestyle='dotted', alpha=0.5)
plt.plot(np.arange(num_steps) * dt, x_est[0, :], label='Kalman Filter Estimate', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.legend()
plt.title('Kalman Filter for Spring-Mass System')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.arange(num_steps) * dt, x_true[1, :], label='True Velocity', linestyle='dashed')
plt.plot(np.arange(num_steps) * dt, z_meas[1, :], label='Measured Velocity', linestyle='dotted', alpha=0.5)
plt.plot(np.arange(num_steps) * dt, x_est[1, :], label='Kalman Filter Estimate', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
