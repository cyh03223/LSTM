# data generation for simple harmonic motion
import numpy as np
import scipy.integrate
import random

random.seed(32)
np.random.seed(32)

print("Generating SHM data")

# Simulation variables
seq_length = 1001
num_records = 50
dim_x = 2  # States: position and velocity
dim_y = 1  # Observable: position only
dt = 0.1
end_t = seq_length * dt

# Arrays to store results
X_data_array = np.empty((num_records, seq_length, dim_x))
Y_data_array = np.empty((num_records, seq_length, dim_y))

# SHM dynamics: dx/dt = v, dv/dt = -k/m * x
def shm_dynamics(t, x, k=1.0, m=1.0):
    pos, vel = x
    dxdt = [vel, -k/m * pos]
    return dxdt

# Process and measurement noise
P = 0.0001 * np.identity(dim_x)  # Covariance of initial condition
mu_pn = np.zeros(dim_x)
Q = 0.01 * np.identity(dim_x)  # Process noise covariance
mu_mn = np.zeros(dim_y)
R = [0.01]  # Measurement noise covariance
G = np.identity(dim_x)  # Process noise matrix
H = np.array([[1, 0]])  # Observation matrix (observe position only)

t_span = np.arange(0, end_t, dt)

for i in range(num_records):
    mu_x0 = np.random.uniform(-2, 2, size=dim_x)
    print(mu_x0)
    x0 = np.random.multivariate_normal(mu_x0, P)
    sol = scipy.integrate.solve_ivp(shm_dynamics, (0, end_t), x0,
                                    t_eval=t_span, args=(), method='RK45')
    
    X_data_array[i, :, :] = np.transpose(sol.y)
    
    for j in range(1, seq_length):
        w = np.random.multivariate_normal(mu_pn, Q)
        X_data_array[i, j, :] += G @ w
        v = np.random.normal(mu_mn, R)
        Y_data_array[i, j, :] = X_data_array[i, j, :] @ H.T + v

np.savez("shm_data.npz", X_data=X_data_array, Y_data=Y_data_array)
