#Kuramoto Sivashinsky equation comparisons for 41 dimensions(20 Nodes) 
import numpy as np
import sympy as sp

# Set random seeds for reproducibility
np.random.seed(1)

# Parameters
N = 20  # Number of Fourier modes (so dim_x = 2N + 1)
dim_x = 2 * N + 1
dim_y = N  # Number of measurements
L = 2*np.pi
c, d = -L, L
dt = 1e-6
seq_length = 3001
num_records = 50

# Physical parameter
nu = 0.01

# Linear eigenvalues
lambda_array = [(i ** 2) - (nu * (i ** 4)) for i in range(1, N + 1)]

# Measurement matrix H
m_points = c + (d - c) * np.random.random_sample(dim_y)
H = np.zeros((dim_y, dim_x))
for i in range(dim_y):
    H[i, 0] = 1
    for j in range(N):
        H[i, j + 1] = np.cos((j + 1) * 2 * np.pi * m_points[i] / L)
        H[i, j + N + 1] = np.sin((j + 1) * 2 * np.pi * m_points[i] / L)

# Noise parameters
sigma_p = 0.0001
Q = np.diag((sigma_p ** 2) * np.ones(dim_x))
R = np.diag((0.01 ** 2) * np.ones(dim_y))
P = np.diag((sigma_p ** 2) * np.ones(dim_x))

def ks_galerkin_rhs(x, N, nu):
    """
    Compute RHS of 21D Galerkin approximation of the KS equation.
    x: state vector [a0, a1, ..., aN, b1, ..., bN] of length 2N + 1
    Returns dx/dt vector of same shape
    """
    assert len(x) == 2 * N + 1
    a0 = x[0]
    a = x[1:N+1]
    b = x[N+1:]

    # Preallocate derivative
    dxdt = np.zeros_like(x)
    dxdt[0] = 0  # a0 is constant due to zero mean in KS

    # Linear part
    for k in range(1, N+1):
        L_k = -(nu * (k**2) + (k**4))
        dxdt[k] = L_k * a[k-1]
        dxdt[N + k] = L_k * b[k-1]

    # Nonlinear part (convolution sum)
    for n in range(1, N+1):
        sum_cos = 0
        sum_sin = 0
        for k in range(1, N+1):
            for m in range(1, N+1):
                if abs(k - m) == n:
                    sum_cos += 0.5 * (a[k-1]*a[m-1] + b[k-1]*b[m-1])
                    sum_sin += a[k-1]*b[m-1]
                if (k + m) == n:
                    sum_cos += 0.5 * (a[k-1]*a[m-1] - b[k-1]*b[m-1])
                    sum_sin += a[k-1]*b[m-1]

        dxdt[n] += -n * sum_sin
        dxdt[N + n] += n * sum_cos

    return dxdt

# Euler integrator
def step_forward(x, N, nu):
    return x + dt * ks_galerkin_rhs(x, N, nu)

# Generate data
X_data_array = np.empty((num_records, seq_length, dim_x))
Y_data_array = np.empty((num_records, seq_length, dim_y))

#initial condition
z = sp.symbols('z')
u0 = 5*z - 0.5 * z**2 - 4

# Orthonormal basis for calculating Fourier coefficients
phi = [1/sp.sqrt(2*sp.pi)]  # for a0
psi = []

for i in range(1, N+1):  # for a1, ..., aN and b1, ..., bN #check this
    phi.append((1/sp.sqrt(sp.pi)) * sp.cos(z*i))
    psi.append((1/sp.sqrt(sp.pi)) * sp.sin(z*i))

# Initial condition for the ODE system projected to the basis by taking L^2-inner product with phi_n and psi_n
a0 = [0] * (N + 1)
a0[0] = sp.integrate(u0*phi[0], (z, -sp.pi, sp.pi)).evalf()

c0 = [0] * N
for i in range(N):
    integ1 = sp.integrate(u0*phi[i+1], (z, -sp.pi, sp.pi)).evalf()
    integ2 = sp.integrate(u0*psi[i], (z, -sp.pi, sp.pi)).evalf()
    a0[i+1] = integ1  
    c0[i] = integ2  

x0_list = [a0, c0]

x0_flat_list = [item for sublist in x0_list for item in sublist]
x0_numpy=np.asarray(x0_flat_list)
x0=x0_numpy.astype(np.float64)

for i in range(num_records):
    # perturbed_mean = x0_mean + np.random.uniform(-2, 2, size=dim_x)
    perturbed_mean=np.random.uniform(x0-0.2,x0+0.2,size=dim_x)
    x = np.random.multivariate_normal(perturbed_mean, P)

    X_data_array[i, 0] = x
    Y_data_array[i, 0] = H @ x + np.random.multivariate_normal(np.zeros(dim_y), R)

    for j in range(1, seq_length):
        w = np.random.multivariate_normal(np.zeros(dim_x), Q)
        x = step_forward(x, N, nu) + w
        X_data_array[i, j] = x
        Y_data_array[i, j] = H @ x + np.random.multivariate_normal(np.zeros(dim_y), R)

np.savez('ks_data.npz', X_data=X_data_array, Y_data=Y_data_array)
np.savez('matrix_h_ks.npz',H=H)

# Generate data
X_data_array_diff_ic = np.empty((num_records, seq_length, dim_x))
Y_data_array_diff_ic = np.empty((num_records, seq_length, dim_y))

for i in range(num_records):
    perturbed_mean_2=np.random.uniform(x0-0.5,x0+0.5,size=dim_x)
    x_2 = np.random.multivariate_normal(perturbed_mean_2, P)
    X_data_array_diff_ic[i, 0] = x_2
    Y_data_array_diff_ic[i, 0] = H @ x_2 + np.random.multivariate_normal(np.zeros(dim_y), R)

    for j in range(1, seq_length):
        w = np.random.multivariate_normal(np.zeros(dim_x), Q)
        x_2 = step_forward(x_2, N, nu) + w
        X_data_array_diff_ic[i, j] = x_2
        Y_data_array_diff_ic[i, j] = H @ x_2 + np.random.multivariate_normal(np.zeros(dim_y), R)
        
np.savez('ks_data_diff_ic.npz', X_data=X_data_array_diff_ic, Y_data=Y_data_array_diff_ic)
