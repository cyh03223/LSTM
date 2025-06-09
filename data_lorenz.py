#data generation for lorenz system
import numpy as np
import scipy.integrate
import random

random.seed(32)  # Python random seed
np.random.seed(32)  # NumPy random seed

print("Generating data")
# Simulation variables
seq_length = 1001 # Number of time steps in each sequence
num_records = 50 # Number of sequences to generate
dim_y = 1  # No. of observables
dim_x= 3 # Number of states
dt=0.1
end_t=seq_length*dt

# Arrays to store results
X_data_array = np.empty((num_records, seq_length, dim_x))
Y_data_array = np.empty((num_records, seq_length, dim_y))

# Continuous-time model
def f(t,x,s,r,b):
    x1,x2,x3=x
    dx_dt = [0,0,0]
    dx_dt[0] = -(s*x1)+(s*x2)
    dx_dt[1] = (r*x1)-x2-(x1*x3) 
    dx_dt[2] = (x1*x2)-(b*x3)
    return dx_dt

# Parameters
s=10
r=28
b=8/3
 
P=0.0001*np.identity(dim_x) #covariance of initial condition
mu_pn = np.zeros(dim_x) #mean of process noise
Q=0.01*np.identity(dim_x) #process noise covariance
mu_mn=np.zeros(dim_y) #mean of measurement noise
R=[0.01] #measurement noise covariance
G=np.identity(dim_x) #process noise coefficient matrix
H=np.matrix(([[1],[1],[0]]))
t_span= np.arange(0,end_t,dt)
num_timesteps=np.shape(t_span)[0]

for i in range(0, num_records, 1):
    mu_x0=np.random.uniform(-2,2,size=dim_x) #mean of initial condition
    print(mu_x0)
    x0_old=np.random.multivariate_normal(mu_x0,P)
    sol=scipy.integrate.solve_ivp(f,(0,end_t),x0_old, method='RK45', args=(s,r,b),dense_output=True, t_eval=np.arange(0, end_t, dt))
    X_data_array[i,:,:]=np.transpose(sol.y)
    for j in range(1,seq_length,1):
        w=np.random.multivariate_normal(mu_pn,Q) 
        X_data_array[i,j,:]=X_data_array[i,j,:]+np.matmul(G,w)
        v=np.random.normal(mu_mn,R)   #change to multivariate if R has greater than 1 dimension
        Y_data_array[i,j,:]=np.matmul(X_data_array[i,j,:],H)+v
        
np.savez("lorenz_data.npz", X_data=X_data_array, Y_data=Y_data_array)

# Arrays to store results
X_data_array_diff_ic = np.empty((num_records, seq_length, dim_x))
Y_data_array_diff_ic = np.empty((num_records, seq_length, dim_y))
#random initial condition
for i in range(0, num_records, 1):
    mu_x0=np.random.uniform(-10,10,size=dim_x) #mean of initial condition
    x0_old=np.random.multivariate_normal(mu_x0,P)
    sol=scipy.integrate.solve_ivp(f,(0,end_t),x0_old, method='RK45', args=(s,r,b),dense_output=True, t_eval=np.arange(0, end_t, dt))
    X_data_array_diff_ic[i,:,:]=np.transpose(sol.y)
    for j in range(1,seq_length,1):
        w=np.random.multivariate_normal(mu_pn,Q) 
        X_data_array_diff_ic[i,j,:]=X_data_array_diff_ic[i,j,:]+np.matmul(G,w)
        v=np.random.normal(mu_mn,R)   #change to multivariate if R has greater than 1 dimension
        Y_data_array_diff_ic[i,j,:]=np.matmul(X_data_array_diff_ic[i,j,:],H)+v
        
np.savez("lorenz_data_diff_ic.npz", X_data=X_data_array_diff_ic, Y_data=Y_data_array_diff_ic)
