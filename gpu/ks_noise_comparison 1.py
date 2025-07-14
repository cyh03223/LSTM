#KS all comparisons
import os
import math
import time
import random
import argparse
import numpy as np
import scipy.integrate
import tensorflow as tf
from numpy import linalg
from tensorflow import keras
from scipy.stats import norm
from scipy.special import logsumexp
from keras.callbacks import EarlyStopping
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

random.seed(6)  # Python random seed
np.random.seed(6)  # NumPy random seed
tf.random.set_seed(6)  # TensorFlow random seed

# Simulation variables
seq_length = 3001 # Number of time steps in each sequence
num_records = 50 # Number of sequences to generate
N=20 #no. of nodes for Galerkin approximation
dim_y = N # Degrees of freedom (number of connected masses) # No. of measured states
dim_x= (2*dim_y)+1 # Number of states
L=2*np.pi
c=-L
d=L
dt=1e-6  #Sampling time (seconds)
end_t=seq_length*dt
nu=0.01

# Arrays to store results
X_data_array = np.empty((num_records, seq_length, dim_x))
Y_data_array = np.empty((num_records, seq_length, dim_y))

# Noise parameters
c_p = 0.0001
c_q = 0.0001
c_r = 100
P = c_p * np.eye(dim_x)
Q = c_q * np.eye(dim_x)
R = c_r * np.eye(dim_y)
alpha_P = 1/c_p
alpha_Q = 1/c_q
alpha_R = 1/c_r
max_c = max(alpha_P, alpha_Q, alpha_R)
alpha_P /= max_c
alpha_Q /= max_c
alpha_R /= max_c
print("alpha P:", alpha_P)
print("alpha Q:", alpha_Q)
print("alpha R:", alpha_R)

data_H=np.load("matrix_h.npz")
H=data_H["H"]

data_initial_state_mean=np.load("x0_mean.npz")
initial_state_mean=data_initial_state_mean["x0_mean"]

t_span= np.arange(0,end_t,dt)
num_timesteps=np.shape(t_span)[0]

#data for training, validation and testing
data = np.load(f"ks_data_cr_{str(c_r).replace('.', '_')}.npz")
X_data_array = data["X_data"]
Y_data_array = data["Y_data"]

#data for random IC testing 
data2=np.load(f"ks_data_diff_ic_cr_{str(c_r).replace('.', '_')}.npz")
X_test_diff_ic_i = data2["X_data"]
Y_test_diff_ic_i = data2["Y_data"]

#split data into training, validation and testing data
num_train_pts=np.floor(0.80*num_records).astype(int)
num_val_pts=np.floor(0.10*num_records).astype(int)
test_data_size=num_records-num_train_pts-num_val_pts
num_train_plus_val=num_train_pts+num_val_pts

Y_train=Y_data_array[0:num_train_pts,:,:]
X_train=X_data_array[0:num_train_pts,:,:]

#z-score normalisation with mean 0 and std 1 along all timesteps and samples for each feature
Y_train_reshaped=Y_train.reshape(-1,Y_train.shape[2])
X_train_reshaped=X_train.reshape(-1,X_train.shape[2])

mean_y=Y_train_reshaped.mean(axis=0)
std_y=Y_train_reshaped.std(axis=0)
mean_x=X_train_reshaped.mean(axis=0)
std_x=X_train_reshaped.std(axis=0)
print("mean of measurements is",mean_y)
print("mean of states is",mean_x)
print("standard deviation of measurements is",std_y)
print("standard deviation of states is",std_x)

Y_train_norm=(Y_train-mean_y)/std_y
X_train_norm=(X_train-mean_x)/std_x

Y_validate=Y_data_array[num_train_pts:num_train_plus_val,:,:]
X_validate=X_data_array[num_train_pts:num_train_plus_val,:,:]
Y_validate_norm=(Y_validate-mean_y)/std_y
X_validate_norm=(X_validate-mean_x)/std_x

Y_test=Y_data_array[num_train_plus_val:num_records,:,:]
X_test=X_data_array[num_train_plus_val:num_records,:,:]
Y_test_norm=(Y_test-mean_y)/std_y
X_test_norm=(X_test-mean_x)/std_x

Y_test_diff_ic=Y_test_diff_ic_i[:,:,:]
X_test_diff_ic=X_test_diff_ic_i[:,:,:]
Y_test_diff_ic_norm=(Y_test_diff_ic-mean_y)/std_y
X_test_diff_ic_norm=(X_test_diff_ic-mean_x)/std_x

batch_size=5

# Convert data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((Y_train_norm, X_train_norm)).batch(batch_size)

val_input_data = tf.convert_to_tensor(Y_validate_norm, dtype=tf.float32)
val_output_data = tf.convert_to_tensor(X_validate_norm, dtype=tf.float32)

test_input_data = tf.convert_to_tensor(Y_test_norm, dtype=tf.float32)
test_output_data = tf.convert_to_tensor(X_test_norm, dtype=tf.float32)

test_input_data_diff_ic=tf.convert_to_tensor(Y_test_diff_ic_norm, dtype=tf.float32)
test_output_data_diff_ic=tf.convert_to_tensor(X_test_diff_ic_norm, dtype=tf.float32)

H_tf = tf.convert_to_tensor(H, dtype=tf.float32)

def h(x):
    return tf.matmul(x, tf.transpose(H_tf))


# Define the Jordan LSTM model 
class JordanLSTM(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(JordanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wxh = self.add_weight(shape=(input_size, hidden_size * 4),
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   trainable=True)
        self.Wyh = self.add_weight(shape=(output_size, hidden_size * 4),
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   trainable=True)
        self.bh = self.add_weight(shape=(hidden_size * 4,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.by = self.add_weight(shape=(output_size,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.Why = self.add_weight(shape=(hidden_size, output_size),
                                   initializer=tf.keras.initializers.Orthogonal(),
                                   trainable=True)

    def call(self, x):
        batch_size, seq_len, _ = x.shape
        y = tf.zeros((batch_size, self.output_size))
        c = tf.zeros((batch_size, self.hidden_size))
        outputs = []
        for i in range(seq_len):
            gates = tf.matmul(x[:, i, :], self.Wxh) + tf.matmul(y, self.Wyh) + self.bh
            ingate, forgetgate, cellgate, outgate = tf.split(gates, 4, axis=1)
            ingate = tf.sigmoid(ingate)
            forgetgate = tf.sigmoid(forgetgate)
            cellgate = tf.tanh(cellgate)
            outgate = tf.sigmoid(outgate)
            c = forgetgate * c + ingate * cellgate
            h = outgate * tf.tanh(c)
            y = tf.matmul(h, self.Why) + self.by
            outputs.append(tf.expand_dims(y, 1))
        return tf.concat(outputs, axis=1)

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt, H, N, nu):
        self.state_estimate = initial_state
        self.P = initial_covariance
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        self.dt = dt
        self.H = H
        self.N = N
        self.nu = nu

    def ks_galerkin_rhs(self,x):
        
        #Compute RHS of 21D Galerkin approximation of the KS equation.
        #x: state vector [a0, a1, ..., aN, b1, ..., bN] of length 2N + 1
        #Returns dx/dt vector of same shape
        # N = self.N
        # nu= self.nu
        
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
                    if k + m == n:
                        sum_cos += 0.5 * (a[k-1]*a[m-1] - b[k-1]*b[m-1])
                        sum_sin += a[k-1]*b[m-1]
                    if k - m == n:
                        sum_cos += 0.5 * (a[k-1]*a[m-1] + b[k-1]*b[m-1])
                        sum_sin += a[k-1]*b[m-1]
                    if m - k == n:
                        sum_cos -= 0.5 * (a[k-1]*a[m-1] + b[k-1]*b[m-1])
                        sum_sin -= a[k-1]*b[m-1]
            dxdt[n] += -n * sum_sin
            dxdt[N + n] += n * sum_cos
    
        return dxdt

    # Euler integrator
    def step_forward(self, x):
        return x + dt * self.ks_galerkin_rhs(x)
    
    def jacobian_f(self, state):
        eps = 1e-6
        n = len(state)
        J = np.zeros((n, n))
        f0 = self.ks_galerkin_rhs(state)
    
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f1 = self.ks_galerkin_rhs(state + dx)
            J[:, i] = (f1 - f0) / eps

        return J
    
    def predict(self):
        Id = np.eye(len(self.state_estimate))

        # Jacobian
        F = Id + self.dt * self.jacobian_f(self.state_estimate)
    
        #Propagate state using Euler
        self.state_estimate = self.step_forward(self.state_estimate)
    
        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q 

    def update(self, measurement):
        z_pred = self.H @ self.state_estimate
        y = measurement - z_pred
        S = self.H @ self.P @ (self.H).T + self.R
        K = self.P @ (self.H).T @ np.linalg.inv(S)
        self.state_estimate = self.state_estimate + K @ y
        self.P = (np.eye(len(self.state_estimate)) - K @ self.H) @ self.P
        return self.state_estimate

def run_on_gpu_1(val_i_data):
    start1=time.time()

   # Initialize variables for early stopping
    best_val_loss = float('inf')  # Set initial best validation loss to infinity
    patience = 5  # Number of epochs to wait for improvement
    counter = 0  # Counter to track the number of epochs with no improvement
    patience_lr = 3 #number of epochs to wait for improvement before reducing learning rate
    counter_lr = 0 # Counter to track the number of epochs with no improvement for learning rate updates
         
    # Define the model, loss function, and optimizer
    hidden_size = 400
    model = JordanLSTM(dim_y, hidden_size, dim_x)
    criterion = tf.keras.losses.MeanSquaredError()

    initial_lr = 1e-3  # Initial learning rate
    min_lr = 1e-4  # Minimum learning rate
    lr_factor = 0.5  # Factor by which learning rate will be reduced
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    ckpt = tf.train.Checkpoint(model=model)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, f'./ks_checkpoints_l_jlstm_cr_{str(c_r).replace(".", "_")}', max_to_keep=1)
    
    # Training the model
    num_epochs = 100
    for epoch in range(num_epochs):
        # Iterate over batches
        for batch in train_dataset:
            input_data, output_data = batch
            # Forward pass
            with tf.GradientTape() as tape:
                outputs = model(input_data)
                loss = 0.5 * (alpha_P * criterion(output_data[:, 0, :], outputs[:, 0, :]) + alpha_Q * criterion(output_data[:, 1:,:], outputs[:, 1:,:]) + alpha_R * criterion(input_data[:, 1:, :], h(outputs[:,1:,:]))) 
                                
            # Backward and optimize
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation
        val_i_data = tf.stop_gradient(val_i_data)
        val_outputs = model(val_i_data)
        val_loss = 0.5 * (alpha_P * criterion(val_output_data[:, 0, :], val_outputs[:, 0, :]) + alpha_Q * criterion(val_output_data[:, 1:,:], val_outputs[:, 1:,:]) + alpha_R * criterion(val_i_data[:, 1:, :], h(val_outputs[:,1:,:]))) 
    
        # Print and check for early stopping
        print(f'Epoch [{epoch}], Loss: {loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  
            ckpt_manager.save()
        else:
            counter += 1
            counter_lr+=1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.')
            break
            
        # Reduce learning rate only once after the patience threshold is reached
        if counter_lr == patience_lr:
            new_lr = max(optimizer.learning_rate.numpy() * lr_factor, min_lr)
            optimizer.learning_rate.assign(new_lr)
            print(f'Reducing learning rate to: {new_lr}')
            counter_lr = 0  # Reset the counter after reducing the learning rate
        
    end1=time.time()
    print("time taken to train JLSTM:",end1-start1)

    #Restore best model
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Loaded best model weights for testing.")
    
    # Test the model
    start2 = time.time()
    test_input_data_1=test_input_data
    test_input_data_1=tf.stop_gradient(test_input_data_1)
    predicted_output_jlstm=model(test_input_data_1)
    test_loss = criterion(predicted_output_jlstm[:,1:,:], test_output_data[:,1:,:])
    print("Test Loss:", test_loss.numpy())
    end2 = time.time()
    print("Time taken to test JLSTM:", end2 - start2)

    #print(np.shape(predicted_output_jlstm))
    predicted_output_jlstm=predicted_output_jlstm.numpy()

    # Test the model on different IC range
    start3 = time.time()
    test_input_data_diff_ic_1=test_input_data_diff_ic
    test_input_data_diff_ic_1=tf.stop_gradient(test_input_data_diff_ic_1)
    predicted_output_jlstm_diff_ic=model(test_input_data_diff_ic_1)
    test_loss_diff_ic_1 = criterion(predicted_output_jlstm_diff_ic[:,1:,:], test_output_data_diff_ic[:,1:,:])
    print("Test Loss for different i.c. range:", test_loss_diff_ic_1.numpy())
    end3 = time.time()
    print("Time taken to test JLSTM for different i.c. range:", end3 - start3)
    
    predicted_output_jlstm_diff_ic=predicted_output_jlstm_diff_ic.numpy()
    
    return predicted_output_jlstm,predicted_output_jlstm_diff_ic

def run_on_gpu_2(val_i_data):
    start4=time.time()
   # Initialize variables for early stopping
    best_val_loss2 = float('inf')  # Set initial best validation loss to infinity
    patience2 = 5  # Number of epochs to wait for improvement
    counter2 = 0  # Counter to track the number of epochs with no improvement
    patience2_lr = 3 #number of epochs to wait for improvement before reducing learning rate
    counter2_lr = 0 # Counter to track the number of epochs with no improvement for learning rate updates
         
    # Define the model, loss function, and optimizer
    hidden_size2 = 400
    model2 = JordanLSTM(dim_y, hidden_size2, dim_x)
    criterion2 = tf.keras.losses.MeanSquaredError()

    initial_lr = 1e-3  # Initial learning rate
    min_lr = 1e-5  # Minimum learning rate
    lr_factor = 0.5  # Factor by which learning rate will be reduced
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    ckpt2 = tf.train.Checkpoint(model=model2)
    ckpt_manager2 = tf.train.CheckpointManager(ckpt2, f'./ks_checkpoints_jlstm_cr_{str(c_r).replace(".", "_")}', max_to_keep=1)
    
    # Training the model
    num_epochs2 = 100
    for epoch in range(num_epochs2):
        # Iterate over batches
        for batch in train_dataset:
            input_data, output_data = batch
            # Forward pass
            with tf.GradientTape() as tape:
                outputs2 = model2(input_data[:,1:,:])
                loss2 = criterion2(output_data[:,1:,:], outputs2)
                
            # Backward and optimize
            gradients2 = tape.gradient(loss2, model2.trainable_variables)
            optimizer2.apply_gradients(zip(gradients2, model2.trainable_variables))

        # Validation
        val_i_data=tf.stop_gradient(val_i_data)
        val_loss2 = criterion2(model2(val_i_data[:,1:,:]), val_output_data[:,1:,:])
    
        # Print and check for early stopping
        print(f'Epoch [{epoch}], Loss: {loss2.numpy():.4f}, Val Loss: {val_loss2.numpy():.4f}')
            
        if val_loss2 < best_val_loss2:
            best_val_loss2 = val_loss2
            counter2 = 0  
            ckpt_manager2.save()
        else:
            counter2 += 1
            counter2_lr+=1
        if counter2 >= patience2:
            print(f'Early stopping at epoch {epoch+1} as validation loss did not improve for {patience2} epochs.')
            break
            
        # Reduce learning rate only once after the patience threshold is reached
        if counter2_lr == patience2_lr:
            new_lr = max(optimizer2.learning_rate.numpy() * lr_factor, min_lr)
            optimizer2.learning_rate.assign(new_lr)
            print(f'Reducing learning rate to: {new_lr}')
            counter2_lr = 0  # Reset the counter after reducing the learning rate
        
    end4=time.time()
    print("time taken to train JLSTM:",end4-start4)

    # Restore best model
    ckpt2.restore(ckpt_manager2.latest_checkpoint)
    
    # Test the model
    start5 = time.time()
    test_input_data_2=test_input_data
    test_input_data_2=tf.stop_gradient(test_input_data_2)
    predicted_output_jlstm=model2(test_input_data_2[:,1:,:])
    test_loss2 = criterion2(predicted_output_jlstm, test_output_data[:,1:,:])
    print("Test Loss:", test_loss2.numpy())
    end5 = time.time()
    print("Time taken to test JLSTM:", end5 - start5)

    #print(np.shape(predicted_output_jlstm))
    predicted_output_jlstm=predicted_output_jlstm.numpy()

    # Test the model on different IC range
    start6 = time.time()
    test_input_data_diff_ic_2=test_input_data_diff_ic
    test_input_data_diff_ic_2=tf.stop_gradient(test_input_data_diff_ic_2)
    predicted_output_jlstm_diff_ic=model2(test_input_data_diff_ic_2[:,1:,:])
    test_loss_diff_ic_2 = criterion2(predicted_output_jlstm_diff_ic, test_output_data_diff_ic[:,1:,:])
    print("Test Loss for different i.c. range:", test_loss_diff_ic_2.numpy())
    end6 = time.time()
    print("Time taken to test JLSTM for different i.c. range:", end6 - start6)
     
    predicted_output_jlstm_diff_ic=predicted_output_jlstm_diff_ic.numpy()
    
    print("Test loss using mean", np.mean((predicted_output_jlstm[:,1:,:]-test_output_data[:,1:,:])**2))
    print("Test loss using mean for different i.c.", np.mean((predicted_output_jlstm_diff_ic[:,1:,:]-test_output_data_diff_ic[:,1:,:])**2))
    
    return predicted_output_jlstm,predicted_output_jlstm_diff_ic

def run_on_gpu_3():
    #ekf implementation
    start_time = time.time()
    all_estimates = []
    input_data_ekf = np.concatenate((Y_test, Y_test_diff_ic), axis=0)
    output_data_ekf = np.concatenate((X_test, X_test_diff_ic), axis=0)

    for seq_idx in range(len(input_data_ekf)):
        print("Running EKF on sequence", seq_idx)
        true_states = output_data_ekf[seq_idx]
        observations = input_data_ekf[seq_idx]
        ekf = ExtendedKalmanFilter(initial_state=initial_state_mean,
                                   initial_covariance=P.copy(),
                                   process_noise_cov=Q.copy(),
                                   measurement_noise_cov=R.copy(),
                                   dt=dt, H=H, N=N, nu=nu)
        estimated_states = []

        for t in range(1, len(observations)):
            ekf.predict()
            est = ekf.update(observations[t])
            estimated_states.append(est)

        all_estimates.append(np.array(estimated_states))

    all_estimates = np.array(all_estimates)
    end_time = time.time()
    print("Time taken to run EKF on test data:", end_time - start_time)

    all_estimates_split = np.split(all_estimates, [test_data_size], axis=0)
    # print(all_estimates_split)
    # all_estimates_split[0]=all_estimates_split[0].reshape(np.shape(X_test[:, 1:, :]))
    # all_estimates_split[1]=all_estimates_split[1].reshape(np.shape(X_test_diff_ic[:, 1:, :]))
    
    mse = np.mean((all_estimates_split[0] - X_test[:, 1:, :])**2)
    print("MSE on test data:", mse)
    
    mse_diff_ic = np.mean((all_estimates_split[1] - X_test_diff_ic[:, 1:, :])**2)
    print("MSE on test data with different ICs:", mse_diff_ic)

    return all_estimates_split
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, required=True, help="Model ID (1,2 or 3)")
    args = parser.parse_args()

    if args.model_id == 1:
        predicted_output_jlstm, predicted_output_jlstm_diff_ic=run_on_gpu_1(val_input_data)
        predicted_output_jlstm=(predicted_output_jlstm*std_x)+mean_x
        predicted_output_jlstm_diff_ic=(predicted_output_jlstm_diff_ic*std_x)+mean_x
        np.savez(f"ks_result_cr_{str(c_r).replace('.', '_')}_l_jlstm.npz", predicted_jlstm_data=predicted_output_jlstm, predicted_jlstm_data_diff_ic=predicted_output_jlstm_diff_ic)
        print("Test error (denorm.):", np.mean((predicted_output_jlstm-X_test[:,1:,:])**2))
        print("Test error diff. ic. (denorm.):", np.mean((predicted_output_jlstm_diff_ic-X_test_diff_ic[:,1:,:])**2))
    elif args.model_id == 2:
        predicted_output_jlstm, predicted_output_jlstm_diff_ic=run_on_gpu_2(val_input_data)
        predicted_output_jlstm=(predicted_output_jlstm*std_x)+mean_x
        predicted_output_jlstm_diff_ic=(predicted_output_jlstm_diff_ic*std_x)+mean_x
        np.savez(f"ks_result_cr_{str(c_r).replace('.', '_')}_jlstm.npz", predicted_jlstm_data=predicted_output_jlstm, predicted_jlstm_data_diff_ic=predicted_output_jlstm_diff_ic)
        print("Test error (denorm.):", np.mean((predicted_output_jlstm-X_test[:,1:,:])**2))
        print("Test error diff. ic. (denorm.):", np.mean((predicted_output_jlstm_diff_ic-X_test_diff_ic[:,1:,:])**2))
    elif args.model_id == 3:
        predicted_output_ekf = run_on_gpu_3()
        np.savez(f"ks_result_cr_{str(c_r).replace('.', '_')}_ekf.npz", predicted_ekf_data=predicted_output_ekf[0], predicted_ekf_data_diff_ic=predicted_output_ekf[1])
    else:
        print("Unknown model_id.")
    
   
    