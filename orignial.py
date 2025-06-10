#Lorenz all comparisons
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
from numpy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.signal import cont2discrete, lti, dlti, dstep
import scipy.integrate
from scipy.stats import norm
import time
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from torch.utils.data import DataLoader, TensorDataset 
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random

# random.seed(36)  # Python random seed
# np.random.seed(36)  # NumPy random seed
# tf.random.set_seed(36)  # TensorFlow random seed

#np.seterr(all='warn')

#print("Generating data")
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
 
P=0.0001*np.identity(dim_x) #covariance of initial condition
mu_pn = np.zeros(dim_x) #mean of process noise
Q=0.01*np.identity(dim_x) #process noise covariance
mu_mn=np.zeros(dim_y) #mean of measurement noise
R=[0.01] #measurement noise covariance
G=np.identity(dim_x) #process noise coefficient matrix

t_span= np.arange(0,end_t,dt)
num_timesteps=np.shape(t_span)[0]

#data for training, validation and testing
#IC is chosen randomly from [-2,2]^3
data = np.load("lorenz_data.npz")
X_data_array = data["X_data"]
Y_data_array = data["Y_data"]

#data for random IC testing 
#IC chosen randomly between [-10,10]^3
data2=np.load("lorenz_data_diff_ic.npz")
X_test_diff_ic = data2["X_data"]
Y_test_diff_ic = data2["Y_data"]

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

# Define the Elman LSTM model
class ElmanLSTM(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
               
        self.Wxh = self.add_weight(shape=(input_size, hidden_size * 4),
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   trainable=True)
        self.Whh = self.add_weight(shape=(hidden_size, hidden_size * 4),
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
        h = tf.zeros((batch_size, self.hidden_size))
        c = tf.zeros((batch_size, self.hidden_size))
        outputs = []
        for i in range(seq_len):
            gates = tf.matmul(x[:, i, :], self.Wxh) + tf.matmul(h, self.Whh) + self.bh
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

# Particle Filter Class
class ParticleFilter:
    def __init__(self, num_particles, process_noise_cov, measurement_noise_cov, dt):
        self.num_particles = num_particles
        self.dt = dt  # Time-step
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        
        # Initialize particles randomly around an initial state
        self.particles = np.random.randn(num_particles, 3)
        self.weights = np.ones(num_particles) / num_particles

    def lorenz(self, state):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        return np.array([dx, dy, dz])

    def rk45_step(self, state):
        #Integrates one step using SciPy's RK45 solver
        t_span = [0, self.dt]  # Solve from t=0 to t=dt
        sol = scipy.integrate.solve_ivp(lambda t, y: self.lorenz(y), t_span, state, method='RK45', max_step=self.dt)
        return sol.y[:, -1]  # Return the final state after dt

    def predict(self):
        for i in range(self.num_particles):
            self.particles[i] = self.rk45_step(self.particles[i])  # Use RK45 integration
            noise = np.random.multivariate_normal([0, 0, 0], self.process_noise_cov)
            self.particles[i] += noise  # Add process noise

    def update(self, observation):
        # Compute the likelihood of the observation given each particle
        predicted_obs = self.particles[:, 0] + self.particles[:, 1]  # Using the first two states for measurement
        residuals = predicted_obs-observation
        
        # Compute importance weights using the normal distribution
        self.weights = norm.pdf(residuals, scale=np.sqrt(self.measurement_noise_cov))
        weight_sum=np.sum(self.weights)
        if weight_sum == 0 or not np.isfinite(weight_sum):
            print("Warning: Weight sum zero or invalid. Resetting weights.")
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= weight_sum  # Normalize weights so that they can take the form of probability
        
    def resample(self):
        # Systematic resampling
        bins = np.cumsum(self.weights)  # Compute cumulative sum of weights
        indices = np.zeros(self.num_particles, dtype=int)  # Store resampled indices
    
        # Step size and initial random pick
        step = 1.0 / self.num_particles
        random_pick = np.random.uniform(0, step)
    
        # Find first index
        index = np.searchsorted(bins, random_pick)
        index = min(index, self.num_particles - 1)
        indices[0] = index
    
        # Iterate to find remaining indices
        for i in range(1, self.num_particles):
            random_pick += step  # Move systematically
            index = np.searchsorted(bins, random_pick)
            index = min(index, self.num_particles - 1)
            indices[i] = index  # Store resampled index
    
        # Resample particles
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights to uniform
    
    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)
    
    def step(self, measurement):
        self.predict()
        self.update(measurement)
        self.resample()
        return self.estimate()

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, dt):
        self.state_estimate = initial_state
        self.P = initial_covariance
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        self.dt = dt

    def lorenz_dynamics(self, state):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    def jacobian_f(self, state):
        x, y, z = state
        return np.array([
            [-10,     10,     0],
            [28 - z,  -1,   -x],
            [y,       x,  -8/3]
        ])

    def jacobian_h(self, state):
        # h(x) = x + y, so Jacobian is [1, 1, 0]
        return np.array([[1, 1, 0]])

    def predict(self):
     
        Id=np.eye(len(self.state_estimate))
        t_span = [0, self.dt]  # Solve from t=0 to t=dt
        solution=scipy.integrate.solve_ivp(lambda t, y: self.lorenz_dynamics(y), t_span, self.state_estimate, method='RK45', max_step=self.dt)
        self.state_estimate=solution.y[:,-1]

        F=Id+(1/6)*((self.dt*self.jacobian_f(self.state_estimate))+(2*self.dt*self.jacobian_f(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate))*(Id+(self.dt/2)*self.jacobian_f(self.state_estimate)))+(2*self.dt*self.jacobian_f(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate)))*(Id+(self.dt/2)*self.jacobian_f(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate)))*(Id+(self.dt/2)*self.jacobian_f(self.state_estimate)))+(self.dt*self.jacobian_f(self.state_estimate+self.dt*self.lorenz_dynamics(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate))))*(Id+self.dt*self.jacobian_f(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate))))*(Id+(self.dt/2)*self.jacobian_f(self.state_estimate+(self.dt/2)*self.lorenz_dynamics(self.state_estimate)))*(Id+(self.dt/2)*self.jacobian_f(self.state_estimate)))) 
        
        self.P = F @ self.P @ F.T + self.Q   
        
    def update(self, measurement):
        H = self.jacobian_h(self.state_estimate)
      #  z_pred = self.state_estimate[0] + self.state_estimate[1]
        z_pred = H @ self.state_estimate
        y = measurement - z_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state_estimate = self.state_estimate + K.flatten() * y
        self.P = (np.eye(len(self.state_estimate)) - K @ H) @ self.P
        return self.state_estimate
        
def run_on_gpu_1(val_i_data):
   # Initialize variables for early stopping
    best_val_loss = float('inf')  # Set initial best validation loss to infinity
    patience = 5  # Number of epochs to wait for improvement
    counter = 0  # Counter to track the number of epochs with no improvement
    patience_lr = 3 #number of epochs to wait for improvement before reducing learning rate
    counter_lr = 0 # Counter to track the number of epochs with no improvement for learning rate updates
                    
    # Define the model, loss function, and optimizer
    hidden_size = 50
    model = ElmanLSTM(dim_y, hidden_size, dim_x)
    criterion = tf.keras.losses.MeanSquaredError()
    
    initial_lr = 1e-1  # Initial learning rate
    min_lr = 1e-2  # Minimum learning rate
    lr_factor = 0.5  # Factor by which learning rate will be reduced
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    start=time.time()
    # Training the model
    num_epochs = 15000
    for epoch in range(num_epochs):
        # Iterate over batches
        for batch in train_dataset:
            input_data, output_data = batch
            # Forward pass
            with tf.GradientTape() as tape:
                outputs = model(input_data)
                loss = criterion(output_data, outputs)

            # Backward and optimize
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation
        val_i_data=tf.stop_gradient(val_i_data)
        val_loss = criterion(model(val_i_data), val_output_data)
    
        # Print and check for early stopping
        print(f'Epoch [{epoch}], Loss: {loss.numpy():.4f}, Val Loss: {val_loss.numpy():.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
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
        
    end=time.time()
    print("time taken to train ELSTM:",end-start)
    
    # Test the model on same IC range
    start2 = time.time()
    test_input_data_1=test_input_data
    test_input_data_1=tf.stop_gradient(test_input_data_1)
    predicted_output_elstm=model(test_input_data_1)
    test_loss = criterion(predicted_output_elstm, test_output_data)
    print("Test Loss:", test_loss.numpy())
    end2 = time.time()
    print("Time taken to test ELSTM:", end2 - start2)

    predicted_output_elstm=predicted_output_elstm.numpy()

    # Test the model on different IC range
    start6 = time.time()
    test_input_data_diff_ic_1=test_input_data_diff_ic
    test_input_data_diff_ic_1=tf.stop_gradient(test_input_data_diff_ic_1)
    predicted_output_elstm_diff_ic=model(test_input_data_diff_ic_1)
    test_loss_diff_ic = criterion(predicted_output_elstm_diff_ic, test_output_data_diff_ic)
    print("Test Loss for different i.c. range:", test_loss_diff_ic.numpy())
    end6 = time.time()
    print("Time taken to test ELSTM for different i.c. range:", end6 - start6)
    
    predicted_output_elstm_diff_ic=predicted_output_elstm_diff_ic.numpy()
    
    
    #appendix-ELSTM weights and test data plots 
    # Output the final weight and bias values
    print("Final Weight (Wxh):")
    print(model.Wxh)
    print("Final Weight (Whh):")
    print(model.Whh)
    print("Final Weight (Why):")
    print(model.Why)
    print("Final bias (bh):")
    print(model.bh)
    print("Final Bias (by):")
    print(model.by)
    
    return predicted_output_elstm, predicted_output_elstm_diff_ic

def run_on_gpu_2(val_i_data):
   # Initialize variables for early stopping
    best_val_loss2 = float('inf')  # Set initial best validation loss to infinity
    patience2 = 5  # Number of epochs to wait for improvement
    counter2 = 0  # Counter to track the number of epochs with no improvement
    patience2_lr = 3 #number of epochs to wait for improvement before reducing learning rate
    counter2_lr = 0 # Counter to track the number of epochs with no improvement for learning rate updates
         
    # Define the model, loss function, and optimizer
    hidden_size2 = 50
    model2 = JordanLSTM(dim_y, hidden_size2, dim_x)
    criterion2 = tf.keras.losses.MeanSquaredError()

    initial_lr = 1e-1  # Initial learning rate
    min_lr = 1e-2  # Minimum learning rate
    lr_factor = 0.5  # Factor by which learning rate will be reduced
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    start3=time.time()
    # Training the model
    num_epochs2 = 15000
    for epoch in range(num_epochs2):
        # Iterate over batches
        for batch in train_dataset:
            input_data, output_data = batch
            # Forward pass
            with tf.GradientTape() as tape:
                outputs2 = model2(input_data)
                loss2 = criterion2(output_data, outputs2)
                
            # Backward and optimize
            gradients2 = tape.gradient(loss2, model2.trainable_variables)
            optimizer2.apply_gradients(zip(gradients2, model2.trainable_variables))

        # Validation
        val_i_data=tf.stop_gradient(val_i_data)
        val_loss2 = criterion2(model2(val_i_data), val_output_data)
    
        # Print and check for early stopping
        print(f'Epoch [{epoch}], Loss: {loss2.numpy():.4f}, Val Loss: {val_loss2.numpy():.4f}')
            
        if val_loss2 < best_val_loss2:
            best_val_loss2 = val_loss2
            counter2 = 0                
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
        
    end3=time.time()
    print("time taken to train JLSTM:",end3-start3)
        
    # Test the model
    start4 = time.time()
    test_input_data_2=test_input_data
    test_input_data_2=tf.stop_gradient(test_input_data_2)
    predicted_output_jlstm=model2(test_input_data_2)
    test_loss2 = criterion2(predicted_output_jlstm, test_output_data)
    print("Test Loss:", test_loss2.numpy())
    end4 = time.time()
    print("Time taken to test JLSTM:", end4 - start4)

    #print(np.shape(predicted_output_jlstm))
    predicted_output_jlstm=predicted_output_jlstm.numpy()

    # Test the model on different IC range
    start7 = time.time()
    test_input_data_diff_ic_2=test_input_data_diff_ic
    test_input_data_diff_ic_2=tf.stop_gradient(test_input_data_diff_ic_2)
    predicted_output_jlstm_diff_ic=model2(test_input_data_diff_ic_2)
    test_loss_diff_ic_2 = criterion2(predicted_output_jlstm_diff_ic, test_output_data_diff_ic)
    print("Test Loss for different i.c. range:", test_loss_diff_ic_2.numpy())
    end7 = time.time()
    print("Time taken to test JLSTM for different i.c. range:", end7 - start7)
    
    predicted_output_jlstm_diff_ic=predicted_output_jlstm_diff_ic.numpy()
    
    #appendix-JLSTM weights and test data plots
    # Output the final weight and bias values
    print("Final Weight (Wxh):")
    print(model2.Wxh)
    print("Final Weight (Whh):")
    print(model2.Wyh)
    print("Final Weight (Why):")
    print(model2.Why)
    print("Final bias (bh):")
    print(model2.bh)
    print("Final Bias (by):")
    print(model2.by)
    
    return predicted_output_jlstm,predicted_output_jlstm_diff_ic

def run_on_gpu_3():
    #paricle filter implementation
    start5=time.time()
    num_particles = 1000  # Number of particles for the filter
    
    # Prepare an array to store the estimated states for all sequences
    all_estimates = []

    input_data_pf=np.concatenate((Y_test,Y_test_diff_ic), axis=0)
    output_data_pf=np.concatenate((X_test,X_test_diff_ic), axis=0)
    
    print("no. of input sequences",len(input_data_pf))
    # Run the particle filter over all sequences
    for seq_idx in range(len(input_data_pf)):
        # Initialize the particle filter
        pf = ParticleFilter(num_particles, Q, R, dt)
        true_states = output_data_pf[seq_idx]  # True states for the current sequence
        observations = input_data_pf[seq_idx]  # Corresponding noisy measurements for the current sequence
    
        estimated_states = []
        print("Running sequence ", seq_idx)
        #print("no. of timesteps", len(observations))
        # Run the particle filter on the current sequence
        for t in range(1, len(observations)):
            observation = observations[t]
            # Get the current estimate of the state (weighted average of particles)
            estimated_state = pf.step(observation)
            estimated_states.append(estimated_state)
    
        # Store the estimated states for the current sequence
        all_estimates.append(np.array(estimated_states))
    
    # Convert all_estimates list to a numpy array (size: num_sequences x time_steps x state_dim)
    all_estimates = np.array(all_estimates)
    end5=time.time()
    print("Time taken to run particle filter on test data:", end5-start5)

    print("the shape of the estimates array is", all_estimates.shape)
    all_estimates_split=np.split(all_estimates,[test_data_size],axis=0)
    
    print("the shape of first part of split estimates array is",all_estimates_split[0].shape)
    print("the shape of second part of split estimates arraty is",all_estimates_split[1].shape)

    # Compute the Mean Squared Error (MSE) for all sequences
    all_mse = np.mean((all_estimates_split[0] - X_test[:, 1:, :])**2)
    print(f'Mean Squared Error for all sequences: {all_mse}')
    
     # Compute the Mean Squared Error (MSE) for all sequences
    all_mse_diff_ic = np.mean((all_estimates_split[1] - X_test_diff_ic[:, 1:, :])**2)
    print(f'Mean Squared Error for all sequences with i.c. sampled from interval larger than training range: {all_mse_diff_ic}')
    
    return all_estimates_split

def run_on_gpu_4():
    #ekf implementation
    start_time = time.time()
    all_estimates = []
    input_data_ekf = np.concatenate((Y_test, Y_test_diff_ic), axis=0)
    output_data_ekf = np.concatenate((X_test, X_test_diff_ic), axis=0)

    for seq_idx in range(len(input_data_ekf)):
        print("Running EKF on sequence", seq_idx)
        true_states = output_data_ekf[seq_idx]
        observations = input_data_ekf[seq_idx]
        ekf = ExtendedKalmanFilter(initial_state=np.random.randn(3),
                                   initial_covariance=P.copy(),
                                   process_noise_cov=Q.copy(),
                                   measurement_noise_cov=R[0],
                                   dt=dt)
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
        predicted_output_elstm, predicted_output_elstm_diff_ic=run_on_gpu_1(val_input_data)
        predicted_output_elstm=(predicted_output_elstm*std_x)+mean_x
        predicted_output_elstm_diff_ic=(predicted_output_elstm_diff_ic*std_x)+mean_x
        np.savez("lorenz_data_result_elstm.npz",predicted_elstm_data=predicted_output_elstm, predicted_elstm_data_diff_ic=predicted_output_elstm_diff_ic)
    elif args.model_id == 2:
        predicted_output_jlstm, predicted_output_jlstm_diff_ic=run_on_gpu_2(val_input_data)
        predicted_output_jlstm=(predicted_output_jlstm*std_x)+mean_x
        predicted_output_jlstm_diff_ic=(predicted_output_jlstm_diff_ic*std_x)+mean_x
        np.savez("lorenz_data_result_jlstm.npz",predicted_jlstm_data=predicted_output_jlstm, predicted_jlstm_data_diff_ic=predicted_output_jlstm_diff_ic)
    elif args.model_id == 3:
        predicted_output_pf=run_on_gpu_3()
        np.savez("lorenz_data_result_pf.npz",predicted_pf_data=predicted_output_pf[0], predicted_pf_data_diff_ic=predicted_output_pf[1])
    elif args.model_id == 4:
        predicted_output_ekf = run_on_gpu_4()
        np.savez("lorenz_data_result_ekf.npz", predicted_ekf_data=predicted_output_ekf[0], predicted_ekf_data_diff_ic=predicted_output_ekf[1])
    else:
        print("Unknown model_id.")
    
   
    