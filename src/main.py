# Gated Recurrent Unit (GRU) Recurrent Neural Network (RNN) Implemenation for Global Monthly Temperature Mean Predictor
# Dataset Utilized: Global Temperature Time Series https://datahub.io/core/global-temp?ref=hackernoon.com
# Note: Temperature means are in Celsius (C) format
# Written in Spyder IDE using Python 3.18
# Created by: Leo Martinez III in Summer 2024

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.random import set_seed

#%%----------------------------------------------------------------------------

# Load Dataset

dataset = pd.read_csv("data/flat-ui__data-Tue Jun 04 2024.csv", index_col="Date", parse_dates=["Date"]).drop(["Source"], axis=1)

# Sort the dataset by the DatetimeIndex in ascending order
dataset = dataset.sort_index(ascending=True)

#%%----------------------------------------------------------------------------

# Plot the numbers for 1950-2012 (Training), then plot 2012+ (Testing)
tstart = 1950
tend = 2012

dataset.loc[f"{tstart}":f"{tend}", "Mean"].plot(figsize=(16, 4), legend=True)
dataset.loc[f"{tend+1}":, "Mean"].plot(figsize=(16, 4), legend=True)
plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and beyond)"])
plt.title("Global Temperature Mean in Celsius")
plt.show()

#%%----------------------------------------------------------------------------

# Training/Test Dataset Split
DFtrain = dataset.loc[f"{tstart}":f"{tend}", "Mean"].values
DFtest = dataset.loc[f"{tend+1}":, "Mean"].values

# Preprocess Data
scalar = MinMaxScaler()
DFtrain = DFtrain.reshape(-1, 1)
DFtrain_scaled = scalar.fit_transform(DFtrain)

#%%----------------------------------------------------------------------------

# Define X and y for training/testing

# Define the configuration
n_steps = 250 # take 250 samples into consideration at a time for predicting
features = 1

# Initialize as empty lists
X = []
y = []

def split_sequence(sequence, n_steps): # will be used again for test data
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define X_train and y_train
X_train, y_train = split_sequence(DFtrain_scaled, n_steps)

# Reshaping X_train for model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],features)

#%%----------------------------------------------------------------------------

# Gated Recurrent Unit (GRU) RNN Implementation

# Define input shape for the time series data (n_steps, features)
input_shape = (n_steps, features)

# Input layer
inputs = Input(shape=input_shape)

# LSTM layer (150 neurons)
gru = GRU(units=125, activation='tanh')(inputs)

# Fully connected layer
outputs = Dense(units=1)(gru)

# Define the model
gru_model = Model(inputs=inputs, outputs=outputs)

# Compile the model
gru_model.compile(optimizer='RMSprop', loss='mse')

# Summary of the model (Optional)
gru_model.summary()

gru_model.fit(X_train, y_train, epochs=50, batch_size=32)

#%%----------------------------------------------------------------------------

# Overall GRU Performance Test

# Only take feature "Mean" into consideration
dataset_total = dataset.loc[:,"Mean"]
test_inputs = dataset_total[len(dataset_total) - len(DFtest) - n_steps :].values
test_inputs = test_inputs.reshape(-1, 1)

# Apply MinMax Normalization
test_inputs = scalar.transform(test_inputs)

# Split into samples
X_test, y_test = split_sequence(test_inputs, n_steps)

# Reshape the data to be suitable format for GRU
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

# Perform predictions
y_pred = gru_model.predict(X_test)

# Inverse transform the values so it is no longer normalized
y_pred = scalar.inverse_transform(y_pred)

#%%----------------------------------------------------------------------------

# Visualize the Results
plt.figure(figsize=(10, 5))
plt.plot(DFtest, color="gray", label="Real")
plt.plot(y_pred, color="red", label="Predicted", linestyle='--')
plt.title("Global Temperature Prediction", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Celsius Mean", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True)

# Save the figure with high DPI
plt.savefig('Temp_Mean_Prediction_GRU.png', dpi=400)

# Show the plot
plt.show()

rmse = np.sqrt(mean_squared_error(DFtest, y_pred))
print("Root Mean Squared Error (RMSE): {:.2f}.".format(rmse))