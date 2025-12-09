from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from helper_functions import data_preprocessing, plot_train_test_values, calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

start_date = '1990-01-01'
end_date = '2023-06-01'

data = np.array((pdr.get_data_fred('SP500', start = start_date, end = end_date)).dropna())

data = np.diff(data[:, 0])

num_lags = 100
train_test_split = 0.80
filters = 64
kernel_size = 4
pool_size = 4
pool_size = 2
num_epochs = 100
batch_size = 8

x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

x_train = x_train.reshape((-1, num_lags, 1))
x_test = x_test.reshape((-1, num_lags, 1))

model = Sequential()
model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', input_shape = (num_lags, 1)))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Flatten())
model.add(Dense(units = 1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size)

y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))
y_predicted = np.reshape(model.predict(x_test), (-1, 1))

print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ',
      round(np.corrcoef(y_predicted_train.reshape(-1), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ',
      round(np.corrcoef(y_predicted.reshape(-1), y_test.reshape(-1))[0][1], 3))
print('Model Bias = ', round(model_bias(y_predicted), 2))
print('---')

plot_train_test_values(100, 50, y_train, y_test, y_predicted)