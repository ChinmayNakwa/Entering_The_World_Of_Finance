import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from helper_functions import data_preprocessing, mass_import, plot_train_test_values, calculate_accuracy, model_bias
from sklearn.metrics import mean_squared_error

data = np.diff(mass_import(0, 'D1')[:, 3])

num_lags = 500
train_test_split = 0.80

x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split)

model = make_pipeline(StandardScaler(), SGDRegressor(max_iter = 50, tol = 1e-3))
model.fit(x_train, y_train)

y_predicted_train = np.reshape(model.predict(x_train), (-1, 1))

y_predicted = np.reshape(model.predict(x_test), (-1, 1))

print('---')
print('Accuracy Train = ', round(calculate_accuracy(y_predicted_train, y_train), 2), '%')
print('Accuracy Test = ', round(calculate_accuracy(y_predicted, y_test), 2), '%')
print('RMSE Train = ', round(np.sqrt(mean_squared_error(y_predicted_train, y_train)), 10))
print('RMSE Test = ', round(np.sqrt(mean_squared_error(y_predicted, y_test)), 10))
print('Correlation In-Sample Predicted/Train = ', round(np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1], 3))
print('Correlation Out-of-Sample Predicted/Test = ', round(np.corrcoef(np.reshape(y_predicted, (-1)), np.reshape(y_test, (-1)))[0][1], 3))
print('Model Bias = ', round(model_bias(y_predicted), 2))

train_corr = np.corrcoef(np.reshape(y_predicted_train, (-1)), y_train)[0][1]
test_corr = np.corrcoef(np.reshape(y_predicted, (-1)), y_test)[0][1]

if np.isnan(train_corr):
    train_corr = 0.0
if np.isnan(test_corr):
    test_corr = 0.0

print("Correlation Train =", round(train_corr, 3))
print("Correlation Test =", round(test_corr, 3))


print('Model Bias = ', round(model_bias(y_predicted), 2))


plot_train_test_values(100, 50, y_train, y_test, y_predicted)
print('----')