import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import mlflow
import mlflow.sklearn
import mlflow.keras

# Sample dataset generation
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100) * 2

# read dataset
df = pd.read_csv(r'finel_train_data.csv')
# Sample a fraction of the dataset (e.g., 50%)
data = df.sample(frac=0.001, random_state=30)  # Adjust the fraction as needed
# Save the sampled data to a new CSV file
data.to_csv('sampled_data.csv', index=False)
        
df1 = pd.read_csv(r'sampled_data.csv')
# Split dataset into train and test sets
y = df1['Sales']
X = df1.drop(['Sales'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regression
with mlflow.start_run(run_name="Decision Tree Regression"):
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)
    y_pred_dt = dt_reg.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    mlflow.log_metric("mse", mse_dt)

# Random Forest Regression
with mlflow.start_run(run_name="Random Forest Regression"):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mlflow.log_metric("mse", mse_rf)

# Linear Regression
with mlflow.start_run(run_name="Linear Regression"):
    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train)
    y_pred_lr = lr_reg.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mlflow.log_metric("mse", mse_lr)

# LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

with mlflow.start_run(run_name="LSTM"):
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=1, verbose=0)
    y_pred_lstm = model_lstm.predict(X_test_lstm).squeeze()
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    mlflow.log_metric("mse", mse_lstm)

# Plotting predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.scatter(X_test, y_pred_dt, color='blue', label='Decision Tree')
plt.scatter(X_test, y_pred_rf, color='green', label='Random Forest')
plt.scatter(X_test, y_pred_lr, color='red', label='Linear Regression')
plt.scatter(X_test, y_pred_lstm, color='orange', label='LSTM')
plt.legend()
plt.title('Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.show()