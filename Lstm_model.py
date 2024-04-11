import os
import logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Check MLflow version
print(mlflow.__version__)

# Load data
df1 = pd.read_csv('sampled_data.csv')
y = df1['Sales']
X = df1.drop(['Sales'], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Initialize MLflow
mlflow.set_experiment('Sales_Prediction')

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Start an MLflow run
with mlflow.start_run(run_name='LSTM'):
    # Train the LSTM model
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_data=(X_test_lstm, y_test))
    
    # Predictions
    y_pred = lstm_model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, y_pred)
    
    # Infer and set the model signature
    signature = infer_signature(X_train_lstm, lstm_model.predict(X_train_lstm))
    mlflow.keras.log_model(lstm_model, 'LSTM_model', signature=signature)

    # Log parameters and metrics
    mlflow.log_params({'epochs': 10, 'batch_size': 64})
    mlflow.log_metric('mse', mse)

# Save the LSTM model
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
model_filename_with_timestamp = f'lstm_model_{timestamp}.pkl'
with open(os.path.join('saved_models', model_filename_with_timestamp), 'wb') as f:
    pickle.dump(lstm_model, f)

print('LSTM model saved successfully.')

# MLflow tracking URI and artifact URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URL
mlflow.tracking.artifact_uri = "s3://your-s3-bucket/path/to/artifacts"  # If storing artifacts in S3

# Log test data with predictions as an artifact
test_data_with_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
test_data_with_predictions.to_csv("test_data_with_predictions.csv", index=False)
mlflow.log_artifact("test_data_with_predictions.csv")

# Generate and save plots
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.legend()
plt.savefig("predicted_sales_plot.png")
mlflow.log_artifact("predicted_sales_plot.png", artifact_path="plots")

# Show the plot
plt.show()
