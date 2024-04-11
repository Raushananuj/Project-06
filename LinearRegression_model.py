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
from sklearn.linear_model import LinearRegression

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

# Initialize MLflow
mlflow.set_experiment('Sales_Prediction')

# Define the Linear Regression model
LinearRegression_model = LinearRegression()

# Train the Linear Regression model
LinearRegression_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = LinearRegression_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

# Log MLflow run
with mlflow.start_run(run_name='LinearRegression'):
    # Log parameters and metrics
    mlflow.log_params({'model': 'Linear Regression'})
    mlflow.log_metric('mse', mse)

    # Save the trained model
    mlflow.sklearn.log_model(LinearRegression_model, "LinearRegression_model")

    # Save the model locally as well
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_filename_with_timestamp = f'LinearRegression_model_{timestamp}.pkl'
    with open(os.path.join('saved_models', model_filename_with_timestamp), 'wb') as f:
        pickle.dump(LinearRegression_model, f)

    print('Linear Regression model saved successfully.')

   
    # Log test data with predictions as an artifact
    test_data_with_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
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
