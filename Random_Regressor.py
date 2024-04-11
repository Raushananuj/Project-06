import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # read experiment parameters
    dataset = str(sys.argv[1]) if len(sys.argv) > 1 else "data.csv"
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    # start mlflow run
    with mlflow.start_run():
        # log run parameters
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("C", C)
        
        # read dataset
        df = pd.read_csv(r'finel_train_data.csv')
        # Sample a fraction of the dataset (e.g., 50%)
        data = df.sample(frac=0.001, random_state=30)  # Adjust the fraction as needed
        # Save the sampled data to a new CSV file
        data.to_csv('sampled_data.csv', index=False)
        
        df1 = pd.read_csv(r'sampled_data.csv')

        # train model
        model = RandomForestRegressor()
        y = df1['Sales']
        x = df1.drop(['Sales'],axis=1)
        model.fit(x, y)

        # score model
        mean_accuracy = model.score(x, y)
        print(f"Mean accuracy: {mean_accuracy}")

        # log run metrics
        mlflow.log_metric("mean accuracy", mean_accuracy)

        # export model
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_uuid
        print(f"Model saved in run {run_id}")