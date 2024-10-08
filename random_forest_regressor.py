import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import os

if __name__ == '__main__':
    # setup logging
    log_folder = 'logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logging.basicConfig(filename=os.path.join(log_folder,
                        "random_forest.log"), filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info("Starting the model training process")

    # Load the dataset
    dataset = (r'/Users/debasishguru/Desktop/untitled folder/Next Project Hike 6/rossmann-store-sales/train_cleaned_df.csv')
    with mlflow.start_run():
        mlflow.log_param('dataset', dataset)
        logging.info(f"Loaded dataset from {dataset}")
        
        try:
            df = pd.read_csv(dataset)
            logging.info(f'Dataset Loaded Successfully with shape {df.shape}')
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise e
        
        # Check if Date column exists and process it
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime format
            # Extract Year, Month, Day as separate features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df = df.drop('Date', axis=1)  # Drop the original Date column after feature extraction
        
        # split the data
        X = df.drop('Sales', axis=1)
        y = df['Sales']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # score model
        mean_accuracy = model.score(X_train, y_train)
        print(f"Mean Accuracy: {mean_accuracy}")
        logging.info(f"Mean Accuracy: {mean_accuracy}")
        mlflow.log_metric('Mean accuracy', mean_accuracy)
        
        # Export model
        mlflow.sklearn.log_model(model, 'model')
        run_id = mlflow.active_run().info.run_uuid 
        logging.info(f"Model saved in run {run_id}")
        print(f"Model saved in run {run_id}")
