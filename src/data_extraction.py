import pandas as pd
import os

def extract_data():
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    data_directory = os.path.join(parent_directory, 'data')

    diabetes_path = os.path.join(data_directory,'diabetes_risk_prediction_dataset.csv')

    df = pd.read_csv(diabetes_path)

    return df