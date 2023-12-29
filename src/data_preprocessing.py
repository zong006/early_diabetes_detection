import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def pre_processing_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    
    X = df.drop(['class_Positive'], axis=1)
    y = df['class_Positive']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote_enn = SMOTEENN(sampling_strategy='auto')
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    
    return X_train, y_train, X_test, y_test