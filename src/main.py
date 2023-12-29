from data_extraction import extract_data
from data_preprocessing import pre_processing_data
from algo import randomforestclassifier
import pandas as pd

def main():
    extracted_data = extract_data()
    X_train, y_train, X_test, y_test = pre_processing_data(extracted_data)

    classifier = randomforestclassifier(X_train, y_train, X_test, y_test)
    
    feature_importances = classifier.feature_importances_
    X = extracted_data.drop(['class'], axis=1)
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })


    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importance:\n", feature_importance_df)
    
    return


if __name__ == "__main__":
    main()