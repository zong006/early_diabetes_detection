from data_extraction import extract_data
from data_preprocessing import pre_processing_data
from algo import randomforestclassifier, logregclassifier
import pandas as pd

def main():
    extracted_data = extract_data()
    X_train, y_train, X_test, y_test = pre_processing_data(extracted_data)

    clf_randomforest = randomforestclassifier(X_train, y_train, X_test, y_test)
    clf_logreg = logregclassifier(X_train, y_train, X_test, y_test)
    
    X = extracted_data.drop(['class'], axis=1)

    feature_importances_rfor = clf_randomforest.feature_importances_
    feature_importance_df_rfor = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances_rfor
    })

    feature_importance_df_rfor = feature_importance_df_rfor.sort_values(by='Importance', ascending=False)
    print("Feature Importance of random forest:\n", feature_importance_df_rfor)

    coefficients = clf_logreg.coef_[0]
    df_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
    df_coefficients = df_coefficients.reindex(df_coefficients['Coefficient'].abs().sort_values(ascending=False).index)
    df_coefficients['Odds Ratio'] = df_coefficients['Coefficient'].apply(lambda x: round(np.exp(x), 4))

    print("Feature Importance of logistic regression:\n",df_coefficients.head(10))
    return


if __name__ == "__main__":
    main()