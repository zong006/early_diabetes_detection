#### Summary

A Random Forest classification model with ScikitLearn to accurately detect early-stage diabetes, achieving exceptional performance with an F1 score exceeding 0.95 across all classes.

#### DataSet 
The dataset used here for early diabetes detection is from the following Kaggle URL :

https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset

- Age: Age of the individual (1.20-65).
- Sex: Gender of the individual (1. Male, 2. Female).
- Polyuria: Presence of excessive urination (1. Yes, 2. No).
- Polydipsia: Excessive thirst (1. Yes, 2. No).
- Sudden Weight Loss: Abrupt weight loss (1. Yes, 2. No).
- Weakness: Generalized weakness (1. Yes, 2. No).
- Polyphagia: Excessive hunger (1. Yes, 2. No).
- Genital Thrush: Presence of genital thrush (1. Yes, 2. No).
- Visual Blurring: Blurring of vision (1. Yes, 2. No).
- Itching: Presence of itching (1. Yes, 2. No).
- Irritability: Display of irritability (1. Yes, 2. No).
- Delayed Healing: Delayed wound healing (1. Yes, 2. No).
- Partial Paresis: Partial loss of voluntary movement (1. Yes, 2. No).
- Muscle Stiffness: Presence of muscle stiffness (1. Yes, 2. No).
- Alopecia: Hair loss (1. Yes, 2. No).
- Obesity: Presence of obesity (1. Yes, 2. No).


- Class: Diabetes classification (1. Positive, 2. Negative).


All categorical variables are nominal



#### Conclusions from EDA:

- Proportion of males and females amongst positive cases are quite even. About 90% of negative cases are males.
- More females tend to develop diabetes when they are younger compared to males. Most females who develop diabetes are in their early 40s, whilst that for males is at the early 60s.
- Having polyuria and/or polydipsia makes it more likely that there is diabetes due to the strong correlation between these two factors with having diabetes.
- Other features with significant but less strong positive correlation with diabetes are having sudden weight loss, partial paresis and polyphagia.


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, train.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into the script data_preprocessing.py, where features are processed, including filling in of missing values, and synthetic data are generated in the training set using the SMOTEEN algorithm. It is subsequently split into training and testing sets stratified by the ratios of the target class, and the script outputs the train and test data.
3. The data is fed into algo.py containing machine learning algorithms which print classification reports for the classification task and outputs the classifier for each algorithm in this script. 


#### Choice of models and evaluations

- This is a classification problem with a slightly imbalanced binary target class, and the random forest classifier and logistic regression algorithms are used here. These three models do not have issues with skewed data.  We can look at the f1 score of these models, given that the target classes are imbalanced.

- The choices and evaluations of the above mentioned models are as follows:

1. logistic regression:
    It is a simple model where feature importance is easily interpretable.
    Performs well on both the majority and minority class, with an f1 score of 0.92 and 0.89, respectively.
    
2. random forest:
    By building multiple decision trees, and considering a random subset of features at each split, this helps to reduce overfitting and improving generalization by making it less sensitive to noise and outliers in individual features. This might further reduce overfitting and is robust on imbalanced classes.
    Performs well on both the majority and minority class, with an f1 score of 0.97 and 0.95, respectively.

    
    
#### Feature Importance
Since the random forest performs best, we look at the feature importances. Top 3 important features are having whether or not the person has polyuria and polydipsia, as well as the sex(male/female). This agrees with observations from EDA. 



