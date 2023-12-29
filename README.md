A simple machine learning project on a classification task with a slightly imbalanced target class. 


#### DataSet 
The dataset used here for early diabetes detection is from the following Kaggle URL :

https://www.kaggle.com/code/therealsampat/early-stage-diabetes-prediction

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


All categorical variables are nominal.


#### Conclusions from EDA:
- Proportion of males and females amongst positive cases are quite even. About 90% of negative cases are males.
- More females tend to develop diabetes when they are younger compared to males. Most females who develop diabetes are in their early 40s, whilst that for males is at the early 60s.
- Having polyuria and/or polydipsia makes it more likely that there is diabetes due to the strong correlation between these two factors with having diabetes.
- Other features with significant but less strong positive correlation with diabetes are having sudden weight loss, partial paresis and polyphagia.


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, diabetes_risk_prediction_dataset.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into the script data_preprocessing.py, where features are scaled and synthetic data are generated in the training set using the SMOTEEN algorithm. It is subsequently split into training and testing sets stratified by the ratios of the target class, and the script outputs the train and test data.
3. The data is fed into algo.py containing machine learning algorithms which print classification reports for the classification task and outputs the classifier for each algorithm in this script. 


#### Choice of models and evaluations

- This is a classification problem with an imbalanced binary target class, and the randomforest classifier and logistic regression is used here. Both these algorithms are less prone to overfitting and are easy to implement.

- Since this is an imbalanced dataset, we can look at the f1 score where the evaluations are as follows:

1. logistic regression:
    Performs well, with an f1 score of 0.89 and 0.92 for the majority and minority class, respectively. 
2. randomforest:
    Performs much better than logistic regression, with an f1 score of 0.95 and 0.97 for the majority and minority   class,respectively. 
    
    
Top 3 important features in both models are the sex(male/female), and whether or not there is polyuria and polydipsia. This agrees with observations from EDA. For the randomforest algorithm which performs better, the next few important features are sudden weight loss, partial paresis, agreeing with EDA as well. Age however, is a factor that is next on the list, instead of polyphagia, contrary to EDA findings.

