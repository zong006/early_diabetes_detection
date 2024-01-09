from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def randomforestclassifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)

    print("Confusion Matrix:")
    print(conf_matrix)
    
    return clf


def logregclassifier(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=3)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report_str)

    print("Confusion Matrix:")
    print(conf_matrix)

    return clf