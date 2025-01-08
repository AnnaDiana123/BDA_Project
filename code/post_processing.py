from sklearn.ensemble import RandomForestClassifier

def post_process_with_rf(train_data, train_labels, test_data):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)
    return predicted_labels

def split_stable_and_misclassified(data, labels, threshold=0.5):
    stable_data = data.sample(frac=0.8, random_state=42)
    stable_labels = labels.loc[stable_data.index]
    misclassified_data = data.drop(stable_data.index)
    return stable_data, stable_labels, misclassified_data
