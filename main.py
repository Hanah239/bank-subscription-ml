import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = pd.read_csv("Bank Data Sample.csv")

# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())

# Select required columns
data = data[['age', 'duration', 'campaign', 'previous', 'housing', 'loan', 'y']]

# Manual encoding
housing_map = {'yes': 1, 'no': 0, 'unknown': 2}
loan_map = {'yes': 1, 'no': 0, 'unknown': 2}
target_map = {'yes': 1, 'no': 0}

# Mapping the values
data['housing'] = data['housing'].map(housing_map)
data['loan'] = data['loan'].map(loan_map)
data['y'] = data['y'].map(target_map)

# Separate features and labels
X = data.drop('y', axis=1)
y = data['y']

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Data prepared and split successfully")

print()

# KNN CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN model trained")

# Predictions on test data
knn_predictions = knn.predict(X_test)
print("KNN Predictions (first 10):", knn_predictions[:10])

# Accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

# Precision
knn_precision = precision_score(y_test, knn_predictions)
print("KNN Precision:", knn_precision)

# Recall
knn_recall = recall_score(y_test, knn_predictions)
print("KNN Recall:", knn_recall)

# F1 Score
knn_f1 = f1_score(y_test, knn_predictions)
print("KNN F1 Score:", knn_f1)

# Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_predictions)
print("KNN Confusion Matrix:\n", knn_cm)

# Visualization
plt.scatter(X_test['age'], X_test['duration'], c=knn_predictions)
plt.show()


print()

# DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree model trained")

# Predictions on test data
dt_predictions = dt.predict(X_test)
print("Decision Tree Predictions (first 10):", dt_predictions[:10])

# Accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)

# Precision
dt_precision = precision_score(y_test, dt_predictions)
print("Decision Tree Precision:", dt_precision)

# Recall
dt_recall = recall_score(y_test, dt_predictions)
print("Decision Tree Recall:", dt_recall)

# F1 Score
dt_f1 = f1_score(y_test, dt_predictions)
print("Decision Tree F1 Score:", dt_f1)

# Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_predictions)
print("Decision Tree Confusion Matrix:\n", dt_cm)

# Visualization
tree.plot_tree(dt, feature_names=X.columns, filled=True)
plt.show()

print()

# Final Comparison
print("MODEL COMPARISON")
print("KNN -> Accuracy:", knn_accuracy, "Precision:", knn_precision, "Recall:", knn_recall, "F1:", knn_f1)
print("DT  -> Accuracy:", dt_accuracy, "Precision:", dt_precision, "Recall:", dt_recall, "F1:", dt_f1)