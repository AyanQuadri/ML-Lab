from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classification algorithms
algorithms = [
    ("Logistic Regression", LogisticRegression(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier())
]

# Evaluate each algorithm
for name, classifier in algorithms:
    
   # Train the classifier
   classifier.fit(X_train, y_train)
   # Make predictions
   y_pred = classifier.predict(X_test)
   
   # Calculate evaluation measures
   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred, average='weighted')
   recall = recall_score(y_test, y_pred, average='weighted')
   f1 = f1_score(y_test, y_pred, average='weighted')
   cm = confusion_matrix(y_test, y_pred)
   
   # For multiclass, calculate TP, TN, FP, FN for each class
   print(f"Algorithm: {name}")
   print(f"Accuracy: {accuracy}")
   print(f"Precision: {precision}")
   print(f"Recall: {recall}")
   print(f"F1 Score: {f1}")
   
   # Calculate TP, TN, FP, FN for each class
   for class_id in range(len(np.unique(y))):
       tp = cm[class_id, class_id]
       fn = cm[class_id, :].sum() - tp
       fp = cm[:, class_id].sum() - tp
       tn = cm.sum() - tp - fn - fp
       
       print(f"\nClass {class_id} \nTrue Negative: {tn}, \nFalse Positive: {fp}, \nFalse Negative: {fn}, \nTrue Positive: {tp}")
   
   print("-----------------------------------")