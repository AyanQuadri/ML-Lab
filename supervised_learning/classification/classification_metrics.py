from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def main():
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
        
        # For multiclass confusion matrix, we'll show the matrix and calculate overall metrics
        cm = confusion_matrix(y_test, y_pred)
        
        # Print the evaluation results
        print(f"Algorithm: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # For multiclass classification, calculate macro-averaged metrics per class
        print("Per-class metrics:")
        for class_idx in range(len(np.unique(y))):
            class_name = data.target_names[class_idx]
            # Calculate TP, TN, FP, FN for each class in one-vs-rest manner
            tp = cm[class_idx, class_idx]
            fn = cm[class_idx, :].sum() - tp
            fp = cm[:, class_idx].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            print(f"  Class {class_name}:")
            print(f"    True Positive: {tp}")
            print(f"    False Positive: {fp}")
            print(f"    True Negative: {tn}")
            print(f"    False Negative: {fn}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()