# Importing the required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load the iris dataset
    data = load_iris()
    X = data.data
    y = data.target
    
    print("Dataset Info:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {data.target_names}")
    print()
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Logistic Regression classifier
    logreg_classifier = LogisticRegression(max_iter=200, random_state=42)
    
    # Train the classifier on the training data
    logreg_classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = logreg_classifier.predict(X_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

if __name__ == "__main__":
    main()