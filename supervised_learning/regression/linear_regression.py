import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([0, 0, 1, 1, 1])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# New instance for classification
X_new = np.array([[6]])

# Predict the class using linear regression
y_pred = model.predict(X_new)

# Print the predicted class
if y_pred[0] >= 0.5:
    print("Class 1")
else:
    print("Class 0")