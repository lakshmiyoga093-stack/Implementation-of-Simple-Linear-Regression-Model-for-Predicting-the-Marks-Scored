# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.
```
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yogalakshmi M 
RegisterNumber:  212224020061
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("exp_2_dataset_student_scores.csv")

# Display the dataset
print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())

# Separate input and output
X = df[["Hours"]]
Y = df["Scores"]

print("\nInput (Hours):\n", X)
print("\nOutput (Scores):\n", Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print("\nSlope:", regressor.coef_[0])
print("Intercept:", regressor.intercept_)

# Predict test values
Y_pred = regressor.predict(X_test)

print("\nPredicted Scores:\n", np.round(Y_pred, 2))
print("\nActual Scores:\n", Y_test.values)

# Visualize the training data
plt.scatter(X_train, Y_train, color="blue", label="Training Data")
plt.plot(
    X_train["Hours"],
    regressor.predict(X_train),
    color="red",
    label="Regression Line"
)
plt.title("Hours vs Scores - Training Set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# Visualize the testing data
plt.scatter(X_test, Y_test, color="blue", label="Actual Scores")
plt.scatter(X_test, Y_pred, color="red", label="Predicted Scores")
plt.title("Hours vs Scores - Testing Set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nMean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Predict marks for a student studying 9.25 hours
new_hours = pd.DataFrame({"Hours": [9.25]})
predicted_score = regressor.predict(new_hours)

print("\nHours Studied:", 9.25)
print("Predicted Score:", round(predicted_score[0], 2))
```

## Output:
<img width="731" height="628" alt="image" src="https://github.com/user-attachments/assets/41eed471-a17d-48f1-9156-2746a71e235b" />
<img width="907" height="570" alt="image" src="https://github.com/user-attachments/assets/b5e7209b-976e-41bb-8b0d-1628d2096890" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
