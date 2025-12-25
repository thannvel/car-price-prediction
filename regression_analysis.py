# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load the data
df = pd.read_csv(r"C:\Users\Dell\Downloads\ToyotaCorolla.csv")
df.head()  # return the first five rows of the dataset
df.isnull().sum()  # confirm the dataset's cleanliness

# perform exploratory analysis through visualization
for column in ["Price", "Age", "KM", "HP", "MetColor", "Automatic", "CC", "Doors", "Weight"]:
    plt.figure()
    df[column].hist()
    plt.title(column)
    plt.show()

# visualize the distribution of the column 'FuelType' using a pie chart
fuel_type_counts = df["FuelType"].value_counts()
plt.figure()
plt.pie(fuel_type_counts, labels = fuel_type_counts.index, autopct = "%1.2f%%")
plt.title("Fuel Type")
plt.show()

# encoding categorical data
encoder = OneHotEncoder(sparse_output = False, drop = "first").set_output(transform = "pandas")
cat_encoded = encoder.fit_transform(df[["FuelType"]])
df1 = pd.concat([df, cat_encoded], axis = 1)
df1.drop(["FuelType"], axis = 1, inplace = True)
df1.head()

# define X and y variables
X = df1[["Age", "KM", "HP", "MetColor", "Automatic", "CC", "Doors", "Weight", "FuelType_Diesel", "FuelType_Petrol"]]
y = df1[["Price"]]

# split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# model definition
lr = LinearRegression()

# fit model
lr.fit(X_train, y_train)

# model prediction
lr_predictions = lr.predict(X_test)

# linear regression model predictions visualization
plt.figure()
plt.scatter(y_test, lr_predictions, color = "yellow")
plt.title("Predicted values vs Actual values")
plt.xlabel("actual values")
plt.ylabel("predicted values")
plt.show()

# evaluation of the predictive model based on the root mean squared error
print("Linear Regression model RMSE:", metrics.mean_squared_error(y_test, lr_predictions, squared = False))
