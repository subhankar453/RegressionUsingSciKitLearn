#Importing the libraries.
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read_csv() function is used to convert a data in a csv file to a pandas DataFrame
#The sample() function returns random examples from a dataset equal to the number 
#passed in the arguments of the function. The shape parameter return the dimension
#of a pandas onject. At first we choose only a single input 'horsepower' and loop for
#the output
automobile_df = pd.read_csv('datasets/auto-mpg-processed.csv')
from sklearn.model_selection import train_test_split
X = automobile_df[['horsepower']]
Y = automobile_df['mpg']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize = True).fit(x_train, y_train)
print("Training score : ", linear_model.score(x_train, y_train))
y_predict = linear_model.predict(x_test)

from sklearn.metrics import r2_score
print("Testing score : ", r2_score(y_test, y_predict))

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_predict, color = 'r')
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')

X = automobile_df[['age']]
Y = automobile_df['mpg']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
linear_model = LinearRegression(normalize = True).fit(x_train, y_train)
print("Training score : ", linear_model.score(x_train, y_train))
y_predict = linear_model.predict(x_test)
print("Testing score : ", r2_score(y_test, y_predict))

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_predict, color = 'r')
plt.xlabel('Age')
plt.ylabel('Miles per gallon')

X = automobile_df[['displacement', 'horsepower', 'weight']]
Y = automobile_df['mpg']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
linear_model = LinearRegression(normalize = True).fit(x_train, y_train)
print("Training score : ", linear_model.score(x_train, y_train))
y_predict = linear_model.predict(x_test)
print("Testing score : ", r2_score(y_test, y_predict))

predictors = x_train.columns
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)

plt.figure(figsize = (20,10))
plt.plot(y_predict, label = 'Predicted')
plt.plot(y_test.values, label = 'Actual')
plt.ylabel('mpg')
plt.legend()
plt.show()

X = automobile_df[['displacement', 'horsepower', 'weight', 'acceleration', 'cylinders']]
Y = automobile_df['mpg']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
linear_model = LinearRegression(normalize = True).fit(x_train, y_train)
print("Training score : ", linear_model.score(x_train, y_train))
y_predict = linear_model.predict(x_test)
print("Testing score : ", r2_score(y_test, y_predict))

predictors = x_train.columns
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)