#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

#read_csv() function is used to convert a data in a csv file to a pandas DataFrame
#The sample() function returns random examples from a dataset equal to the number 
#passed in the arguments of the function. The shape parameter return the dimension
#of a pandas onject
housing_data = pd.read_csv('datasets/housing.csv')
housing_data.sample(5)
housing_data = housing_data.dropna()
housing_data.shape

#In our model we see from the graphs that there is a cut-off limit to the price
#of a house at 500001 and almost 958 examples out of the 20000 examples have this
#price and this will make our model skewed towards the higher value. There are other
#methods to deal with skewness but for this model we will drop those those values
#from our dataset and to access those dataset we will use the loc method of pandas
#and send the index values of those examples to the drop() function to remove those examples
housing_data.loc[housing_data['median_house_value'] == 500001].count()
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index)
housing_data.shape

#Now our ocean_proximity feature is a categorical data and ML models deals with numerical
#values, so we have to convert the categorical data into numnerical data and we do this using
#OneHotEncoder by using the get_dummies function of pandas which will transorm a categorical data 
#into binary values having a value of 0 and 1 for a given set of features. Like if a category has 
#values of X, Y, Z unique categories and for certain training example the value of this attribute is 
#Y, then from oneHotEncoding it is converted to 0,1,0 which are stored in different columns in the dataFrame'
housing_data['ocean_proximity'].unique()
housing_data = pd.get_dummies(housing_data, columns = ['ocean_proximity'])
housing_data.shape

housing_data.sample(5)

#Now we prepare our input features by dropping the median_house_value from the dataFrame
#and storing it in a object X and the median_house_value is our Y value or the final outcome
#we want from our regression model
X = housing_data.drop('median_house_value', axis = 1)
Y = housing_data['median_house_value']

X.columns

#Now we split our model into train and test datasets using the train_test_split function
#from sklearn and send the parameters X and Y. The test_size = 0.2 specifies that test_set
#will contain 20% of the original dataset and the rest 80% is in training_set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

x_train.shape, x_test.shape
y_train.shape, y_test.shape

#Now we fit our dataset into a linearRegression model and the parameter 'normalize'
#is used because we have different attributes of different ranges and one might higher
#than the other, so we normalize all the values so that all values are in the range of 
#0 - 1 and one attribute doesnot overpower another attribute
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize = True).fit(x_train, y_train)

#Since this is a linear Regression we are using R2 value to know how well our line
#fits our training data and this obtained using the score method
print("Training Score : ", linear_model.score(x_train, y_train))

#here we find the coefficient values to see how each feature/attribute contrinutes
#in finding the price of a house. A very negative value means that particular feature
#is responsible for bring the price down of a house, while a very high coefficient
#means that particular feature is responsible for increasing the price of the house.
predictors = x_train.columns
predictors
coef = pd.Series(linear_model.coef_, predictors).sort_values()
coef

#Now we see our model predicts the values of house for our test set and store in a
#y_predict object and then we create a DataFram using these predicted values of our model
#and the actual values in our dataset called df_pred_actual
y_predict = linear_model.predict(x_test)
df_pred_actual = pd.DataFrame({'predicted' : y_predict, 'actual' : y_test})
df_pred_actual.head(10)

#here also we use R2 score as the metric to see how well our model performs in 
#predicting the house values correctly for our test ddata
from sklearn.metrics import r2_score
print("Training score: ", r2_score(y_test, y_predict))

#The next lines of code to plot the y_predict and y_test(actual value) to understand 
#how close they are to each other 
fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(y_test, y_predict)
plt.show()

df_pred_actual_sample = df_pred_actual.sample(100)
df_pred_actual_sample = df_pred_actual_sample.reset_index()

plt.figure(figsize = (12, 8))
plt.plot(df_pred_actual_sample['predicted'], label = 'Predicted')
plt.plot(df_pred_actual_sample['actual'], label = 'Actual')
plt.ylabel('Median House Value')
plt.legend()
plt.show()