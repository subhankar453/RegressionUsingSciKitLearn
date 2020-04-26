#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

#read_csv() function is used to convert a data in a csv file to a pandas DataFrame
#The sample() function returns random examples from a dataset equal to the number 
#passed in the arguments of the function. The shape parameter return the dimension
#of a pandas onject
housing_data = pd.read_csv('datasets/housing.csv')
housing_data.sample(5)

#In our model we see from the graphs that there is a cut-off limit to the price
#of a house at 500001 and almost 958 examples out of the 20000 examples have this
#price and this will make our model skewed towards the higher value. There are other
#methods to deal with skewness but for this model we will drop those those values
#from our dataset and to access those dataset we will use the loc method of pandas
#and send the index values of those examples to the drop() function to remove those examples
#Now our ocean_proximity feature is a categorical data and ML models deals with numerical
#values, so we have to convert the categorical data into numnerical data and we do this using
#OneHotEncoder by using the get_dummies function of pandas which will transorm a categorical data 
#into binary values having a value of 0 and 1 for a given set of features. Like if a category has 
#values of X, Y, Z unique categories and for certain training example the value of this attribute is 
#Y, then from oneHotEncoding it is converted to 0,1,0 which are stored in different columns in the dataFrame'
housing_data = housing_data.dropna()
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index)
housing_data.shape
housing_data = pd.get_dummies(housing_data, columns = ['ocean_proximity'])

#Now this is a classification problem we want to compare a house to certain entity
#like we calculate the median of all the houses in the dataset and create a new feature 'above_median'
#to categorize our dataset into houses whose value is higher than the median value or less than the median.
median = housing_data['median_house_value'].median()
housing_data['above_median'] = (housing_data['median_house_value'] - median) > 0

#Now we prepare our input features by dropping the median_house_value & above_median from the dataFrame
#and storing it in a object X and the above_median is our Y value or the final outcome
#we want from our logistic model
X = housing_data.drop(['median_house_value', 'above_median'], axis = 1)
Y = housing_data['above_median']

#Now we split our model into train and test datasets using the train_test_split function
#from sklearn and send the parameters X and Y. The test_size = 0.2 specifies that test_set
#will contain 20% of the original dataset and the rest 80% is in training_set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Here we fit our data to a logistic model classifier and the solver = 'linlinear' argument
#LogisticRegression is to tell sklearn which underlying algorithm to use to train ot=ur model.
#Liblinear is used when we have a small dataset and the classifier is a binary classifier as
#in our case. Then we predict the values for our test set on the model we have trained
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver = 'liblinear').fit(x_train, y_train)
print("Training Score : ", logistic_model.score(x_train, y_train))
y_predict = logistic_model.predict(x_test)

df_pred_actual = pd.DataFrame({'predicted' : y_predict, 'actual' : y_test})

#Here we check the accuracy score which is actually the percentage of predictions our classifer
#was correct to predict
from sklearn.metrics import accuracy_score
print("Testing score : ", accuracy_score(y_test, y_predict))

