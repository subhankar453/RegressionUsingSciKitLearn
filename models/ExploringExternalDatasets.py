#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the data from a csv file to a pandas DataFrame
housing_data = pd.read_csv('datasets/housing.csv')

#Show the first five data examples of our dataset
housing_data.head()

#Show randomly five data examples of our dataset
housing_data.sample(5)

#Show the shape(rows and columns) of our dataset
housing_data.shape

#Some dataset may have missing values and we have to preprocess our data
#to use in an algorithm. One of the option to deal with missing values is to
#remove the example having the missing data all together
housing_data = housing_data.dropna()
housing_data.shape

#The describe() function some of the common statistics of the entire data like
#mean, median etc.
housing_data.describe()

#The dataset has an attribute 'ocean_proximity' which is a categorical data
#To know all the unique values a categorical data has over a dataset is given
#the unique function.
housing_data['ocean_proximity'].unique()

#We will matplotlib to plot our data
#In this we will use a scatter plot to plot a graph between total rooms and
#median house value of our dataset. plt.scatter is used to plot a scatter plot
#the first paramter in scatter() represents x axis values and the second parameter
#represents y axis values. xlabel() and ylabel() functions are used to given labels
#to the two axis of a plot. The next two plots are between median house age and
#median house value and the last one is between median income and median house value.
fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(housing_data['total_rooms'], housing_data['median_house_value'])
plt.xlabel('Total rooms')
plt.ylabel('Median house value')

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(housing_data['housing_median_age'], housing_data['median_house_value'])
plt.xlabel('Median age')
plt.ylabel('Median house value')

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(housing_data['median_income'] * 10000, housing_data['median_house_value'])
plt.xlabel('Median Income')
plt.ylabel('Median house value')

#Correlation is a statiscal parameter to understand the direction in which two
#parameters are moving. It is a parameter between 1 and -1. A 1 correlation means
#the two parameters are perfectly correlated in the same direction, while a -1
#correlation means the two parameters are perfectly correlated in opposite directions
#A positive correlation is marked with light colours while negative correlation
#is marked with dark colours. corr() function gives out the correlations between
#parameters of a dataset in numerical values in a tabular format.
housing_data_corr = housing_data.corr()
housing_data_corr

#A better way to analyze a correlation is to use a heatMap from seaborn library
#A high positive correlation suggests that the two parameters are very strongly
#related. The 'annot' parameter if set to 'True' shows the value of the correlation
fig, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(housing_data_corr, annot = True)