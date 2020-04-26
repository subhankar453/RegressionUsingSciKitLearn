#Importing the libraries
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#Reading the data from a csv file using the read_csv function and fitting it to 
#a pandas dataFrame. In this dataset there are empty values marked with '?'. So
#to drop all these examples we convert the '?' character to NaN of numpy type and 
#then use the dropna() function to remove these examples.
automobile_df = pd.read_csv('datasets/auto-mpg.csv')
automobile_df.sample(5)
automobile_df.shape
automobile_df = automobile_df.replace('?', np.nan)
automobile_df = automobile_df.dropna()
automobile_df.shape

#We can see that the features 'origin' and 'car name' doesnot contribute to the
#mile per gallon(mpg) value and hence we drop these parameters from the input set using 
#the drop() function and the inplace argument if set to True changes the value inline
#and doesnot create another object for it. The model_year parameter may influence the
#mpg value and there we make a better metric out of it by finding the age of the car
#by sutracting the model_year to the current time. Fist we add 19 to the number in model_year
#parameter to make the year 1972 from the given 72 only and then finds the Age and then we
#drop the model_year parameter as it doesnot hold any significance.
automobile_df.drop(['origin', 'car name'], axis = 1, inplace = True)
automobile_df['model year'] = '19' + automobile_df['model year'].astype(str)
automobile_df['age'] = datetime.datetime.now().year - pd.to_numeric(automobile_df['model year'])
automobile_df.drop(['model year'], axis = 1, inplace = True)

#ML models require all data to be in numerical format but on checking we find the
#horsepower parameter to be of Object type nad there we use the to_numeric() function
#to convert the horsepower to numeric type(float642 or int64). The describe function shows
#general metrics over the entire dataset like mean, median etc.
automobile_df.dtypes
automobile_df['horsepower'] = pd.to_numeric(automobile_df['horsepower'], errors = 'coerce')
automobile_df.describe()

#Next we plot mpg versus various input features to know how the mpg varies with
#different features using the matplotlib library
fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['age'], automobile_df['mpg'])
plt.xlabel('Age')
plt.ylabel('Miles per gallon')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['acceleration'], automobile_df['mpg'])
plt.xlabel('Acceleration')
plt.ylabel('Miles per gallon')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['weight'], automobile_df['mpg'])
plt.xlabel('Weight')
plt.ylabel('Miles per gallon')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['displacement'], automobile_df['mpg'])
plt.xlabel('Displacement')
plt.ylabel('Miles per gallon')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['horsepower'], automobile_df['mpg'])
plt.xlabel('horsePower')
plt.ylabel('Miles per gallon')

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(automobile_df['cylinders'], automobile_df['mpg'])
plt.xlabel('Cylinders')
plt.ylabel('Miles per gallon')

#Correlation is a statiscal parameter to understand the direction in which two
#parameters are moving. It is a parameter between 1 and -1. A 1 correlation means
#the two parameters are perfectly correlated in the same direction, while a -1
#correlation means the two parameters are perfectly correlated in opposite directions
#A positive correlation is marked with light colours while negative correlation
#is marked with dark colours. corr() function gives out the correlations between
#parameters of a dataset in numerical values in a tabular format.
#A better way to analyze a correlation is to use a heatMap from seaborn library
#A high positive correlation suggests that the two parameters are very strongly
#related. The 'annot' parameter if set to 'True' shows the value of the correlation
automobile_corr = automobile_df.corr()
automobile_corr
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(automobile_corr, annot = True)

#To shuffle the data before preparing the model we use the sample() function and
#pass the frac parameter equals 1 to retain all the values and then reset_index
automobile_df = automobile_df.sample(frac = 1).reset_index(drop = True)

#Export or save the file in a csv format
automobile_df.to_csv('datasets/auto-mpg-processed.csv', index = False)

