#Importing the packages
import sklearn
import numpy as np
import pandas as pd

#Importing a built in dataset to the program of breast cancer records
from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()

#This dataset has many attributes and to know the description of the dataset we will
#use the DESCR key that is included in the given dataset to give information of the same.
print(breast_cancer_dataset.DESCR)

#The structure of the dataset is in the form of dictionary of a numpy array and to find
#the keys of the dictionary
breast_cancer_dataset.keys()

#To find out the features in the dataset we will use the following command
breast_cancer_dataset.feature_names

#To find the size of the dataset we use the following
breast_cancer_dataset.data.shape

#To findout the type of outputs we are dealing with we use the following to access the
#attribute target_names provided in the dataset
breast_cancer_dataset.target_names

#To know the dimension of target of the dataset we will use the follwoing code
breast_cancer_dataset.target.shape

#Now we load our dataset to a pandas dataframe using the pd.DataFrame function
#Here we create two objects a features object and a target object. Features object
#will have the input data to the model while the target object will have the target values
#columns attribute in the pd.DataFrame object is available to name the columns of the dataset
df_features = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
df_target = pd.DataFrame(breast_cancer_dataset.target, columns = ['cancer'])

#To use the both the features and target in a single dataFrame we can concatenate the 
#features and target objects to obtain a single object cantaining both the dataFrames
#To concatenate two dataframes along the rows we specify the axis to be '1'
df = pd.concat([df_features, df_target], axis = 1)

#To see the first five examples of the dataFrame we use head function
df.head()

#To know the shape(rows and columns) of the dataFrame we use the shape function
df.shape

