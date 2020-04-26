#Import the libraries
import sklearn
import numpy as np
import pandas as pd

#Importing the dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

#To see the keys available in the dataset
boston_dataset.keys()

#To see the descrption of the dataset
print(boston_dataset.DESCR)

#For 20 nwesgroup import and its has different training and test dataset
from sklearn.datasets import fetch_20newsgroups
fetch_20_train = fetch_20newsgroups(subset = 'train')
fetch_20_train.keys()
print(fetch_20_train.DESCR)
fetch_20_train.data[0:2]
fetch_20_train.target_names
fetch_20_train.target

#For digits dataset
from sklearn.datasets import load_digits
digits_dataset = load_digits(n_class = 10)
digits_dataset.keys()
digits_dataset.DESCR

#To show one of the images we will matplotlib
import matplotlib.pyplot as plt
plt.imshow(digits_dataset.images[1], cmap = 'Greys')
plt.show()