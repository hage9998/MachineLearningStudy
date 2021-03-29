import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#import dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Enconding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])] , remainder= 'passthrough' )
x = np.array(ct.fit_transform(x))

#Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 

#Training The MLR model on Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

#Visualising the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))