import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 

#Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(x_test)

#Visualising the Training set results
plt.scatter(x_train, y_train, color = ('red'))
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(x_test, y_test, color = ('red'))
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Final regression equation
Salary = f'{regressor.coef_[0]} X YearsExperience + {regressor.intercept_}'
print(Salary) 