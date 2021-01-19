import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting 

def simpleLinear():
    # import data
    data = pd.read_csv('Salary_Data.csv')
    print(data) # no missing data with this dataset

    # independent (x) and dependent (y)
    x = data.iloc[:, :-1].values # does not work [:, 0] specifying columns might cause of the training error
    y = data.iloc[:, -1].values # does not work [:, 1] specifying columns might cause of the training error
    print(x)
    print(y)

    # skip the handle missing data section

    # splitting the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    # Training linear regression models
    regressor = LinearRegression() # for training on the train datasets
    regressor.fit(x_train, y_train) # fit trains regressor here with x and y data: if errors occur, change iloc[:, here]

    # Predicting test results
    y_pred = regressor.predict(x_test) # from independent values (x), this predicts the dependent values (y): real y values are y_test holds

    # plotting the training results
    plt.scatter(x_train, y_train, color='red') # creating scatter plots
    plt.plot(x_train, regressor.predict(x_train), color='blue') # creating the linear line for the plot
    plt.title('Salary vs Experience (Training Sets)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

    # plotting the test results
    plt.scatter(x_test, y_test, color='green', marker='X')  # creating scatter plots
    plt.plot(x_train, regressor.predict(x_train), color='blue')  # do not change x and y because it is linear regression and predict the values with the same plot
    plt.title('Salary vs Experience (Test Sets)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

    # predicting year 12 salary
    pred_salary = regressor.predict([[12]]) # 2D array
    print("12 years of salary will be: %d dollars" %pred_salary)

    # knowing equations y^ = b0 + b1x1
    print(regressor.intercept_) # b0
    print(regressor.coef_) # b1

if __name__ == '__main__':
    simpleLinear()
