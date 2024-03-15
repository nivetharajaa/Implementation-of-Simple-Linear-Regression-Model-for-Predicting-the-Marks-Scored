# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Nivetha A 
RegisterNumber: 212222230101 

```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

# Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
1. df.head()

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/588a9f8d-96a2-4fb8-b1c6-d47c9b6d56b3)

2.df.tail()

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/4b2e0ead-d431-49a2-bbe5-0596f4e65ab5)

3. Array value of X

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/6a2874ea-ba26-4ffd-ad37-f34478a9e570)

4. Array value of Y

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/549e40bb-5b61-47b6-90f7-3cce30b57668)

5. Values of Y prediction
   
![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/b4bd5a82-a8b9-4777-9e45-598189ea7524)

6.Array values of Y test
   
![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/806fb86e-dc25-4cd5-b5c9-b9b5be32bd82)

7.Training Set Graph

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/390e6864-e7cf-4c19-9b03-5d4fe1be1c1d)

8.TEST SET GRAPH

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/d44d98e7-f496-4366-b3e0-80b191eb4fa5)

9. Values of MSE, MAE and RMSE

![image](https://github.com/nivetharajaa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120543388/b8c5e634-8214-4777-87eb-242dd2f800d0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
