# DATA SCIENCE INTERN - LGMVIP FEB 2023
# TASK-1 Iris flower classification ML project
# Dataset link:- https://www.canva.com/link?target=http%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fdatasets%2FIris&design=DAEjrwWV35w&accessRole=viewer&linkSource=document

#importing some important library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# uploading dataset of iris flowers
column=['sepal length','sepal width','petal length','petal width','lables']
data=pd.read_csv("iris.data",names=column)
print(data.head())
print(data.shape)
print(type(data))

# plotting the dataset
import seaborn as sns
sns.pairplot(data)

# Spliting the dataset in training and testing dataset
x=data.iloc[:,:4]
y=data.iloc[:,4]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Using LinearRegression
# importing the library
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()

# Fitting the model
LR.fit(x_train,y_train)

# predicting the value
y_predict=LR.predict(x_test)
print(y_predict)

# finding the accuracy of model
print("accuracy percent is: ",accuracy_score(y_test,y_predict)*100)

# For new observation
x_new=[[1,2,3,4]]
print(LR.predict(x_new))
