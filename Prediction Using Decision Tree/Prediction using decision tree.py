# Data Science Intern LGMVIP FEB 2023
# Task-3 "Prediction using Decision Tree Algorithm"
# Dataset link:- https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view

# Importing some important libraries...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Uploading the dataset in csv file format using pandas
data1=pd.read_csv("Iris.csv")
print(data1.head())

# Spliting the dataset
data=data1.iloc[:,1:]
data.head()
x=data.iloc[:,:4]
y=data.iloc[:,4]
data.describe()

# Plotting the data
import seaborn as sns
sns.pairplot(data)

# Spliting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("The shape of x_train",x_train.shape)
print("The shape of y_train",y_train.shape)
print("The shape of x_test",x_test.shape)
print("The shape of y_test",y_test.shape)

# Using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier()

# Fitting the model
model_dtc.fit(x_train,y_train)

# Predicitng the value of y_predict
y_predict=model_dtc.predict(x_test)

# finding the accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

# finding the accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
