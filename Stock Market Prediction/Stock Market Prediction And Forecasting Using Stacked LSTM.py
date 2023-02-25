# Data Science Intern LGM FEB 2023
# Task-4 "Stock Market Prediction And Forecasting Using Stacked LSTM"
# DataSet link:- https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv

# Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# uploading dataset
data=pd.read_csv("/content/NSE-TATAGLOBAL.csv")

# printing dataset
print(data.head())
print(data.tail())

# sort with date
data['Date']=pd.to_datetime(data['Date'])
print(type(data['Date']))
print(data.head())

df=data.sort_values(by='Date')
print(df.head())

df.reset_index(inplace=False)
print(df.head())

plt.plot(df['Close'])
df1=df['Close']

#LSTM are sensitive to the scale of data, Therefore applying minmax scaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1)

# Spliting dataset into train and test data
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[:training_size,:],df1[training_size:len(df1),:1]
print(training_size)
print(test_size)

# convert an array of values into dataset matrix
def create_dataset(dataset,time_step=1):
  dataX,dataY=[],[]
  for i in range(len(dataset)-time_step-1):
    a=dataset[i:(i+time_step),0]
    dataX.append(a)
    dataY.append(dataset[i+time_step,0])
  return np.array(dataX),np.array(dataY)

#reshape into x=t,t+1,t+2,t+3 and Y=t+4
time_step=100
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#reshape input which is required to LSTM
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(x_train,y_train,validation_split=0.1,batch_size=64,verbose=1)

# prediction
test_predict=model.predict(x_test)
test_predict1=scaler.inverse_transform(test_predict)
print(test_predict1)

# calculate RMSE performance matrix
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_test,test_predict))
