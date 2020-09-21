# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:46:38 2020

@author: Jigar Kumar
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('Churn_Modelling.csv')

# mking a seperate variable
x=df.iloc[:,3:-1]
y=df.iloc[:,-1]

# now we are having a categorical varaibele, so need to  convert into dummy variable
x['Geography'].value_counts()
# by seeing this values of value_counts() i have the total of three category ,
#so we can use the dummy variable

#craeting the dummy variable 
geo=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

#joining the data with main variable x
x=pd.concat([x,geo,gender],axis=1)

#droping the geography and gender
x=x.drop(['Geography','Gender'],axis=1)

#spliting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.fit_transform(x_test) 

#make a NN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

#initialization
model=Sequential()

#input layer and first hidden layer
model.add(Dense(8,kernel_initializer='he_uniform',activation='relu',input_dim=11))

#second layer
model.add(Dense(6,kernel_initializer='he_uniform',activation='relu'))

#output
model.add(Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#compile
model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the  model
fit_model=model.fit(x_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

#evaluation model
y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)