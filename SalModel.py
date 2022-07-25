# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:52:11 2022

@author: akash
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

#Reading the CSV file
train_data = pd.read_csv('train_dataset.csv')

#Importing the Train salaries file
train_salaries = pd.read_csv('train_salaries.csv')

#Merging the Training dataset with the Training salaries dataset
train_data = train_data.merge(train_salaries,how = 'inner',on = 'jobId')

#Dropping the Unwanted Data ie: Removing JobID and CompanyID from the Train Dataset.
train_data = train_data.drop(['jobId','companyId'],axis = 1)

#Label Encoding the features of the Dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for x in train_data.columns:
    if x in ['jobType','degree','industry','major']:
        train_data[x] = le.fit_transform(train_data[x])

#Standardizing the features using MinMaxScaler or StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data = pd.DataFrame(scaler.fit_transform(train_data),columns = train_data.columns)

#Splitting the data into dependent and independent features
x = train_data.drop(['salary'],axis = 1)
y = train_data['salary']

#Splitting the Data into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)

#Considering Random Forest Regressor with Hyper-tuned parameters is the best fit model for the current dataset.
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100,min_samples_leaf = 0.1,max_features = 'sqrt',max_depth = 29.0,bootstrap = False)

#Building the Model
rf.fit(x_train,y_train)

#Dumping the model 
pickle.dump(rf,open('SalModel.pkl','wb'),protocol = pickle.HIGHEST_PROTOCOL)
