#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:02:05 2024

@author: mithu
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit 
from sklearn.modelselection import cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Step 1: Data Processing
df = pd.read_csv ('/Users/mithu/Documents/GitHub/AER850_Project/Project_1_Data.csv')

print(df.head())
#step 2: Data Visualization 
#Load data for plot
x = df.iloc[:,0] 
y = df.iloc[:,1]
z = df.iloc[:,2]
step = df.iloc[:,3]

#plot
fig=plt.figure()
ax = fig.add_subplot(projection = '3d') 
ax.scatter(x,y,z, marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()



#step 3:Correlation Analysis
corr_matrix = df.corr()
abs_corr_matrix = corr_matrix.abs()
plt.figure()
sns.heatmap(abs_corr_matrix)
plt.title('Correlation Matrix Plot')

print(corr_matrix)


#Step 4 Classification Model Development/Engineering
#Data Splitting 


#X_train, X_test, y_train, y_test = train_test_split
df["Coordinates"] = pd.cut(df["Step"], bins = [0,2,4,6, np.inf], labels = [1,2,3,4,])
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)


#Variable Selection
#X_columns = 'X', 'Y', 'Z'
#y_columns = 'Step'

#Regression
#Logistic
