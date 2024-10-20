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
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


# Step 1: Data Processing
df = pd.read_csv ('/Users/mithu/Documents/GitHub/AER850_Project/Project_1_Data.csv')


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

#Stratified Sampling
df["Coordinates"] = pd.cut(df["Step"], bins = [0,2,4,6, np.inf], labels = [1,2,3,4,])
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in my_splitter.split(df, df["Coordinates"]):
    strat_df_train = df.loc[train_index].reset_index(drop = True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
    strat_df_train = strat_df_train.drop(columns=["Coordinates"], axis =1)
    strat_df_test = strat_df_test.drop(columns=["Coordinates"], axis =1)

#Variable Selection
#X_columns = 'X', 'Y', 'Z'
#y_columns = 'Step'
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]



#Linear Regression
mymodel_1 = LinearRegression()
param_grid_lr = {}
grid_search_lr = GridSearchCV(linear_reg,param_grid_lr, scoring = 'neg_mean_absolute_error')


#Logistic
