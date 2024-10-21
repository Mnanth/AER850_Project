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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
sns.heatmap(abs_corr_matrix, annot=True)
plt.title('Correlation Matrix Plot')

print(corr_matrix)


#Step 4: Classification Model Development/Engineering
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
linear_reg = LinearRegression()
param_grid_lr = {}
grid_search_lr = GridSearchCV(linear_reg,param_grid_lr, scoring = 'neg_mean_absolute_error') #n_jobs = 1 and CV =5 as default
grid_search_lr.fit(X_train,y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)

#Support vector machine

svr = SVC()
param_grid_svr = {'kernel' : ['linear', 'rbf'],
                  'C' : [0.1, 1, 10, 100],
                  'gamma' : ['scale', 'auto']}
grid_search_svr = GridSearchCV(svr,param_grid_svr, scoring = 'neg_mean_absolute_error') #n_jobs = 1 and CV =5 as default
grid_search_svr.fit(X_train, y_train)
best_model_svr = grid_search_svr.best_estimator_
print ("Best SVM Model", best_model_svr)

# Decision Tree Classifier
decision_tree = tree.DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth' :[None, 120, 180, 220],
    'min_samples_split' : [50,60,70],
    'min_samples_leaf' : [7,8,9]}
random_search = RandomizedSearchCV(estimator = decision_tree, param_distributions = param_grid_dt, random_state=42)
random_search.fit(X_train,y_train)
best_model_dt = random_search.best_estimator_
print ("Best Parameters found:", best_model_dt)


#Step 5:Model Performance Analysis


#Support Vector Machine
y_pred_svc = best_model_svr.predict(X_test)
y_pred_binary_svc = (y_pred_svc>0.5).astype(int)
F1_svc = f1_score(y_test,y_pred_svc, average ='macro')
Precision_svc = precision_score(y_test,y_pred_svc, average ='macro')
Accuracy_svc = accuracy_score(y_test,y_pred_svc)
print(f"F1 Score Support Vector Machine: {F1_svc}")
print(f"Precision Score Support Vector Machine: {Precision_svc}")
print(f"Accuracy Score Support Vector Machine: {Accuracy_svc}")

Cm = confusion_matrix(y_test,y_pred_svc)
disp = ConfusionMatrixDisplay(confusion_matrix=Cm)
disp.plot()
plt.title('Confusion Matrix of Support Vector Machine')
plt.show()

#Decision Tree
y_pred_dt = best_model_dt.predict(X_test)
y_pred_binary_dt = (y_pred_dt>0.5).astype(int)
F1_dt = f1_score(y_test,y_pred_dt, average ='weighted')
Precision_dt = precision_score(y_test,y_pred_dt, average ='micro')
Accuracy_dt = accuracy_score(y_test,y_pred_dt)
print(f"F1 Score Decision Tree{F1_dt}")
print(f"Precision Score Decision Tree: {Precision_dt}")
print(f"Accuracy Score Decision Tree: {Accuracy_dt}")

Cm = confusion_matrix(y_test,y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=Cm)
disp.plot()
plt.title('Confusion Matrix of Decision Tree')
plt.show()

#Step 6: Stacked Model Performance Analysis

