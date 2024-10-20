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
#Regression
#Logistic
