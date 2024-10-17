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

#step 2: Data Visualization 

df.hist(bins = 12)
plt.suptitle('Histograms of Coordinates')



#step 3:Correlation Analysis
corr_matrix = df.corr()
print(corr_matrix)
