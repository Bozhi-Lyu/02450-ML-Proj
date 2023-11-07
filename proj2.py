# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:57:24 2023

@author: lbz
"""
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary
from scipy import stats
import torch
#test
# Load data
path = "Algerian_forest_fires_dataset_UPDATE.csv"
df = pd.read_csv(path, header = 1)

# Pre-processing
df = df.drop(index = [122,123,124])
df = df.drop(columns = ['day','month','year'])
df.columns.values[0] = 'Temp'
raw_data = df.values

cols = range(0,10)
X = raw_data[:, cols]

# convert X string array to float array
X = np.asfarray(X,dtype = float)

# extract the attribute names 
attributeNames = np.asarray(df.columns[cols])

# extract the last column
classLabels = raw_data[:,-1]
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: x.strip())

#determine the class labels
classNames = np.flip(np.unique(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Fire or not
y = np.array([classDict[cl] for cl in classLabels])
C = len(classNames)
N,M = X.shape # N: number of samples, M: number of attributes

# A.1
# We hope to explain FWI based on 1)other variables like temperature, RH, etc.
# 2) just based on ISI and BUI like indicated by Canadian Forestry Service,
y_fwi = X[:,-1]
X_fwi_1 = X[:,:9]
X_fwi_2 = X[:, 7:9]

# Calculate mean and std of columns
mean1 = np.mean(X_fwi_1, axis=0)
std1 = np.std(X_fwi_1, axis=0)

mean2 = np.mean(X_fwi_2, axis=0)
std2 = np.std(X_fwi_2, axis=0)

# Normalization
normalized_X_fwi_1 = (X_fwi_1 - mean1) / std1
normalized_X_fwi_2 = (X_fwi_2 - mean2) / std2




















