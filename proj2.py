# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:57:24 2023

@author: lbz
"""
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary
from scipy import stats
import torch

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

# A.2
# choose lambda from e-6 to e3
lambda_range = np.power(10.,range(-6,3))

kfold = model_selection.KFold(n_splits=10)
generalization_errors = []
 
k = 0
for lambda_value in lambda_range:

    validation_errors = []
    
    for train_index, val_index in kfold.split(normalized_X_fwi_1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 创建Ridge回归模型，并使用当前λ值训练模型
        model = Ridge(alpha=lambda_value)
        model.fit(X_train, y_train)
        
        # 在验证集上进行预测
        y_pred = model.predict(X_val)
        
        # 计算验证误差（均方误差）
        val_error = mean_squared_error(y_val, y_pred)
        validation_errors.append(val_error)
    
    # 计算当前λ值下的平均验证误差，作为广义误差估计
    generalization_error = np.mean(validation_errors)
    generalization_errors.append(generalization_error)

# 打印每个λ值下的广义误差
for i, lambda_value in enumerate(lambda_range):
    print(f"λ: {lambda_value}, Generalization Error: {generalization_errors[i]}")

















