#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:27:34 2023

@author: billhikari
"""

# import packages
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,plot,hist)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary
from scipy import stats
import torch
from prettytable import PrettyTable
# load data
path = '/Users/billhikari/Documents/02450 ML/02450ML_report1/forest_fire.csv'

df = pd.read_csv(path, header = 1)

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

N,M = X.shape

y_fwi = X[:,-1]
X_fwi = X[:,:9]

y = y_fwi.reshape(-1,1)

# Normalize data
X = stats.zscore(X_fwi);
# update M and N
N,M = X.shape

#%%
# cross validation
K = 2
CV = model_selection.KFold(K, shuffle=True,random_state=42)

# linear regression
X_lr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M_lr = M+1
# Values of lambda
lambdas = np.power(10.,range(-5,9))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M_lr,K))
mu = np.empty((K, M))
sigma = np.empty((K, M))
lambda_l = []

# ANN parameters
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 10000 
opt_hidden_units = []
final_loss_l = []
n_hidden_units_l = [1,2]
errors = []
# baseline
baseline = []
baseline_errors_test = np.empty(K)

# Outer crossvalidation loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    # linear regression
    X_train_lr = X_lr[train_index]
    y_train_lr = y[train_index]
    X_test_lr = X_lr[test_index]
    y_test_lr = y[test_index]
    internal_cross_validation = 10    
    
    mu[k, :] = np.mean(X_train_lr[:, 1:], axis=0)
    sigma[k, :] = np.std(X_train_lr[:, 1:], axis=0)
    
    X_train_lr[:, 1:] = (X_train_lr[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test_lr[:, 1:] = (X_test_lr[:, 1:] - mu[k, :]) / sigma[k, :]
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_lr, y_train_lr, lambdas, internal_cross_validation)
    
    Xty = X_train_lr.T @ y_train_lr
    XtX = X_train_lr.T @ X_train_lr
    
    lambda_l.append(opt_lambda)
    lambdaI = opt_lambda * np.eye(M_lr)
    lambdaI[0,0] = 0
    
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    Error_train_rlr[k] = np.sum(np.square(y_train_lr - (X_train_lr @ w_rlr[:, k]).reshape((len(y_train_lr), 1))), axis=0) / np.shape(y_train_lr)[0]
    Error_test_rlr[k] = np.sum(np.square(y_test_lr - (X_test_lr @ w_rlr[:, k]).reshape((len(y_test_lr), 1))), axis=0) / np.shape(y_test_lr)[0]
    
    # baseline
    y_train_mean = np.mean(y[train_index])
    baseline_test_predict = np.full((len(test_index),1),y_train_mean)
    baseline_errors_test[k] = np.mean(np.square(y[test_index] - baseline_test_predict))
    
    # ANN
    # outer cross validation
    print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K))
    # Extract training and testing set for current CV fold, convert to tensors
    X_train_outer = torch.Tensor(X[train_index,:])
    y_train_outer = torch.Tensor(y[train_index])
    X_test_outer = torch.Tensor(X[test_index,:])
    y_test_outer = torch.Tensor(y[test_index])
      
    error_i = np.empty((K, len(n_hidden_units_l)))
    
    # Inner loop for hyperparameter tuning
    for (j, (train_index_inner, test_index_inner)) in enumerate(CV.split(X_train_outer,y_train_outer)):
        print('\nInner crossvalidation fold: {0}/{1}'.format(j+1,K))
          
        # Extract training and testing set for current CV fold, convert to tensors
        X_train_inner = X_train_outer[train_index_inner,:]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner,:]
        y_test_inner = y_train_outer[test_index_inner]
        
        loss_fn = torch.nn.MSELoss()
        for n_hidden_units in n_hidden_units_l:
            print('\nHidden units:{}'.format(n_hidden_units))
            
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_inner,
                                                           y=y_train_inner,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
            print('\n\tBest loss: {}\n'.format(final_loss))
            # Determine estimated class labels for test set
            y_test_est_inner = net(X_test_inner)
        
            # Determine errors and errors
            se_inner = (y_test_est_inner.float()-y_test_inner.float())**2 # squared error
            mse_inner = (sum(se_inner).type(torch.float)/len(y_test_inner)).data.numpy() #mean
            error_i[j,n_hidden_units-1] = mse_inner # save the best model
    
    n_hidden_units = np.unravel_index(np.argmin(error_i), np.asarray(error_i).shape)[1] + 1
    opt_hidden_units.append(n_hidden_units)
    
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units), 
            torch.nn.Tanh(), 
            torch.nn.Linear(n_hidden_units, 1))
    
    final_loss_l.append(final_loss)
    print('Optimal hidden units: {} for outer CV: {}'.format(n_hidden_units, k+1))
    # After inner CV, train on entire outer train set with best model,test on outer test set
    # Determine estimated class labels for test set
    net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_train_outer,
                                                   y=y_train_outer,
                                                   n_replicates=n_replicates,
                                                   max_iter=max_iter)
    y_test_est_outer = net(X_test_outer)
    
    # Determine errors and errors
    se_outer = (y_test_est_outer.float()-y_test_outer.float())**2 # squared error
    mse_outer = (sum(se_outer).type(torch.float)/len(y_test_outer)).data.numpy() #mean
    errors.append(mse_outer)
    # baseline
    baseline.append(np.sum(y[test_index])/len(y[test_index]))
    
    
    
    k = k + 1

#%%
# compute confidence interval
alpha = 0.05
# linear regression
w = np.empty((M_lr,K))
X_train_pe, X_test_pe, y_train_pe, y_test_pe = model_selection.train_test_split(X,
                                                                                y,
                                                                                test_size=0.2,
                                                                                random_state=42)
X_train_pe_lr = np.concatenate((np.ones((len(X_train_pe),1)), X_train_pe), 1)
X_test_pe_lr = np.concatenate((np.ones((len(X_test_pe),1)), X_test_pe), 1)

Xty_pe = X_train_pe_lr.T @ y_train_pe
XtX_pe = X_train_pe_lr.T @ X_train_pe_lr

lambdaI_pe = opt_lambda * np.eye(M_lr)
lambdaI_pe[0,0] = 0

w = np.linalg.solve(XtX_pe+lambdaI_pe,Xty_pe).squeeze()

z_est_lr = np.abs(y_test_pe - (X_test_pe_lr @ w).reshape((len(y_test_pe)), 1)) **2


# ANN
h_pe = opt_hidden_units[errors.index(min(errors))]

model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, h_pe), #M features to n_hidden_units
                    torch.nn.Tanh(),   
                    torch.nn.Linear(h_pe, 1), 
                    )
loss_fn = torch.nn.MSELoss()

net, final_loss, learning_curve = train_neural_net(model,
                                               loss_fn,
                                               X=torch.Tensor(X_train_pe),
                                               y=torch.Tensor(y_train_pe),
                                               n_replicates=n_replicates,
                                               max_iter=max_iter)

y_test_est_pe = net(torch.tensor(X_test_pe).float()).detach().numpy()
z_est_ann = np.abs(y_test_est_pe - y_test_pe) **2

# baseline
y_mean_bl = np.mean(y_train_pe)
baseline_predict_pe = np.full((len(y_test_pe),1),y_mean_bl)
z_est_bl = np.abs(baseline_predict_pe - y_test_pe) **2

# linear vs ANN
z1 = z_est_lr - z_est_ann
CI1 = stats.t.interval(1-alpha, df=len(z1)-1, loc=np.mean(z1), scale=stats.sem(z1))
p1 = 2*stats.t.cdf( -np.abs( np.mean(z1) )/stats.sem(z1), df=len(z1)-1)  # p-value

# linear vs baseline
z2 = z_est_lr - z_est_bl
CI2 = stats.t.interval(1-alpha, df=len(z2)-1, loc=np.mean(z2), scale=stats.sem(z2))
p2 = 2*stats.t.cdf( -np.abs( np.mean(z2) )/stats.sem(z2), df=len(z2)-1)  # p-value

# ANN vs baseline
z3 = z_est_ann - z_est_bl
CI3 = stats.t.interval(1-alpha, df=len(z3)-1, loc=np.mean(z3), scale=stats.sem(z3))
p3 = 2*stats.t.cdf( -np.abs( np.mean(z3) )/stats.sem(z3), df=len(z3)-1)  # p-value


CI_list = [CI1,CI2,CI3]
p_list = [p1,p2,p3]
table1 = PrettyTable()
table1.field_names = ['CI_lr_ann','CI_lr_bl','CI_ann_bl']
table1.add_row(CI_list)
print(table1)

table2 = PrettyTable()
table2.field_names = ['p_lr_ann','p_lr_bl','p_ann_bl']
table2.add_row(p_list)
print(table2)

