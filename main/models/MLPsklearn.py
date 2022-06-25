# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:12:28 2022

@author: drichter
"""

%matplotlib qt5

import os 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk 

from tqdm import tqdm

#%%
# Load all generated data
keys =  [f'20simpleHam{i}' for i in np.arange(1,11,1)]
dataset = [pd.read_hdf('dataset.h5',key) for key in keys]
dataset = pd.concat(dataset,ignore_index = True)
#%%
# Load a single chunk => much faster
dataset = pd.read_hdf('dataset.h5','20simpleHam1')
#%%
# Scale parameters to have a mean of 0 and std of 1; and split in train/test sets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X,y= np.array(dataset["spectrum"].tolist()),np.array([dataset["aFieldStrength"].tolist(),dataset["b"].tolist(),dataset["c"].tolist()])

# only scale parameters not spectra
sc0,sc1,sc2 = StandardScaler().fit(y[0].reshape(-1,1)),StandardScaler().fit(y[1].reshape(-1,1)),StandardScaler().fit(y[2].reshape(-1,1))
# concatenate scaled parameters and split into training and test set
y = np.array([sc0.transform(y[0].reshape(-1,1)),sc1.transform(y[1].reshape(-1,1)),sc2.transform(y[2].reshape(-1,1))]).T.reshape(-1,3)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#%%
from sklearn.neural_network import MLPRegressor
# fit simple multi layer perceptron on scaled input data
mlp = MLPRegressor(hidden_layer_sizes=(100,100),learning_rate_init = 0.001,learning_rate = 'adaptive',max_iter=1,alpha =0.1,tol=0.005,warm_start = True)
# metrics to keep track of 
testscore = []
trainscore = []
#%%
for i in tqdm(range(250)):
    mlp.partial_fit(X_train,y_train)
    if i%5 == 0:
        trainscore.append(mlp.score(X_train,y_train))
        testscore.append(mlp.score(X_test,y_test))  
    #loss.append(mlp.loss_)
#%%
rand = np.random.randint(2000)
print(mlp.predict(X_test[rand].reshape(1,-1)))
print(y_test[rand])
#%% 
#plt.plot(loss)
plt.plot(mlp.loss_curve_)
#%%
plt.plot(trainscore)
plt.plot(testscore)



    
#%%

mlp2 = MLPRegressor(hidden_layer_sizes=(200,100,100,100,32),learning_rate_init = 0.001,learning_rate = 'adaptive',max_iter=1,alpha =0.1,tol=0.005,warm_start = True)
# metrics to keep track of 
testscore2 = []
trainscore2 = []
#%%
for i in tqdm(range(250)):
    mlp2.partial_fit(X_train,y_train)
    if i%5 == 0:
        trainscore2.append(mlp.score(X_train,y_train))
        testscore2.append(mlp.score(X_test,y_test))  

#%%
plt.plot(mlp2.loss_curve_)
















                         







