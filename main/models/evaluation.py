# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:17:27 2022

@author: danie
"""
import os 
from os import path

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sklearn as sk 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

import datetime
#%%
# Load all generated data
# =============================================================================
# keys =  [f'20simpleHam{i}' for i in np.arange(1,11,1)]
# dataset = [pd.read_hdf('dataset.h5',key) for key in keys]
# dataset = pd.concat(dataset,ignore_index = True)
# =============================================================================
#%%
filename = 'dataset.h5'
keyname = '20simpleHam1'
filepath = path.abspath(path.join(path.dirname(__file__), "..", "..", f"main/data/{filename}"))
# Load a single chunk => much faster
dataset = pd.read_hdf(filepath,keyname)
#%% 
# Scale parameters to have a mean of 0 and std of 1; and split in train/test sets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X,y= np.array(dataset["spectrum"].tolist()),np.array([dataset["aFieldStrength"].tolist(),dataset["b"].tolist(),dataset["c"].tolist()])

# only scale parameters not spectra
sc0,sc1,sc2 = StandardScaler().fit(y[0].reshape(-1,1)),StandardScaler().fit(y[1].reshape(-1,1)),StandardScaler().fit(y[2].reshape(-1,1))
# concatenate scaled parameters and split into training and test set
y = np.array([sc0.transform(y[0].reshape(-1,1)),sc1.transform(y[1].reshape(-1,1)),sc2.transform(y[2].reshape(-1,1))]).T.reshape(-1,3)
# Split in test and train_set, make sure to use same randomstate_ as before during training 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#%%
# ################### EVALUATION OF MODEL ######################
# Load respective models 
model = tf.keras.models.load_model('trained_models/mlp_mae_mape')
model.summary()
#%% 
y_pred = model.predict(X_test)

#%%
absoluteerror = y_pred-y_test

plt.hist(absoluteerror.T[0],bins=100,alpha = 0.9,label = 'aFieldStrength',histtype = 'step')
plt.hist(absoluteerror.T[1],bins=100,alpha = 0.9,label = 'b',histtype = 'step')
plt.hist(absoluteerror.T[2],bins=100,alpha = 0.9,label = 'c',histtype = 'step')
plt.title('Error distributions before inverse scaling')
plt.xlabel('Error')
plt.ylabel('Counts')
plt.legend()
#%%
def scaleinv(scaler,data):
    return scaler.inverse_transform(data)
absoluteerrorinvscale = np.array([scaleinv(sc0,y_pred.T[0])-scaleinv(sc0,y_test.T[0]),scaleinv(sc1,y_pred.T[1])-scaleinv(sc1,y_test.T[1]),scaleinv(sc2,y_pred.T[2])-scaleinv(sc2,y_test.T[2])]).T

plt.hist(absoluteerrorinvscale.T[0],bins=100,alpha = 0.9,label = 'aFieldStrength',histtype = 'step')
plt.hist(absoluteerrorinvscale.T[1],bins=100,alpha = 0.9,label = 'b',histtype = 'step')
plt.hist(absoluteerrorinvscale.T[2],bins=100,alpha = 0.9,label = 'c',histtype = 'step')
plt.title('Error distributions after inverse scaling')
plt.xlabel('Absolute error')
plt.ylabel('Counts')
plt.legend()









