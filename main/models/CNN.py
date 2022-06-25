# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:50:29 2022

@author: drichter
"""


import os 
from os import path

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm
import datetime

#%%
# Load all generated data
keys =  [f'20simpleHam{i}' for i in np.arange(1,11,1)]
dataset = [pd.read_hdf('dataset.h5',key) for key in keys]
dataset = pd.concat(dataset,ignore_index = True)
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
X_train, X_test, y_train, y_test = train_test_split(X[:,:,None],y, test_size=0.2, random_state=42)

#%%
def create_conv_model(regressionparameters):
    input_shape = (3276,1)
    model = keras.Sequential()  
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv1D(8,strides = 3,kernel_size=(3),padding='valid',activation='relu'))
    model.add(layers.Conv1D(16,strides = 3,kernel_size=(3),padding='valid',activation='relu'))
    model.add(layers.Conv1D(32,strides = 3,kernel_size=(3),padding='valid',activation='relu'))
    model.add(layers.Conv1D(64,strides = 4,kernel_size=(4),padding='valid',activation='relu'))
    model.add(layers.Conv1D(128,strides =5,kernel_size=(5),padding='valid',activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(regressionparameters, activation='linear'))
    return model
#%%
model = create_conv_model(3)
#%%
model.summary()
#%%
opt = tf.keras.optimizers.Adam(lr=0.00005)
lossfunc = 'mean_squared_error'
#%%
model.compile(
    optimizer=opt,
    loss=lossfunc,
    metrics=['accuracy']
    )
#%%
logpath = path.abspath(path.join(path.dirname(__file__), "..", "..", "main\\monitoring\\logs\\fit\\"))
log_dir = logpath +'\\cnn'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#%%
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=200,
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2,
    callbacks=[tensorboard_callback])
#%%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


