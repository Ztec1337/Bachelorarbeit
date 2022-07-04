# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:25:48 2022

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
plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(dataset['optdensity'][i][1000:2130],label = f'FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
plt.legend()
#%%
plt.figure(figsize=(10,6))
i = 2
plt.plot(dataset['optdensity'][i][1000:2130],label = f'FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
plt.legend()