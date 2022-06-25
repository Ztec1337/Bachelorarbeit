# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:55:11 2022

@author: danie
"""

import os 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk 

# Load generated data
dataset = pd.read_hdf('dataset.h5', 'all') 
# filter for relevant data
dataset_filter = dataset.drop(columns = ["dt","steps","stepsCont","frequency","spectralwidth","delay","dim","ODscaler"])

# Plot for some simple correlations
x,y = dataset_filter["aFieldStrength"],np.array(dataset_filter["spectrum"])
#%%
print(x.shape,y.shape)

print(type(y[0]))
#print(y)
#plt.plot(dataset_filter["spectrum"][0])
plt.plot(dataset_filter["optdensity"][0])
plt.show()
#plt.imshow(dataset["CoupledHamiltonian"][0])
#sns.regplot(x,y)
