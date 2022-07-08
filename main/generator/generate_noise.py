# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:30:07 2022

@author: danie
"""

import os 
from os import path

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from tqdm import tqdm
#%%
filename = 'dataset.h5'
keyname = '20simpleHam1'
filepath = path.abspath(path.join(path.dirname(__file__), "..", "..", f"main/data/{filename}"))
# Load a single chunk => much faster
dataset = pd.read_hdf(filepath,keyname)
#%%
max_val = np.array([x.max() for x in dataset["spectrum"]]).max()
print(max_val)
noise_02 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.02*max_val
noise_05 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.05*max_val
noise_10 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.10*max_val
noise_15 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.15*max_val
#%%
mod_spectra_02 = np.vstack(dataset["spectrum"])+noise_02
mod_spectra_05 = np.vstack(dataset["spectrum"])+noise_05
mod_spectra_10 = np.vstack(dataset["spectrum"])+noise_10
mod_spectra_15 = np.vstack(dataset["spectrum"])+noise_15
#%%
for i in range(5):
    plt.figure(figsize=(10,6))
    plt.plot(mod_spectra_02[i],label = f'2% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend() 
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(mod_spectra_05[i],label = f'5% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend() 
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(mod_spectra_10[i],label = f'10% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend() 
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(mod_spectra_15[i],label = f'15% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend() 
    plt.show()