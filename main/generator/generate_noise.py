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
noise_05 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.05*max_val
noise_10 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.10*max_val
noise_15 = np.random.normal(0,1,size = (dataset["spectrum"].shape[0],dataset["spectrum"][0].shape[0]))*0.15*max_val
#%%

mod_spectra_05 = 