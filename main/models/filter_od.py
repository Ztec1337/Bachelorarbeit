# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:25:48 2022

@author: danie
"""

from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

# %%
# Load all generated data
# =============================================================================
# keys =  [f'20simpleHam{i}' for i in np.arange(1,11,1)]
# dataset = [pd.read_hdf('dataset.h5',key) for key in keys]
# dataset = pd.concat(dataset,ignore_index = True)
# =============================================================================
# %%
filename = 'dataset.h5'
keyname = '20simpleHam1'
filepath = path.abspath(path.join(path.dirname(__file__), "..", "..", f"main/data/{filename}"))
# Load a single chunk => much faster
dataset = pd.read_hdf(filepath, keyname)
# %%
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(dataset['optdensity'][i][1000:2130],
             label=f'FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
plt.legend()
# %%
for i in range(10):
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['optdensity'][i][1000:2130],
             label=f'FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('OD')
    plt.xlabel('time [steps]')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['spectrum'][i][1000:2130],
             label=f'FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.legend()
    plt.title('transmission intensity')
    plt.xlabel('time [steps]')
    plt.show()
