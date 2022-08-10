# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:30:07 2022

@author: danie
"""

from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

##
filename = 'dataset2.h5'
keyname = '20simpleHam_noise'
filepath = path.abspath(path.join(path.dirname('generate_noise.py'), "..", f"data/{filename}"))
import h5py
f = h5py.File(filepath, 'r')
print([key for key in f.keys()])
# Load a single chunk => much faster
dataset = pd.read_hdf(filepath, keyname)

##
max_val = np.array([x.max() for x in dataset["spectrum"]]).max()
print(max_val)
print(type(dataset["spectrum"]))
noise_01 = np.random.normal(0, 1, size=(dataset["spectrum"].shape[0], dataset["spectrum"][0].shape[0])) * 0.01 * max_val
noise_03 = np.random.normal(0, 1, size=(dataset["spectrum"].shape[0], dataset["spectrum"][0].shape[0])) * 0.03 * max_val
noise_05 = np.random.normal(0, 1, size=(dataset["spectrum"].shape[0], dataset["spectrum"][0].shape[0])) * 0.05 * max_val
##
mod_spectra_01 = np.vstack(dataset["spectrum"]) + noise_01
mod_spectra_03 = np.vstack(dataset["spectrum"]) + noise_03
mod_spectra_05 = np.vstack(dataset["spectrum"]) + noise_05
##

dataset['noise_spectrum_01'] = pd.Series(mod_spectra_01.tolist())
dataset['noise_spectrum_03'] = pd.Series(mod_spectra_03.tolist())
dataset['noise_spectrum_05'] = pd.Series(mod_spectra_05.tolist())

##
dataset.to_hdf('dataset.h5', key='20simpleHam_noise', mode='w')
##
for i in range(5):
    plt.figure(figsize=(10, 6))
    plt.plot(mod_spectra_01[i],
             label=f'2% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(mod_spectra_03[i],
             label=f'5% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(mod_spectra_05[i],
             label=f'10% noise FieldStrength: {dataset["aFieldStrength"][i]:.2f},b {dataset["b"][i]:.5f} , c {dataset["c"][i]:.5f}  ')
    plt.title('absorbtion')
    plt.xlabel('time [steps]')
    plt.legend()
    plt.show()
##
