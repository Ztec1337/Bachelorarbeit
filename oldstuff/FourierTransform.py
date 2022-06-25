# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:47:55 2022

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,256,1000)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])

plt.plot(freq, sp.real, freq, sp.imag)