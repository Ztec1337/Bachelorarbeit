# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:56:10 2022

@author: danie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from Setup import parameters
from simgen import sim


def simpleHamiltonian(dim):
    IntH = np.zeros((dim, dim), dtype=complex)
    IntH[0] = np.ones(dim)

    CoupledHamiltonian = (IntH + IntH.conj().T)
    np.fill_diagonal(CoupledHamiltonian, 0)
    return CoupledHamiltonian


def checkpatHamiltonian(dim, even=True):
    IntH = np.zeros((dim, dim), dtype=complex)
    if even:
        IntH += np.triu((np.arange(0, dim) + np.arange(0, dim)[:, None]) % 2)
    else:
        IntH += np.triu((np.arange(1, dim + 1) + np.arange(0, dim)[:, None]) % 2)

    CoupledHamiltonian = (IntH + IntH.conj().T)
    np.fill_diagonal(CoupledHamiltonian, 0)
    return CoupledHamiltonian


CoupledHamiltonian = simpleHamiltonian(20)

aFieldStrength = 0.2 #np.linspace(0.01,1,100)
#aFieldStrength = np.ones(100)*0.01
b, c = (np.random.random(2) - 0.5) * 2 * (0.01, 0.0002)
b, c = np.zeros(100),np.linspace(-0.0002,0.0002,100)
#b, c = np.zeros(100),np.zeros(100)
ODscaler = 5*np.ones(100)

numberofSpectra = 100

simulation = sim(key='20simpleHam_crafted_cvar2')
simulation.generate(numberofSpectra, aFieldStrength, b, c, ODscaler, CoupledHamiltonian,randomizeFieldstrength=False,randomizePhase=False)

# =============================================================================
# CoupledHamiltonian = simpleHamiltonian(20)
# 
# simulation = sim(key = '20checkpatHam')
# simulation.generate(numberofSpectra,aFieldStrength,b,c,ODscaler,CoupledHamiltonian,)
# =============================================================================

##

