# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:56:10 2022

@author: danie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from Setup import parameters 
from simgen import sim



def simpleHamiltonian(dim):
    IntH=np.zeros((dim,dim),dtype = complex)
    IntH[0]=np.ones(dim)
    
    CoupledHamiltonian=(IntH+IntH.conj().T)
    np.fill_diagonal(CoupledHamiltonian, 0)   
    return CoupledHamiltonian

def checkpatHamiltonian(dim,even = True):
    IntH = np.zeros((dim,dim),dtype = complex)
    if even: 
        IntH += np.triu((np.arange(0,dim)+np.arange(0,dim)[:,None])%2)
    else: 
        IntH += np.triu((np.arange(1,dim+1)+np.arange(0,dim)[:,None])%2)
        
    CoupledHamiltonian=(IntH+IntH.conj().T)
    np.fill_diagonal(CoupledHamiltonian, 0)   
    return CoupledHamiltonian

    
CoupledHamiltonian = simpleHamiltonian(20)

aFieldStrength = 0.5

b,c=(np.random.random(2)-0.5)*2*(0.01,0.0002)

ODscaler = 5

numberofSpectra = 100000

simulation = sim(key = '20simpleHam')
simulation.generate(numberofSpectra,aFieldStrength,b,c,ODscaler,CoupledHamiltonian)

# =============================================================================
# CoupledHamiltonian = simpleHamiltonian(20)
# 
# simulation = sim(key = '20checkpatHam')
# simulation.generate(numberofSpectra,aFieldStrength,b,c,ODscaler,CoupledHamiltonian,)
# =============================================================================
