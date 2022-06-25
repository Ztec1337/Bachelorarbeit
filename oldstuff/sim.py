# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:29:35 2022

@author: danie
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.linalg as la
import time as time
from tqdm import tqdm

class sim():
    def __init__(self,dt = 1 ,steps = 4096 , frequency = 190, spectralwidth = 30): #,stepsCont = 2**15-4096 ,delay = int(4096/2)
        self.dt = dt
        self.steps = steps
        self.stepsCont = 2**15-steps
        self.delay = int(steps/2)
        self.frequency = frequency
        self.spectralwidth = spectralwidth
    
    def start(self,aFieldStrength,b,c,IntH,ODscaler):
        start0=time.time()
        dim = IntH.shape[0]
        eigenvalues0=np.array([0*1j]+list(np.linspace(-0.1/self.dt,0.1/self.dt,dim-1)+0.3/self.dt-.005j))
        #InitialHamiltonian=np.multiply(np.identity(dim), eigenvalues0[:, np.newaxis])
        #assert((FreeH.shape == IntH.shape) and (IntH.shape[0] == IntH.shape[1]))
        CoupledHamiltonian=(IntH+IntH.conj().T)
        np.fill_diagonal(CoupledHamiltonian, 0) 
        eigenvalues,eigenvectors=la.eig(CoupledHamiltonian)
        InitialState=np.zeros(dim,dtype = complex)
        InitialState[0]+=1
        # Define pulse (with quadratic phase (chirp) b and third-order phase c)
        Pulsespec=np.array([np.exp(-((w-self.frequency)/self.spectralwidth)**2+b*(w-self.frequency)**2*1j+c*(w-self.frequency)**3*1j) for w in range(self.steps)])
        Pulse=2*np.fft.ifft(Pulsespec)
        Pulse=np.array(list(np.roll(Pulse,self.delay).real)+[0]*self.stepsCont)
        Pulsespec=np.fft.fft(Pulse)
        
        #*******Initialization of State
        State=InitialState
        StateHistory=np.zeros((self.steps+self.stepsCont,dim),dtype = complex)
        StateHistory[0,0:dim]=State/np.sqrt(sum(abs(State)**2))
        Counter=0
        
        #*******Evolve State while pulse is "on"/acting
        for i in range(self.steps):
            State=State*np.exp(-1j*self.dt*eigenvalues0)
            State=np.matmul(eigenvectors.conj().T,State)
            State=State*np.exp(-1j*self.dt*eigenvalues*aFieldStrength*Pulse[Counter])
            State=np.matmul(eigenvectors,State)
            StateHistory[Counter,0:dim]=State/np.sqrt(sum(abs(State)**2))
            Counter+=1
    
        #*******Evolve State while pulse is "off"/not acting        
        for i in range(self.stepsCont):
            State=State*np.exp(-1j*self.dt*eigenvalues0)
            StateHistory[Counter,0:dim]=State/np.sqrt(sum(abs(State)**2))
            Counter+=1
            
        # Calculate time-dependent dipole moment, and from it the Spectrum
        Dipole=ODscaler*np.array([state.T.conj() @ CoupledHamiltonian @ state for state in StateHistory])
        Spectrum=abs(0.0001j/aFieldStrength*np.fft.fft(Dipole))#+Pulsespec) #!!!!!
         
        thresh = abs(Pulsespec).max() *1e-8
        save_divide = np.divide(Spectrum, abs(Pulsespec), out=np.zeros_like(Spectrum), where=abs(Pulsespec)>thresh)
        optdensity = np.nan_to_num(- np.log10(save_divide, out=np.zeros_like(save_divide), where=save_divide>0))
        print(time.time()-start0)
        plt.imshow(abs(StateHistory.T),aspect='auto', interpolation='None')
        plt.show()
        plt.plot(Pulsespec[0:int((self.stepsCont+self.steps)/10)])
        plt.plot(Spectrum[0:int((self.stepsCont+self.steps)/10)])
        return Spectrum[0:int((self.stepsCont+self.steps)/10)],optdensity[0:int((self.stepsCont+self.steps)/10)]
        

# Interacting/coupling part of Hamiltonian one sided only! 
# IntH=np.random.random((dim,dim))*0.0001+np.random.random((dim,dim))*0j+1
dim = 20
IntH=np.zeros((dim,dim))*1j
IntH[0]=np.array([1]*dim)

aFieldStrength = 0.5

b,c=(np.random.random(2)-0.5)*2*(0.01,0.0002)
b,c = 0,0
ODscaler = 5

simulation = sim()
x = simulation.start(aFieldStrength,b,c,IntH,ODscaler)


# =============================================================================
# # Store Absorbtion Spectrum
# Store[ind,12:] = Spectrum[0:int((stepsCont+steps)/10)]
# 
# 
# # Convert to Dataframe Format 
# temp = pd.DataFrame(Store[:,0:12],columns=["dt","steps","stepsCont","frequency","spectralwidth","delay","randomizeFactor","fieldstrength","b","c","dim","ODscaler"])
# temp['data'] = pd.Series(list(Store[:,12:]))
# 
# # Load dataframe, append generated data 
# database = pd.read_csv("dataset.csv")
# database = database.append(temp,ignore_index=True)
# database.to_csv("dataset.csv",index=False) 
# 
# =============================================================================
