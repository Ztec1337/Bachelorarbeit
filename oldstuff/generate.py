# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:05:59 2022

@author: Daniel Richter 
@mainauthor: Thomas Pfeifer
"""
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
import time
from tqdm import tqdm
# general parameters: constant 
dt=1 # time step size
steps=4096 # number of time steps
stepsCont=2**15-steps # for adding more steps and evolve quickly after pulse is over
SpectraNumber=10 # number of spectra to calculate

# Light field/pulse parameters
# constants
delay=int(steps/2) # time delay of pulse

# normal range
f_min,f_max = 100,300 # frequency range
sw_min,sw_max = 10,30 # spectralwidth range

# phase parameters
aRandomizePhase=1
b=0 # quadratic phase
c=0 # cubic phase

aFS_min, aFS_max= 0,1 # aFieldStrength

# Info about the medium: optical density (make sure the absorption lines, at low intensity, are never deeper than 90-95% of the total spectral intensity)
dim=20 # number of states in Hamiltonian
ODscaler=5

# generation parameters 
frequency = np.random.randint(low = f_min, high = f_max, size = SpectraNumber)
spectralwidth = np.random.randint(low = sw_min, high = sw_max, size = SpectraNumber)
aFieldStrengths = np.random.random(size = SpectraNumber)

# frequency constant or not
if 1:
    frequency = 190
    spectralwidth = 30

#*******Define Hamiltonian
# Non-interacting/non-coupling part of Hamiltonian
eigenvalues0=np.array([0*1j]+list(np.linspace(-0.1/dt,0.1/dt,dim-1)+0.3/dt-.001j))
InitialHamiltonian=np.multiply(np.identity(dim), eigenvalues0[:, np.newaxis])

# Interacting/coupling part of Hamiltonian
# IntH=np.random.random((dim,dim))*0.0001+np.random.random((dim,dim))*0j+1
IntH=np.zeros((dim,dim))*1j
IntH[0]=np.array([1]*dim)
CoupledHamiltonian=(IntH+IntH.conj().T)
for i in range(dim):
    CoupledHamiltonian[i,i]=0 
    
#*******Diagonalization of Hamiltonian
eigenvalues,eigenvectors=la.eig(CoupledHamiltonian)
InitialState=np.zeros(dim)*1j

# InitialState[int(dim/2)]=1
InitialState[0]=1+0j

# Storage 1000 Spectra => 16MB RAM
Store = np.zeros((SpectraNumber,12+int((stepsCont+steps)/10))) 
# Set constant parameters 
Store[:,0],Store[:,1],Store[:,2],Store[:,3],Store[:,4],Store[:,5],Store[:,6] = dt,steps,stepsCont,frequency,spectralwidth,delay,aRandomizePhase
Store[:,10],Store[:,11] = dim,ODscaler


# Main loop (for loop can be removed to just calculate one pulse shape interaction with Hamiltonian)
for ind,aFieldStrength in enumerate(tqdm(aFieldStrengths)): # for more random parameters use enumerate(zip(x_1,x_2))
    # Choose random pulse shapes (only 2nd order and 3rd order phase, more coefficients can be added, scaling factors need to be adjusted when e.g. spectral width or length of arrays are changed)
    if aRandomizePhase!=0:
        b,c=(np.random.random(2)-0.5)*2*(0.01,0.0002)
    # Store FieldStrength and 2nd,3rd order Phases
    Store[ind,7],Store[ind,8],Store[ind,9] = aFieldStrength,b,c
        
    # Define pulse (with quadratic phase (chirp) b and third-order phase c)
    Pulsespec=np.array([np.exp(-((w-frequency)/spectralwidth)**2+b*(w-frequency)**2*1j+c*(w-frequency)**3*1j) for w in range(steps)])
    Pulse=2*np.fft.ifft(Pulsespec)
    Pulse=np.array(list(np.roll(Pulse,delay).real)+[0]*stepsCont)
    Pulsespec=np.fft.fft(Pulse)

    #*******Initialization of State
    State=InitialState
    StateHistory=np.zeros((steps+stepsCont,dim))*1j*0
    StateHistory[0,0:dim]=State/np.sqrt(sum(abs(State)**2))
    Counter=0
    
    #*******Evolve State while pulse is "on"/acting
    for i in range(steps):
        State=State*np.exp(-1j*dt*eigenvalues0)
        State=np.matmul(eigenvectors.conj().T,State)
        State=State*np.exp(-1j*dt*eigenvalues*aFieldStrength*Pulse[Counter])
        State=np.matmul(eigenvectors,State)
        # StateHistory+=[State]
        StateHistory[Counter,0:dim]=State/np.sqrt(sum(abs(State)**2))
        # StateHistory=np.r_[StateHistory,[State]]
        Counter+=1

    #*******Evolve State while pulse is "off"/not acting        
    for i in range(stepsCont):
        State=State*np.exp(-1j*dt*eigenvalues0)
        StateHistory[Counter,0:dim]=State/np.sqrt(sum(abs(State)**2))
        Counter+=1
        
    # Calculate time-dependent dipole moment, and from it the Spectrum
    Dipole=ODscaler*np.array([state.T.conj() @ CoupledHamiltonian @ state for state in StateHistory])
    Spectrum=abs(0.0001j/aFieldStrength*np.fft.fft(Dipole))#+Pulsespec)
    
    # Store Absorbtion Spectrum
    Store[ind,12:] = Spectrum[0:int((stepsCont+steps)/10)]


# Convert to Dataframe Format 
temp = pd.DataFrame(Store[:,0:12],columns=["dt","steps","stepsCont","frequency","spectralwidth","delay","randomizeFactor","fieldstrength","b","c","dim","ODscaler"])
temp['data'] = pd.Series(list(Store[:,12:]))

# Load dataframe, append generated data 
database = pd.read_csv("dataset.csv")
database = database.append(temp,ignore_index=True)
database.to_csv("dataset.csv",index=False) 

database = pd.read_csv("dataset.csv")

plt.plot(database["data"][0])