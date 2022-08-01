# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:57:49 2022

@author: danie
"""

import os
import numpy as np
import pandas as pd
import scipy.linalg as la
from tqdm import tqdm


class sim():
    def __init__(self, key, dt=1, steps=4096, frequency=190,
                 spectralwidth=30):  # ,stepsCont = 2**15-4096 ,delay = int(4096/2)
        self.dt = dt
        self.steps = steps
        self.stepsCont = 2 ** 15 - steps
        self.delay = int(steps / 2)
        self.frequency = frequency
        self.spectralwidth = spectralwidth
        self.key = key

    def generate(self, numberofSpectra, aFieldStrength, b, c, ODscaler, CoupledHamiltonian, randomizeFieldstrength=True,
                 randomizePhase=True, randomizeHamiltonian=False, randomizeODscaler=False):
        dim = CoupledHamiltonian.shape[0]
        eigenvalues0 = np.array(
            [0 * 1j] + list(np.linspace(-0.1 / self.dt, 0.1 / self.dt, dim - 1) + 0.3 / self.dt - .001j))

        eigenvalues, eigenvectors = la.eig(CoupledHamiltonian)
        InitialState = np.zeros(dim, dtype=complex)
        InitialState[0] += 1
        aFieldStrengths, bs, cs, ODscalers = self.getparams(numberofSpectra, aFieldStrength, b, c, ODscaler,
                                                            randomizeFieldstrength, randomizePhase, randomizeODscaler,
                                                            randomizeHamiltonian)

        if numberofSpectra == 1:
            return self.calculate(InitialState, eigenvalues0, eigenvectors, eigenvalues, CoupledHamiltonian, dim,
                                  aFieldStrength, b, c, ODscaler)

        self.data = []
        for indx, (aFieldStrength, b, c, ODscaler) in tqdm(enumerate(zip(aFieldStrengths, bs, cs, ODscalers)),
                                                           total=numberofSpectra):
            spectrum, optdensity = self.calculate(InitialState, eigenvalues0, eigenvectors, eigenvalues,
                                                  CoupledHamiltonian, dim, aFieldStrength, b, c, ODscaler)
            self.data.append(
                [self.dt, self.steps, self.stepsCont, self.delay, self.frequency, self.spectralwidth, aFieldStrength, b,
                 c, ODscaler, dim, CoupledHamiltonian.real, spectrum, optdensity])

            # Every 1000 Spectra save parameters and calculated Spectra, to save RAM empty temporary storage
            if (indx + 1) % 10000 == 0:
                temp = pd.DataFrame(self.data,
                                    columns=["dt", "steps", "stepsCont", "delay", "frequency", "spectralwidth",
                                             "aFieldStrength", "b", "c", "ODscaler", "dim", "CoupledHamiltonian",
                                             "spectrum", "optdensity"])
                temp.to_hdf('dataset.h5', key=self.key + str(int((indx + 1) / 10000)), mode='a')
                self.data = []

    def calculate(self, InitialState, eigenvalues0, eigenvectors, eigenvalues, CoupledHamiltonian, dim, aFieldStrength,
                  b, c, ODscaler):
        # Define pulse (with quadratic phase (chirp) b and third-order phase c)
        Pulsespec = np.array([np.exp(
            -((w - self.frequency) / self.spectralwidth) ** 2 + b * (w - self.frequency) ** 2 * 1j + c * (
                        w - self.frequency) ** 3 * 1j) for w in range(self.steps)])
        Pulse = 2 * np.fft.ifft(Pulsespec)
        Pulse = np.array(list(np.roll(Pulse, self.delay).real) + [0] * self.stepsCont)
        Pulsespec = np.fft.fft(Pulse)

        # *******Initialization of State
        State = InitialState
        StateHistory = np.zeros((self.steps + self.stepsCont, dim), dtype=complex)
        StateHistory[0, 0:dim] = State / np.sqrt(sum(abs(State) ** 2))
        Counter = 0

        # *******Evolve State while pulse is "on"/acting
        for i in range(self.steps):
            State = State * np.exp(-1j * self.dt * eigenvalues0)
            State = np.matmul(eigenvectors.conj().T, State)
            State = State * np.exp(-1j * self.dt * eigenvalues * aFieldStrength * Pulse[Counter])
            State = np.matmul(eigenvectors, State)
            StateHistory[Counter, 0:dim] = State / np.sqrt(sum(abs(State) ** 2))
            Counter += 1

        # *******Evolve State while pulse is "off"/not acting
        for i in range(self.stepsCont):
            State = State * np.exp(-1j * self.dt * eigenvalues0)
            StateHistory[Counter, 0:dim] = State / np.sqrt(sum(abs(State) ** 2))
            Counter += 1

        # Calculate time-dependent dipole moment, and from it the Spectrum
        Dipole = ODscaler * np.array([state.T.conj() @ CoupledHamiltonian @ state for state in StateHistory])
        Spectrum = abs(0.0001j / aFieldStrength * np.fft.fft(Dipole) + Pulsespec)

        # Calculate optical density 
        thresh = abs(Pulsespec).max() * 1e-8
        save_divide = np.divide(Spectrum, abs(Pulsespec), out=np.zeros_like(Spectrum), where=abs(Pulsespec) > thresh)
        optdensity = np.nan_to_num(- np.log10(save_divide, out=np.zeros_like(save_divide), where=save_divide > 0))

        return Spectrum[0:int((self.stepsCont + self.steps) / 10)], optdensity[
                                                                    0:int((self.stepsCont + self.steps) / 10)]

    def getparams(self, numberofSpectra, aFieldStrength, b, c, ODscaler, randomizeFieldstrength, randomizePhase,
                  randomizeODscaler, randomizeHamiltonian):
        aFieldStrengths, bs, cs, ODscalers = None, None, None, None
        if type(aFieldStrength) == int or type(aFieldStrength) == float:
            if randomizeFieldstrength == True:
                aFieldStrengths = np.random.random(numberofSpectra)
            else:
                aFieldStrengths = np.ones(numberofSpectra) * aFieldStrength
        else:
            aFieldStrengths = aFieldStrength

        if (type(b) == int or type(b) == np.float64) and (type(c) == int or type(c) == np.float64):

            if randomizePhase == True:
                bs = (np.random.random(numberofSpectra) - 0.5) * 2 * 0.01
                cs = (np.random.random(numberofSpectra) - 0.5) * 2 * 0.0002
            else:
                bs = np.ones(numberofSpectra) * b
                cs = np.ones(numberofSpectra) * c
        if type(ODscaler) == int or type(ODscaler) == float:
            if randomizeODscaler == True:
                ODscalers = np.random.randint(10, size=numberofSpectra)
            else:
                ODscalers = np.ones(numberofSpectra) * ODscaler
        return aFieldStrengths, bs, cs, ODscalers
# Interacting/coupling part of Hamiltonian one sided only! 
# IntH=np.random.random((dim,dim))*0.0001+np.random.random((dim,dim))*0j+1
# =============================================================================
# dim = 10
# IntH=np.zeros((dim,dim))*1j
# IntH[0]=np.array([1]*dim)
# 
# aFieldStrength = 0.5
# 
# b,c=(np.random.random(2)-0.5)*2*(0.01,0.0002)
# 
# ODscaler = 5
# 
# simulation = sim()
# x = simulation.generate(2,aFieldStrength,b,c,IntH,ODscaler)
# =============================================================================


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
