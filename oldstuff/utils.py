# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:14:45 2022

@author: danie
"""
# general parameters
dt = 1 # timestep
steps = 4096 # number of time steps
stepsCont = 2**15-steps # for adding more steps and evolve quickly after pulse is over
SpectraNumber = 1 # number of spectra to calculate

# Light field/pulse parameters
frequency = 190
spectralwidth = 30
delay = int(steps/2) # time delay of pulse
b = 0 # quadratic phase
c = 0 # cubic phase
aRandomizePhase = 1
aFieldStrength = 1