"""
Created on Sat Jun  6 14:36:55 2020, modified 2022-02-12, thomas.pfeifer@mpi-hd.mpg.de
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import time

# general parameters:
dt=1 # time step size
steps=4096 # number of time steps
stepsCont=2**15-steps # for adding more steps and evolve quickly after pulse is over
SpectraNumber=1 # number of spectra to calculate

# some conversions, currently not used
times=np.linspace(0,(steps+stepsCont)*dt*0.02419,steps+stepsCont,endpoint=False)
de=1/((steps+stepsCont)*dt)*171.2
eV=np.linspace(0,(steps+stepsCont)*de,steps+stepsCont,endpoint=False)
print(times)
# Light field/pulse parameters
frequency=190
spectralwidth=30
delay=int(steps/2) # time delay of pulse
b=0.01 # quadratic phase
c=0 # cubic phase
aRandomizePhase=0
aFieldStrength=0.01

# Info about the medium: optical density (make sure the absorption lines, at low intensity, are never deeper than 90-95% of the total spectral intensity)
dim=20  # number of states in Hamiltonian
ODscaler=5

#*******Define Hamiltonian
# Non-interacting/non-coupling part of Hamiltonian
eigenvalues0=np.array([0*1j]+list(np.linspace(-0.1/dt,0.1/dt,dim-1)+0.3/dt-.001j))
InitialHamiltonian=np.multiply(np.identity(dim), eigenvalues0[:, np.newaxis])
print(InitialHamiltonian.shape)
##
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
    

# Define figures
fig = plt.figure(figsize=(10,20))
ax = [fig.add_subplot().axis('off')]*(6)
ax[0] = fig.add_subplot(321)
ax[1] = fig.add_subplot(322)
ax[2] = fig.add_subplot(323)
ax[3] = fig.add_subplot(324)
ax[4] = fig.add_subplot(325)
ax[5] = fig.add_subplot(326)


# Main loop (for loop can be removed to just calculate one pulse shape interaction with Hamiltonian)
for SpecCounter in range(SpectraNumber):
    start0=time.time()
 
    # Choose random pulse shapes (only 2nd order and 3rd order phase, more coefficients can be added, scaling factors need to be adjusted when e.g. spectral width or length of arrays are changed)
    if aRandomizePhase!=0:
        b,c=(np.random.random(2)-0.5)*2*(0.01,0.0002)
    
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
    Spectrum=abs(0.0001j/aFieldStrength*np.fft.fft(Dipole)+Pulsespec)**2

        
    # Calculate computing time
    start = time.time()
    
    # Plotting    
    for i in range(len(ax)):
        ax[i].clear()
     
    ax[0].imshow(abs(StateHistory.T),aspect='auto', interpolation='none')
    ax[0].set_title("State Evolution")

    ax[2].plot(times[1850:2250],np.real(Pulse[1850:2250]))
    #ax[2].plot(np.real(Pulse[1850:2250]))
    ax[2].set_title("Temporal Pulse Shape")
    #ax[2].text(0,1.5,'%d' %(SpecCounter))
    ax[2].margins(x=0)
    ax[1].imshow(abs(InitialHamiltonian))
    ax[1].set_title("Free Hamiltonian")
    ax[3].imshow(abs(CoupledHamiltonian))
    ax[3].set_title("Interacting Hamiltonian")
    ax[4].plot(Dipole.real)
    ax[4].set_title("Temporal Dipole")
    ax[5].plot(eV[0:int((stepsCont+steps)/10)],Spectrum[0:int((stepsCont+steps)/10)])
    ax[5].set_title("Absorption Spectrum")

    fig.canvas.flush_events()
    fig.canvas.draw()
    
    # Calculate computing time    
    timenow=time.time()
    print (SpecCounter, timenow-start, timenow-start0)

    


##

