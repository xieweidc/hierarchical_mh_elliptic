"""
    This py file is devoted to test the computation savings in time.
"""

import sys
import time
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

from probset import ProbSetup
from mhe import MulticontinuumHomogenization
from hmh import HierarchicalMultiHomogenization


pn = int(sys.argv[1])
ts = int(sys.argv[2])

if pn == 5:
    L = 3
    nxc = nyc = 5
    NX = NY = 96
    m = 1
else:
    nxc = nyc = 12
    NX = NY = 240
    4m = ceil(2*np.log(nxc))
    if pn in [1, 2]:
        L = 3
    elif pn in [3, 4]:
        L = 2
    

T = np.zeros((10, 2))

for tth in range(ts):
    
    E = 2 ** (L-1)
    Omega = np.array([0, E/nxc, 0, E/nyc])
    nxf = nyf = NX // nxc
    nxc = nyc = E
    NX = NY = nxc*nxf
    
    
    ad = 'res/ex%d_'%pn
    ps = ProbSetup(pn, NX, NY, ad, Omega)
    
    # multicontinuum homogenization
    k, Psi, f = ps.generate_kfI()
    plt.imshow(Psi)
    k, Psi = ps.generate_kI_extend(nxc, nxf, nyc, nyf, m)
    
    MH = MulticontinuumHomogenization(Omega, nxc, nyc, nxf*4, nyf*4, m)
    t1 = time.time()
    f = f.reshape(nxc, nxf, nyc, nyf)
    k = MH.broadcast(k)
    Psi = MH.broadcast(Psi)
    f = MH.broadcast(f)
    
    I = np.arange((MH.nxc+2*m)*(MH.nyc+2*m)).reshape(MH.nxc+2*m, MH.nyc+2*m)
    I = I[:-2*m, :-2*m]
    I = I.reshape(-1)
    J = MH.ada(0, 2*m+1, 2*m+1, MH.nyc+2*m)
    I = I[:, np.newaxis] + J
    
    k = k[I] # (NCC, (2M+1)^2, NXF, NYF)
    k = k.reshape(MH.ncc, 2*m+1, 2*m+1, MH.nxf, MH.nyf)
    k = k.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
    
    Psi = Psi[I] # (NCC, (2M+1)^2, NXF, NYF)
    Psi = Psi.reshape(MH.ncc, 2*m+1, 2*m+1, MH.nxf, MH.nyf)
    Psi = Psi.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
    Rs = np.zeros((MH.ncc, 2, 2))
    Ds = np.zeros((MH.ncc, 4, 4))
    Fs = np.zeros((MH.ncc, 2))
    for i in range(MH.ncc):
        Rs[i], Ds[i], Fs[i] = MH.phi_for_Kim(i, k[i], Psi[i], f[i])
    t2 = time.time()
    T[tth, 0] = t2 -  t1
    Rs1 = Rs.copy()
    Ds1 = Ds.copy()
    Fs1 = Fs.copy()
    
    # hierarchical multicontinuum homogenization
    k, Psi, f = ps.generate_kfI()
    k, Psi = ps.generate_kI_extend(nxc, nxf, nyc, nyf, m)
    
    t1 = time.time()
    HMH = HierarchicalMultiHomogenization(Omega, nxc, nyc, nxf, nyf, m, L)
    xr = nxc // E
    yr = nyc // E
    
    f = f.reshape(xr, E, nxf, yr, E, nyf)
    f = f.transpose(0, 3, 1, 4, 2, 5) # (XR, YR, 4, 4, NXF, NYF)
    f = f.reshape(-1, E**2, nxf, nyf)
    
    I = np.arange(xr) * E * (nyc+2*m)
    I = I[:, np.newaxis] + np.arange(0, nyc, E)
    I = I.reshape(-1)
    J = HMH.ada(0, 2*m+E, 2*m+E, nyc+2*m)
    I = I[:, np.newaxis] + J
    
    k = HMH.reshape(k, xr, yr, I)
    Psi = HMH.reshape(Psi, xr, yr, I)
    
    Rs = np.zeros((xr*yr, E**2, 2, 2))
    Ds = np.zeros((xr*yr, E**2, 4, 4))
    Fs = np.zeros((xr*yr, E**2, 2))
    
    for i in range(xr*yr):
        if L == 2:
            Rs[i], Ds[i], Fs[i] = HMH.phi_for_Kim_L2(i, k[i], Psi[i], f[i])
        if L == 3:
            Rs[i], Ds[i], Fs[i] = HMH.phi_for_Kim_L3(i, k[i], Psi[i], f[i])
    
    Rs = Rs.reshape(xr, yr, E, E, 2, 2)
    Rs = Rs.swapaxes(1, 2)
    Rs = Rs.reshape(HMH.ncc, 2, 2)
    Ds = Ds.reshape(xr, yr, E, E, 4, 4)
    Ds = Ds.swapaxes(1, 2)
    Ds = Ds.reshape(HMH.ncc, 4, 4)
    Fs = Fs.reshape(xr, yr, E, E, 2)
    Fs = Fs.swapaxes(1, 2)
    Fs = Fs.reshape(HMH.ncc, 2)
    
    t2 = time.time()
    T[tth, 1] = t2 -  t1

T = np.mean(T, axis=0)
print("MH: %.2f (s)"%T[0])
print("HMH: %.2f (s)"%T[1])

# def printAB(A, B):
#     for i in range(A.shape[0]):
#         print("---   %d   ---"%i)
#         print(A[i]-B[i])
#         print("---------\n")
        
        
# print("Rs: ")
# printAB(Rs, Rs1)
        
        
# print("Ds: ")
# printAB(Ds, Ds1)
        
        
# print("Fs: ")
# printAB(Fs, Fs1)
        
        
