"""
    Low is the first continua, high is the second continua.
"""

import sys
import numpy as np
from math import ceil

from probset import ProbSetup
from mesh import RectangleMesh
from mhe import MulticontinuumHomogenization
from hmh import HierarchicalMultiHomogenization


PI = np.pi
SIN = np.sin
EXP = np.exp


pn = int(sys.argv[1])
# pn = 1
add = 'res/H_'
show = False
Omega = np.array([0, 1, 0, 1])

NX = NY = 240
Nr = 3

ers = np.zeros((3, Nr, 2))
ad = add + 'case%d_'%pn

ps = ProbSetup(pn, NX, NY, ad)

k, Psi, f = ps.generate_kfI()
u = ps.reference_solution(k, f)
np.save(ad+'u.npy', u)

for rth in range(Nr):
    nxc = nyc = 12 * 2**rth
    nxf = nyf = NX // nxc
    m = ceil(2*np.log(nxc))
    # m = 4
    ad = add + 'case%d_%d_%d_'%(pn, nxc, m)
    ps.ad = ad
    print("-------------------")
    print("nxc: %d, m: %d"%(nxc, m))
    
    # Reference solution
    ua = ps.average_u(u, nxc, nxf, nyc, nyf, Psi)
    np.save(ad+'ua.npy', ua)
    
    k, P = ps.generate_kI_extend(nxc, nxf, nyc, nyf, m)
    
    # Multicontinuum homogenization
    # MH = MulticontinuumHomogenization(Omega, nxc, nyc, nxf, nyf, m)
    MH = MulticontinuumHomogenization(Omega, nxc, nyc, nxf*4, nyf*4, m)
    UH, UHa, Rs, Ds, Fs = MH.solve(k, P, f, 't')
    np.save(ad+'uh_mh.npy', UH)
    np.save(ad+'uh_mha.npy', UHa)
    np.save(ad+'Rs.npy', Rs)
    np.save(ad+'Ds.npy', Ds)
    np.save(ad+'Fs.npy', Fs)
    
    ps.plot_pointwise(UH[0], r'$U_1$', 'U1.png')
    ps.plot_pointwise(UH[1], r'$U_2$', 'U2.png')
    
    ps.plot_average(UHa[0], r'$U_1$', 'UH1_a.png')
    ps.plot_average(UHa[1], r'$U_2$', 'UH2_a.png')
    
    ers[0, rth, 0] = np.sqrt(np.sum((ua[0]-UHa[0])**2) / np.sum(ua[0]**2))
    ers[0, rth, 1] = np.sqrt(np.sum((ua[1]-UHa[1])**2) / np.sum(ua[1]**2))
    

    # Hierarchical multicontinuum homogenization
    HMH = HierarchicalMultiHomogenization(Omega, nxc, nyc, nxf, nyf, m)
    hUH, hUHa, Rs1, Ds1, Fs1 = HMH.solve(k, P, f, 't')
    np.save(ad+'uh_hmh.npy', hUH)
    np.save(ad+'uh_hmha.npy', hUHa)
    np.save(ad+'Rs1.npy', Rs1)
    np.save(ad+'Ds1.npy', Ds1)
    np.save(ad+'Fs1.npy', Fs1)
    
    ps.plot_pointwise(hUH[0], r'$U_1$', 'hU1.png')
    ps.plot_pointwise(hUH[1], r'$U_2$', 'hU2.png')
    
    ps.plot_average(hUHa[0], r'$U_1$', 'hUH1_a.png')
    ps.plot_average(hUHa[1], r'$U_2$', 'hUH2_a.png')
    
    ers[1, rth, 0] = np.sqrt(np.sum((ua[0]-hUHa[0])**2) / np.sum(ua[0]**2))
    ers[1, rth, 1] = np.sqrt(np.sum((ua[1]-hUHa[1])**2) / np.sum(ua[1]**2))
    
    ers[2, rth, 0] = np.sqrt(np.sum((UHa[0]-hUHa[0])**2) / np.sum(UHa[0]**2))
    ers[2, rth, 1] = np.sqrt(np.sum((UHa[1]-hUHa[1])**2) / np.sum(UHa[1]**2))
    
    np.save(add+'case%d_ers.npy'%pn, ers)
    
print("ers: \n", ers)