"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import numpy as np
from math import ceil

from mesh2d import RectangleMesh
from probset import ProbSetup


PI = np.pi
SIN = np.sin
EXP = np.exp

pn = int(sys.argv[1])
# pn = 1
add = 'res/'
show = False
    
if pn in [1, 2]:
    NX = NY = 240
    Nr = 3
    L = 3
elif pn in [3, 4]:
    NX = NY = 240
    Nr = 3
    L = 2
elif pn == 5:
    NX = NY = 96
    Nr = 1
    L = 2

ers = np.zeros((3, Nr, 2))
ad = add + 'ex%d_'%pn

ps = ProbSetup(pn, NX, NY, ad)
k, Psi, f = ps.generate_kfI()

for rth in range(Nr):
    if pn == 5:
        nxc = nyc = 4 * 2**rth
    else:
        nxc = nyc = 12 * 2**rth
    nxf = nyf = NX // nxc
    m = ceil(2*np.log(nxc))
    ad = add + 'ex%d_%d_%d_'%(pn, nxc, m)
    ps.ad = ad
    print("-------------------")
    print("nxc: %d, m: %d"%(nxc, m))
    
    # Reference solution
    u = np.load(add+'ex%d_u.npy'%pn)
    mesh = RectangleMesh(ps.Omega, ps.NX*4, ps.NY*4)
    c2d = mesh.cell_to_dof(ps.NX*4, ps.NY*4)
    u = u[c2d]
    ua = ps.average_u(u, nxc, nxf, nyc, nyf, Psi)
    np.save(ad+'ua.npy', ua)
    
    k, P = ps.generate_kI_extend(nxc, nxf, nyc, nyf, m)
    
    # Error 1
    UHa = np.load(ad+'uh_mha.npy')
    ers[0, rth, 0] = np.sqrt(np.sum((ua[0]-UHa[0])**2) / np.sum(ua[0]**2))
    ers[0, rth, 1] = np.sqrt(np.sum((ua[1]-UHa[1])**2) / np.sum(ua[1]**2))
    
    # Error 2
    hUHa = np.load(ad+'uh_hmha.npy')
    ers[1, rth, 0] = np.sqrt(np.sum((ua[0]-hUHa[0])**2) / np.sum(ua[0]**2))
    ers[1, rth, 1] = np.sqrt(np.sum((ua[1]-hUHa[1])**2) / np.sum(ua[1]**2))

    # Error 3
    ers[2, rth, 0] = np.sqrt(np.sum((UHa[0]-hUHa[0])**2) / np.sum(UHa[0]**2))
    ers[2, rth, 1] = np.sqrt(np.sum((UHa[1]-hUHa[1])**2) / np.sum(UHa[1]**2))
    
    np.save(add+'ex%d_ers.npy'%pn, ers)

print("ers: \n", ers)
