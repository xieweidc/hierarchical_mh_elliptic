"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import time
import numpy as np
from math import ceil

from probset import ProbSetup
from mhe import MulticontinuumHomogenization
from hmh import HierarchicalMultiHomogenization


PI = np.pi
SIN = np.sin
EXP = np.exp

pn = int(sys.argv[1])
# pn = 1
add = 'res/'
show = False
Omega = np.array([0, 1, 0, 1])
    
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
k, P, f = ps.generate_kfI()

for rth in range(Nr):
    if pn == 5:
        nxc = nyc = 4 * 2**rth
    else:
        nxc = nyc = 12 * 2**rth
    nxf = nyf = NX // nxc
    m = ceil(2*np.log(nxc))
    ad = add + 'ex%d_%d_%d_'%(pn, nxc, m)
    ps.ad = ad
    
    k, P = ps.generate_kI_extend(nxc, nxf, nyc, nyf, m)
    # Hierarchical multicontinuum homogenization
    t1 = time.time()
    HMH = HierarchicalMultiHomogenization(Omega, nxc, nyc, nxf, nyf, m, L)
    hUH, hUHa = HMH.solve(k, P, f, 't')
    np.save(ad+'uh_hmha.npy', hUHa)
    t2 = time.time()
    print("HMH (H=1/%d, m=%d): %.2f (s)"%(nxc, m, t2-t1))
    