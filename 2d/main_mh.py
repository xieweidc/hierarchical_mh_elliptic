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
elif pn in [3, 4]:
    NX = NY = 240
    Nr = 3
elif pn == 5:
    NX = NY = 96
    Nr = 1
    
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
    # Multicontinuum homogenization
    t1 = time.time()
    MH = MulticontinuumHomogenization(Omega, nxc, nyc, nxf*4, nyf*4, m)
    UH, UHa = MH.solve(k, P, f, 't')
    np.save(ad+'uh_mha.npy', UHa)
    t2 = time.time()
    print("MH (H=1/%d, m=%d): %.2f (s)"%(nxc, m, t2-t1))
    