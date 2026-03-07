"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import time
import numpy as np

from probset import ProbSetup
from mhe import MulticontinuumHomogenization


t1 = time.time()
# pn = 1
pn = int(sys.argv[1])
m = int(sys.argv[2])
ps = ProbSetup(pn, m)


k, P, f = ps.generate_kfI()
k, P = ps.generate_kI_extend()

#### Multicontinuum homogenization
MH = MulticontinuumHomogenization(ps)

UH, UHa, Rs, Ds, Fs = MH.solve(k, P, f, 'm')
np.save('res/ex%d_mh.npy'%pn, UH)
np.save('res/ex%d_mha.npy'%pn, UHa)
np.save('res/ex%d_Rs1.npy'%pn, Rs)
np.save('res/ex%d_Ds1.npy'%pn, Ds)
np.save('res/ex%d_Fs1.npy'%pn, Fs)
t2 = time.time()
print("MH: %.2f (s)"%(t2-t1))

