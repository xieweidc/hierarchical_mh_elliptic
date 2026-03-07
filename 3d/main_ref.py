"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import time
import numpy as np

from probset import ProbSetup


pn = int(sys.argv[1])
m = int(sys.argv[2])
# pn = 1

t1 = time.time()
ps = ProbSetup(pn, m)
k, Psi, f = ps.generate_kfI()
u = ps.reference_solution(k, f)
np.save('res/ex%d_u_%d.npy'%(pn, ps.NX), u)

ua = ps.average_u(u, Psi)
np.save('res/ex%d_ua_%d.npy'%(pn, ps.NX), ua)
t2 = time.time()
print("Fine-grid: %.2f (s)"%(t2-t1))

