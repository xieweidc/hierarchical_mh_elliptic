"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import time
import numpy as np

from probset import ProbSetup


PI = np.pi
SIN = np.sin
EXP = np.exp

pn = int(sys.argv[1])
# pn = 5
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

t1 = time.time()
k, Psi, f = ps.generate_kfI()
u = ps.reference_solution(k, f)
t2 = time.time()
print("Fine grid: %.2f (s)"%(t2-t1))
