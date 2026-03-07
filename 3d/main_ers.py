"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import numpy as np

from probset import ProbSetup
from mesh import HexahedronMesh


# pn = 1

pn = int(sys.argv[1])
m = int(sys.argv[2])
ps = ProbSetup(pn, m)

ers = np.zeros((3, 2))

# Fine-grid solution
ua = np.load('res/ex%d_ua_%d.npy'%(pn, ps.NX))
ua = ua.reshape(2, -1)
# Multicontinuum homogenization solution
UHa = np.load('res/ex%d_mha.npy'%pn)
UHa = UHa.reshape(2, -1)
# Hierarchical solution
HUHa = np.load('res/ex%d_hmha.npy'%pn)
HUHa = HUHa.reshape(2, -1)

mesh = HexahedronMesh(ps.Omega, ps.nxc, ps.nyc, ps.nzc)
mesh.celldata['ua_0'] = ua[0]
mesh.celldata['ua_1'] = ua[1]
mesh.celldata['UHa_0'] = UHa[0]
mesh.celldata['UHa_1'] = UHa[1]
mesh.celldata['HUHa_0'] = HUHa[0]
mesh.celldata['HUHa_1'] = HUHa[1]
mesh.write_to_vtk('coarse.vtu')

# ers[0, 0] = np.sqrt(np.sum((ua[0]-UHa[0])**2) / np.sum(ua[0]**2))
# ers[0, 1] = np.sqrt(np.sum((ua[1]-UHa[1])**2) / np.sum(ua[1]**2))

# ers[1, 0] = np.sqrt(np.sum((ua[0]-HUHa[0])**2) / np.sum(ua[0]**2))
# ers[1, 1] = np.sqrt(np.sum((ua[1]-HUHa[1])**2) / np.sum(ua[1]**2))

# ers[2, 0] = np.sqrt(np.sum((HUHa[0]-UHa[0])**2) / np.sum(UHa[0]**2))
# ers[2, 1] = np.sqrt(np.sum((HUHa[1]-UHa[1])**2) / np.sum(UHa[1]**2))

# print("ua[0]/UHa[0]: \n", ua[0]/UHa[0])
# print("ua[1]/UHa[1]: \n", ua[1]/UHa[1])

# print("ua[0]/HUHa[0]: \n", ua[0]/HUHa[0])
# print("ua[1]/HUHa[1]: \n", ua[1]/HUHa[1])

# print("UHa[0]/HUHa[0]: \n", UHa[0]/HUHa[0])
# print("UHa[1]/HUHa[1]: \n", UHa[1]/HUHa[1])

# print("ers: \n", ers)


c1 = np.mean(ua[0]/UHa[0])
c2 = np.mean(ua[1]/UHa[1])
ers[0, 0] = np.sqrt(np.sum((ua[0]-c1*UHa[0])**2) / np.sum(ua[0]**2))
ers[0, 1] = np.sqrt(np.sum((ua[1]-c2*UHa[1])**2) / np.sum(ua[1]**2))
print("ua[0]/UHa[0]/(%f): \n"%c1, ua[0]/UHa[0]/c1)
print("ua[1]/UHa[1]/(%f): \n"%c2, ua[1]/UHa[1]/c2)

c1 = np.mean(ua[0]/HUHa[0])
c2 = np.mean(ua[1]/HUHa[1])
ers[1, 0] = np.sqrt(np.sum((ua[0]-c1*HUHa[0])**2) / np.sum(ua[0]**2))
ers[1, 1] = np.sqrt(np.sum((ua[1]-c2*HUHa[1])**2) / np.sum(ua[1]**2))
print("ua[0]/HUHa[0]/(%f): \n"%c1, ua[0]/HUHa[0]/c1)
print("ua[1]/HUHa[1]/(%f): \n"%c2, ua[1]/HUHa[1]/c2)

c1 = np.mean(UHa[0]/HUHa[0])
c2 = np.mean(UHa[1]/HUHa[1])
ers[2, 0] = np.sqrt(np.sum((HUHa[0]-UHa[0])**2) / np.sum(UHa[0]**2))
ers[2, 1] = np.sqrt(np.sum((HUHa[1]-UHa[1])**2) / np.sum(UHa[1]**2))
print("UHa[0]/HUHa[0]/(%f): \n"%c1, UHa[0]/HUHa[0])
print("UHa[1]/HUHa[1]/(%f): \n"%c2, UHa[1]/HUHa[1])

print("ers: \n", ers)

