"""
    Permeability field:
        - Small values -> first continuum
        - Large values -> second continuum
"""
import sys
import time
import numpy as np

from probset import ProbSetup
from mhe import MulticontinuumHomogenization, phi_for_Rpm_mh
from hmh import HierarchicalMulticontinuumHomogenization, phi_for_Rpm_hmh


ts = int(sys.argv[1])
T = np.zeros((ts, 2))

class mh_test(MulticontinuumHomogenization):
    def mh_test(self, k, Psi, f):
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = f.reshape(self.nxc*self.nxp, self.nxf//2, self.nyc*self.nyp, 
                      self.nyf//2, self.nzc*self.nzp, self.nzf//2)
        f = self.broadcast(f, 0)
        
        Rpp = np.array([0, (2*self.m+1)*self.hxc/self.nxp, 
                        0, (2*self.m+1)*self.hyc/self.nyp,
                        0, (2*self.m+1)*self.hzc/self.nzp])
    
        Rs = np.zeros((self.ncc, 2, 2))
        Ds = np.zeros((self.ncc, 6, 6))
        Fs = np.zeros((self.ncc, 2))
        for i in range(self.ncc):
            Rs[i], Ds[i], Fs[i] = phi_for_Rpm_mh(k[i], Psi[i], f[i], Rpp)
        
        return Rs, Ds, Fs


class hmh_test(HierarchicalMulticontinuumHomogenization):
    def hmh_test(self, k, Psi, f):
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = f.reshape(self.nxc*self.nxp, self.nxf, self.nyc*self.nyp, 
                      self.nyf, self.nzc*self.nzp, self.nzf)
        f = self.broadcast(f, 0)
        
        Rpp = np.array([0, (2*self.m+1)*self.hxc/self.nxp, 
                        0, (2*self.m+1)*self.hyc/self.nyp,
                        0, (2*self.m+1)*self.hzc/self.nzp])
        
        Rs = np.zeros((self.ncc//8, 8, 2, 2))
        Ds = np.zeros((self.ncc//8, 8, 6, 6))
        Fs = np.zeros((self.ncc//8, 8, 2))
        for i in range(self.ncc//8):
            Rs[i], Ds[i], Fs[i] = phi_for_Rpm_hmh(k[i], Psi[i], f[i], Rpp)
        
        # Reaction term coefficient
        Rs = Rs.reshape(self.nxc//2, self.nyc//2, self.nzc//2, 2, 2, 2, 2, 2)
        Rs = Rs.transpose(0, 3, 1, 4, 2, 5, 6, 7)
        Rs = Rs.reshape(-1, 2, 2)
        # Diffusion term coefficient
        Ds = Ds.reshape(self.nxc//2, self.nyc//2, self.nzc//2, 2, 2, 2, 6, 6)
        Ds = Ds.transpose(0, 3, 1, 4, 2, 5, 6, 7)
        Ds = Ds.reshape(-1, 6, 6)
        # Source term coefficient
        Fs = Fs.reshape(self.nxc//2, self.nyc//2, self.nzc//2, 2, 2, 2, 2)
        Fs = Fs.transpose(0, 3, 1, 4, 2, 5, 6)
        Fs = Fs.reshape(-1, 2)

        return Rs, Ds, Fs


for tth in range(ts):
    pn = 2
    m = 4
    ps = ProbSetup(pn, m)
    
    
    k, P, f = ps.generate_kfI()
    k, P = ps.generate_kI_extend()
    
    ### Multicontinuum homogenization
    t1 = time.time()
    MH = mh_test(ps)
    Rs1, Ds1, Fs1 = MH.mh_test(k, P, f)
    t2 = time.time()
    T[tth, 0] = t2 - t1
    
    ### Hierarchical Multicontinuum homogenization
    t1 = time.time()
    HMH = hmh_test(ps)
    Rs2, Ds2, Fs2 = HMH.hmh_test(k, P, f)
    t2 = time.time()
    T[tth, 1] = t2 - t1


T = np.mean(T, axis=0)
print("MH: %.2f (s)"%T[0])
print("HMH: %.2f (s)"%T[1])

# def printAB(A, B):
#     for i in range(A.shape[0]):
#         er = np.max(np.abs(A[i]-B[i]))
#         if er > 1e-12:
#             print("---   %d   ---   %f"%(i, np.log10(er)))
#             print(A[i]-B[i])
#             print("---------\n")
        
        
# print("Rs: ")
# printAB(Rs1, Rs2)

# print("Ds: ")
# printAB(Ds1, Ds2)

# print("Fs: ")
# printAB(Fs1, Fs2)
        



