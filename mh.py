import numpy as np
from copy import deepcopy
import multiprocessing as mp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from scipy.sparse.linalg import cg
from mesh import RectangleMesh, MHMacroMesh


class MicroPhi:
    def __init__(self, Omega, nxc, nyc, nxf, nyf, m, nc=None):
        self.Omega = Omega
        self.nxc = nxc
        self.nyc = nyc
        self.nxf = nxf
        self.nyf = nyf
        self.ncc = self.nxc * self.nyc
        self.nnc = (self.nxc+1) * (self.nyc+1)
        
        self.hxc = (Omega[1] - Omega[0]) / self.nxc
        self.hyc = (Omega[3] - Omega[2]) / self.nyc
        
        self.m = m 
        self.nc=nc if nc is not None else mp.cpu_count()-2
        
    def ada(self, x0, xn, yn, xr, yr=1):
        """
        Generate the following array.
        
        np.array([[x0,     x0+yr,     ...],
                  [x0+xr,  x0+xr+yr,  ...],
                  [...,    ...,       ...]], shape=(xn, yn))
    
        Parameters
        ----------
        x0 : int
            start number.
        xn : int
            The number in x axis.
        yn : int
            The number in y axis.
        xr : int
            Difference between x axis.
        yr : int, optional
            Difference between y axis. The default is 1.
    
        Returns
        -------
        ix : ndarray
            can be used for find the inner node number.
            
        Examples
        --------
        Get the inner node number of nxc=4, nyc=4.
        
        >>> index = ada(6, 3, 3, 5)
        >>> index = np.array([6,  7,  8, 11, 12, 13, 16, 17, 18])
            
        """
        ix = np.arange(x0, x0+xn*xr, xr)
        iy = np.arange(0, yr*yn, yr)
        return ix[:, np.newaxis] + iy
        
    def phi_for_Kim(self, ith):
        xth, yth = divmod(ith, self.nyc)
        
        ml = min(xth, self.m)
        mr = min(self.nxc-xth-1, self.m)
        mb = min(yth, self.m)
        mt = min(self.nyc-yth-1, self.m)
        
        mxc = ml + mr + 1
        myc = mb + mt + 1
        mxf = mxc * self.nxf
        myf = myc * self.nyf
        NN = (mxf+1) * (myf+1)
        NC = mxc * myc
            
        wKim = np.zeros(4)
        wKim[0] =  (xth-ml)*self.hxc + self.Omega[0]
        wKim[1] =  wKim[0] + self.hxc * mxc
        wKim[2] =  (yth-mb)*self.hyc + self.Omega[2]
        wKim[3] =  wKim[2] + self.hyc * myc
    
        mesh = RectangleMesh(wKim, mxf, myf)
        iaux = (xth-ml)*self.nyc + yth - mb
        iaux = self.ada(iaux, mxc, myc, self.nyc)
        
        ### Left hand side
        k = self.k[iaux] # (mxc, myc, nxf, nyf)
        k = k.swapaxes(1, 2)
        psi = self.Psi[iaux] # (mxc, myc, nxf, nyf)
        psi = psi.swapaxes(1, 2)
        c2d = mesh.cell_to_dof(mxf, myf)
        S = np.full(c2d.shape, mesh.cellm*0.25)
        
        # Constraint
        I0, = np.where(psi.flat == 1)
        J1 = np.arange(NN, NN+mxc*myc).reshape(mxc, myc)
        J1 = np.broadcast_to(J1[:, None, :, None], (mxc, self.nxf, myc, self.nyf))
        J1 = J1.reshape(-1)
        J1[I0] += NC
        J1 = np.broadcast_to(J1[:, None], S.shape) 
        
        A = mesh.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)

        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*NC
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
        
        ### Right hand side
        S = np.zeros((A.shape[0], 6))
        I = np.sum(psi==0, axis=(1, 3)) # (mxc, myc)
        S[-2*NC:-NC, 0] = I.flat * mesh.cellm
        S[-NC:, 1] = (self.nxf*self.nyf-I).flat * mesh.cellm
        I = np.mgrid[wKim[0]:wKim[1]:complex(0,mxf+1),
                     wKim[2]:wKim[3]:complex(0,myf+1)] # (GD, mxf+1, myf+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, LDF)
        I = I.reshape(2, mxc, self.nxf, myc, self.nyf, 4)
        J = np.c_[np.full(c2d.shape[0], 0.25*mesh.cellm), np.zeros(c2d.shape[0])] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 0.25 * mesh.cellm
        J = J.reshape(mxc, self.nxf, myc, self.nyf, 2)
        I = np.einsum('tmpnql, mpnqi -> mnit', I, J) # (mxc, myc, 2, GD)
        I = I.reshape(NC, 4)
        I -= I[ml*myc+mb]
        S[-2*NC:-NC, 2] = I[:, 0]
        S[-2*NC:-NC, 3] = I[:, 1]
        S[-NC:, 4] = I[:, 2]
        S[-NC:, 5] = I[:, 3]
        
        ### Boudary condition
        phi = mesh.set_zero_dirichlet(A, S)
        
        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh.ada(ml*self.nxf*(myf+1)+mb*self.nyf, self.nxf+1, self.nyf+1, myf+1)        
        phi = phi[I] # (NN, 6)
        
        c2d = mesh.cell_to_dof(self.nxf, self.nyf)
        phi = phi[c2d] # (NC, 4, 6)
        A = mesh.cell_stiff_matrix_varphi()
        Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, self.k[ith].flat)
        bs = np.einsum('lmi, l, m -> i', phi[:,:,:2], self.f[ith].flat, np.full(4, mesh.cellm*0.25))
        
        I = np.max(np.abs(Bs[:2, 2:]))
        if I > 1e10:
            print(r"$K_{%d}$ advection coefficient is "%ith, I)
        return Bs[:2,:2], Bs[2:,2:], bs
        
        
class MulticontinuumHomogenization(MicroPhi):
    
    def solve(self, k, Psi, f, way='m'):
        """
            Psi: 0 stands for the 1st continua, 1 for 2nd continua.
        """
        self.k = self.vary_ndarray(k)
        self.Psi = self.vary_ndarray(Psi)
        self.f = self.vary_ndarray(f)
        
        if way == 't':
            Rs = np.zeros((self.ncc, 2, 2))
            Ds = np.zeros((self.ncc, 4, 4))
            Fs = np.zeros((self.ncc, 2))
            for i in range(self.ncc):
                Rs[i], Ds[i], Fs[i] = self.phi_for_Kim(i)
                
        Ds = Ds.reshape(self.ncc, 2, 2, 2, 2)
        # return Rs, Ds, Fs
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa, Rs, Ds, Fs
        
        
    def vary_ndarray(self, a):
        a = a.reshape(self.nxc, self.nxf, self.nyc, self.nyf)
        a = a.swapaxes(1, 2)
        a = a.reshape(-1, self.nxf, self.nyf)
        return a
        

