import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from mesh2d import RectangleMesh

PI = np.pi
SIN = np.sin
EXP = np.exp

class ProbSetup:
    def __init__(self, pn, NX, NY, ad, Omega=np.array([0, 1, 0, 1])):
        self.pn = pn
        self.NX = NX
        self.NY = NY
        self.ad = ad
        self.Omega = Omega
    
    def average_u(self, u, nxc, nxf, nyc, nyf, Psi):
        
        u = u.reshape(nxc, nxf, 4, nyc, nyf, 4, 4)
        J = Psi.astype(np.float64)
        J = np.stack(((1-J)*0.25, J*0.25))
        J = J.reshape(2, nxc, nxf, nyc, nyf)
        I = np.count_nonzero(J, axis=(2, 4)) # (2, NXC, NYC)
        ua = np.einsum('imrjnsl, timjn, tij -> tij', u, J, 1/I/16) # (2, NXC, NYC)
        return ua
    
    def generate_kfI(self):
        
        hx = (self.Omega[1] - self.Omega[0]) / self.NX / 2
        hy = (self.Omega[3] - self.Omega[2]) / self.NY / 2

        X, Y = np.mgrid[self.Omega[0]:self.Omega[1]:complex(0,self.NX+1), 
                        self.Omega[2]:self.Omega[3]:complex(0,self.NY+1)]
        X = X[1:, 1:] - hx
        Y = Y[1:, 1:] - hy
        X = X.flatten()
        Y = Y.flatten()
        
        Ihigh = self.psi_function(X, Y)
        
        k = (2 + SIN(PI*X)*SIN(PI*Y)) * 1e-4
        # k = np.full(X.shape, 1e-4)
        k[Ihigh] = k[Ihigh] * 1e4
        
        Psi = np.zeros(self.NX*self.NY, dtype=np.int_)
        Psi[Ihigh] = 1
        f = EXP(-40*((X-0.5)**2 + (Y-0.5)**2)) * 1e-2
        # f = np.full(X.shape, 1e-2)
        f[Ihigh] = f[Ihigh] * 1e2
        
        k = k.reshape(self.NX, self.NY)
        Psi = Psi.reshape(self.NX, self.NY)
        f = f.reshape(self.NX, self.NY)
        return k, Psi, f
    
    def generate_kI_extend(self, nxc, nxf, nyc, nyf, m):
        mxc = nxc + 2*m
        myc = nxc + 2*m
        NX = mxc * nxf
        NY = myc * nyf
        hxc = (self.Omega[1] - self.Omega[0]) / nxc
        hyc = (self.Omega[3] - self.Omega[2]) / nyc
        hx = hxc / nxf / 2
        hy = hyc / nyf / 2
        Omega = np.array([self.Omega[0]-hxc*m, self.Omega[1]+hxc*m,
                          self.Omega[2]-hyc*m, self.Omega[3]+hyc*m])

        X, Y = np.mgrid[Omega[0]:Omega[1]:complex(0,NX+1), 
                        Omega[2]:Omega[3]:complex(0,NY+1)]
        X = X[1:, 1:] - hx
        Y = Y[1:, 1:] - hy
        X = X.flatten()
        Y = Y.flatten()
        
        Ihigh = self.psi_function(X, Y)
        
        k = (2 + SIN(PI*X)*SIN(PI*Y)) * 1e-4
        # k = np.full(X.shape, 1e-4)
        k[Ihigh] = k[Ihigh] * 1e4
        
        Psi = np.zeros(NX*NY, dtype=np.int_)
        Psi[Ihigh] = 1
        
        k = k.reshape(mxc, nxf, myc, nyf)
        Psi = Psi.reshape(mxc, nxf, myc, nyf)
        return k, Psi
    
    def psi_function(self, X, Y):
        if self.pn == 1:
            self.eps = 1/48
            I = np.where(np.abs(SIN(2*PI*X/self.eps))<0.4)
        if self.pn == 2:
            self.eps = 1/48
            I0, = np.where(np.abs(SIN(PI*X/self.eps)) > 0.95)
            I1, = np.where(np.abs(SIN(PI*Y/self.eps)) > 0.85)
            I = np.union1d(I0, I1)
        elif self.pn == 3:
            self.eps = 1/30
            # I = np.where(np.abs(SIN(PI*(Y**2+X)/self.eps)*SIN(PI*(X**2+Y)/self.eps))<0.4)
            I = np.where(np.abs(SIN(PI*(Y*(1-Y)+X)/self.eps)*SIN(PI*(X*(X-1)+Y)/self.eps))<0.4)
        elif self.pn == 4:
            self.eps = 1/12
            I = np.where(np.abs(SIN(2*PI*(X*(0.5+Y))/self.eps)*SIN(2*PI*((1+Y)*(1/3-X)/self.eps)))<0.35)
        elif self.pn == 5:
            self.eps = 1/8
            I = np.where(np.abs(SIN(2*PI*X/self.eps))<0.4)
        if self.pn == 10:
            self.eps = 1/40
            I, = np.where(np.abs(SIN(PI*((1-Y)*Y+X)/self.eps)*SIN(PI*(X**2+Y)/self.eps))<0.35)
        return I
    
    def reference_solution(self, k, f):
        mesh = RectangleMesh(self.Omega, self.NX*4, self.NY*4)
    
        NN = (self.NX*4+1) * (self.NY*4+1)
        A = mesh.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k.flat, A)
        A = A.reshape(self.NX, self.NY, 4, 4)
        A = np.broadcast_to(A[:,None,:,None,:,:], (self.NX,4,self.NY,4,4,4))
        A = A.reshape(-1, 4, 4)
        c2d = mesh.cell_to_dof(self.NX*4, self.NY*4)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        b = np.broadcast_to(f[:,None,:,None], (self.NX,4,self.NY,4))
        b = np.einsum('l, i -> li', b.flat, np.full(4, mesh.cellm*0.25))
        F = np.zeros(NN)
        np.add.at(F, c2d, b)
        u = mesh.set_zero_dirichlet(A, F)
        np.save(self.ad+'u.npy', u)
        return u[c2d] # (NC, 4)
    
    
    
