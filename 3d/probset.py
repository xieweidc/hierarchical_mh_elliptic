import numpy as np
from scipy.sparse import csr_matrix

from mesh import HexahedronMesh


PI = np.pi
SIN = np.sin
EXP = np.exp

class ProbSetup:
    def __init__(self, pn, m):
        self.pn = pn
        self.m = m
        # self.m = 1
        
        self.Omega = np.array([0, 1, 0, 1, 0, 1])
        if pn == 1:
            self.nxc = self.nyc = self.nzc = 8
            self.nxp = self.nyp = self.nzp = 3            
        elif pn == 2:
            # self.nxc = self.nyc = self.nzc = 2
            # self.nxp = self.nyp = self.nzp = 3
            # self.Omega = np.array([0, 2/self.nxc, 0, 2/self.nyc, 0, 2/self.nzc])
            
            self.nxc = self.nyc = self.nzc = 8
            self.nxp = self.nyp = self.nzp = 1
            self.Omega = np.array([0, 2/self.nxc, 0, 2/self.nyc, 0, 2/self.nzc])
            self.nxc = self.nyc = self.nzc = 2
        elif pn == 3:
            self.nxc = self.nyc = self.nzc = 2
            self.nxp = self.nyp = self.nzp = 1
        elif pn == 4:
            self.nxc = self.nyc = self.nzc = 10
            self.nxp = self.nyp = self.nzp = 1
        elif pn == 5:
            self.nxc = self.nyc = self.nzc = 12
            self.nxp = self.nyp = self.nzp = 1           
        elif pn == 6:
            self.nxc = self.nyc = self.nzc = 8
            self.nxp = self.nyp = self.nzp = 3
                        
        self.nxf = self.nyf = self.nzf = 6
        self.NX = self.NY = self.NZ = self.nxc * self.nxp * self.nxf
        
        self.mxc = self.nxc*self.nxp + 2*self.m
        self.myc = self.nxc*self.nyp + 2*self.m
        self.mzc = self.nzc*self.nzp + 2*self.m

        self.ad = 'res/ex%d_'%pn
        
    def average_u(self, u, Psi):
        u = u.reshape(self.nxc, self.nxp*self.nxf, 2, 
                      self.nyc, self.nyp*self.nyf, 2, 
                      self.nzc, self.nzp*self.nzf, 2, 8)
        J = Psi.astype(np.float64)
        J = np.stack((1-J, J))
        J = J.reshape(2, self.nxc, self.nxp*self.nxf,
                      self.nyc, self.nyp*self.nyf, self.nzc, self.nzp*self.nzf)
        I = np.sum(J, axis=(2, 4, 6)) # (2, NXC, NYC, NZC)
        ua = np.einsum('imrjnskpql, timjnkp, tijk -> tijk', u, J, 1/I/64)
        return ua # (2, NXC, NYC, NZC)
        
    def generate_kfI(self):
        """
        

        Returns
        -------
        k : TYPE
            (NX, NY, NZ).
        Psi : TYPE
            (NX, NY, NZ).
        f : TYPE
            (NX, NY, NZ).

        """
        
        hx = (self.Omega[1] - self.Omega[0]) / self.NX / 2
        hy = (self.Omega[3] - self.Omega[2]) / self.NY / 2
        hz = (self.Omega[5] - self.Omega[4]) / self.NZ / 2

        X = np.mgrid[self.Omega[0]:self.Omega[1]:complex(0,self.NX+1), 
                     self.Omega[2]:self.Omega[3]:complex(0,self.NY+1),
                     self.Omega[4]:self.Omega[5]:complex(0,self.NZ+1)]
        X, Y, Z = X
        X = X[1:, 1:, 1:] - hx
        Y = Y[1:, 1:, 1:] - hy
        Z = Z[1:, 1:, 1:] - hz
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        # k = np.ones((self.NX, self.NY, self.NZ))
        
        eta = 1e1
        # k1 = np.full((self.NX, self.NY, self.NZ), 1/eta)
        # k1 = (2 + SIN(PI*X)*SIN(PI*Y)*SIN(PI*Z)) / eta
        k1 = EXP(X/2) * EXP(Y/2) * EXP(Z/2) / eta
        k1 = k1.reshape(self.NX, self.NY, self.NZ)
        k = k1.copy()
        
        k[2::6] = k1[2::6] * eta
        k[3::6] = k1[3::6] * eta
        k[:, 2::6] = k1[:, 2::6] * eta
        k[:, 3::6] = k1[:, 3::6] * eta

        # k[2::6] = eta
        # k[3::6] = eta
        # k[:, 2::6] = eta
        # k[:, 3::6] = eta
        
        # f = np.ones((self.NX, self.NY, self.NZ))
        # f = np.zeros((self.NX, self.NY, self.NZ))
        # f[54:72, 54:72, 54:72] = 1
        k1 = EXP(-40*((X-0.5)**2 + (Y-0.5)**2) + (Z-0.5)**2) / eta
        k1 = k1.reshape(self.NX, self.NY, self.NZ)
        f = k1.copy()
        f[2::6] = k1[2::6] * eta
        f[3::6] = k1[3::6] * eta
        f[:, 2::6] = k1[:, 2::6] * eta
        f[:, 3::6] = k1[:, 3::6] * eta

        Psi = np.zeros((self.NX, self.NY, self.NZ), dtype=np.int_)
        Psi[2::6] = 1
        Psi[3::6] = 1
        Psi[:, 2::6] = 1
        Psi[:, 3::6] = 1
        return k, Psi, f
        
    def generate_kI_extend(self):
        """
        

        Returns
        -------
        k : TYPE
            (MXC, NXF, NYC, MYF, NZC, MZF).
        Psi : TYPE
            (MXC, NXF, NYC, MYF, NZC, MZF).

        """
        NX = self.mxc * self.nxf
        NY = self.myc * self.nyf
        NZ = self.mzc * self.nzf
        
        hx = (self.Omega[1] - self.Omega[0]) / self.NX / 2
        hy = (self.Omega[3] - self.Omega[2]) / self.NY / 2
        hz = (self.Omega[5] - self.Omega[4]) / self.NZ / 2

        hxc = (self.Omega[1] - self.Omega[0]) / self.nxc
        hyc = (self.Omega[3] - self.Omega[2]) / self.nyc
        hzc = (self.Omega[5] - self.Omega[4]) / self.nzc
        w = np.array([self.Omega[0]-self.m*hxc, self.Omega[1]+self.m*hxc,
                      self.Omega[2]-self.m*hyc, self.Omega[3]+self.m*hyc,
                      self.Omega[4]-self.m*hzc, self.Omega[5]+self.m*hzc])
        X = np.mgrid[w[0]:w[1]:complex(0,NX+1), 
                     w[2]:w[3]:complex(0,NY+1),
                     w[4]:w[5]:complex(0,NZ+1)]
        X, Y, Z = X
        X = X[1:, 1:, 1:] - hx
        Y = Y[1:, 1:, 1:] - hy
        Z = Z[1:, 1:, 1:] - hz
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        # k = np.ones((NX, NY, NZ))
        
        eta = 1e1
        # k1 = (2 + SIN(PI*X)*SIN(PI*Y)*SIN(PI*Z)) / eta
        k1 = EXP(X/2) * EXP(Y/2) * EXP(Z/2) / eta
        k1 = k1.reshape(NX, NY, NZ)
        k = k1.copy()
        
        k[2::6] = k1[2::6] * eta
        k[3::6] = k1[3::6] * eta
        k[:, 2::6] = k1[:, 2::6] * eta
        k[:, 3::6] = k1[:, 3::6] * eta

        Psi = np.zeros((NX, NY, NZ), dtype=np.int_)        
        Psi[2::6] = 1
        Psi[3::6] = 1
        Psi[:, 2::6] = 1
        Psi[:, 3::6] = 1
        
        k = k.reshape(self.mxc, self.nxf, self.myc, self.nyf, self.mzc, self.nzf)
        Psi = Psi.reshape(self.mxc, self.nxf, self.myc, self.nyf, self.mzc, self.nzf)
        return k, Psi
    
    def reference_solution(self, k, f):
        mesh = HexahedronMesh(self.Omega, self.NX*2, self.NY*2, self.NZ*2)
        I = np.broadcast_to(k[:,None,:,None,:,None], (self.NX,2,self.NY,2,self.NZ,2))
        mesh.celldata['k'] = I.flatten()
        I = np.broadcast_to(f[:,None,:,None,:,None], (self.NX,2,self.NY,2,self.NZ,2))
        mesh.celldata['f'] = I.flatten()
    
        NN = (self.NX*2+1) * (self.NY*2+1) * (self.NZ*2+1)
        A = mesh.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k.flat, A)
        A = A.reshape(self.NX, self.NY, self.NZ, 8, 8)
        A = np.broadcast_to(A[:,None,:,None,:,None,:,:], (self.NX,2,self.NY,2,self.NZ,2,8,8))
        A = A.reshape(-1, 8, 8)
        
        c2d = mesh.cell_to_dof()
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))
        
        b = np.broadcast_to(f[:,None,:,None,:,None], (self.NX,2,self.NY,2,self.NZ,2))
        b = np.einsum('l, i -> li', b.flat, np.full(8, mesh.cellm/8))
        F = np.zeros(NN)
        np.add.at(F, c2d, b)
        u = mesh.set_zero_dirichlet_bc(A, F)
        mesh.nodedata['u'] = u
        mesh.write_to_vtk('ref.vtu')
        # return u
        return u[c2d] # (NC, 8)
    
        
        
        
    