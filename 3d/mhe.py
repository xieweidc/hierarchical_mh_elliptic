import numpy as np
from copy import deepcopy
import multiprocessing as mp
from scipy.sparse import csr_matrix

from mesh import HexahedronMesh, MHMacroMesh


def phi_for_Rpm_mh(k, psi, f, w):
    """
    (2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
    """
    m2, nx, I0, ny, J0, nz = k.shape
    meshp = HexahedronMesh(w, m2*nx, m2*ny, m2*nz)
    
    # Constraint
    c2d = meshp.cell_to_dof()      # (NC, 8)
    I0 = psi.astype(np.float64)
    I0 = np.stack((1-I0, I0))      # (2, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
    J0 = np.sum(I0, axis=(2,4,6))  # (2, 2m+1, 2m+1, 2m+1)
    J0 = np.broadcast_to(J0[:,:,None,:,None,:,None], (2,m2,nx,m2,ny,m2,nz))
    J0 = J0.reshape(2, -1)
    NN = (m2*nx+1) * (m2*ny+1) * (m2*nz+1)
    S = np.full(c2d.shape, -0.125) # (NC, 8)
    I0, = np.where(psi.flat == 0)
    S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
    I0, = np.where(psi.flat == 1)
    S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])
    J1 = np.arange(NN, NN+m2**3).reshape(m2, m2, m2)
    J1 = np.broadcast_to(J1[:,None,:,None,:,None], (m2,nx,m2,ny,m2,nz))
    J1 = J1.reshape(-1)
    J1[I0] += m2**3
    J1 = np.broadcast_to(J1[:,None], S.shape)
    
    # Stiff matrix
    A = meshp.cell_stiff_matrix_varphi()
    A = np.einsum('l, ij -> lij', k.flat, A)
    I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
    J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
    
    A = np.r_[A.flat, S.flat, S.flat]
    I = np.r_[I.flat, c2d.flat, J1.flat]
    J = np.r_[J.flat, J1.flat, c2d.flat]
    NN += 2*m2**3
    A = csr_matrix((A, (I, J)), shape=(NN, NN))
    
    # Right hand side
    S = np.zeros((A.shape[0], 8))
    S[-2*m2**3:-m2**3, 0] = -1
    S[-m2**3:, 1] = -1
    
    NC = c2d.shape[0]
    I = np.sum(psi==0, axis=(1,3,5)) # (m2, m2, m2)
    J0 = np.stack((I, nx*ny*nz-I))   # (2, m2, m2, m2)
    I = np.mgrid[w[0]:w[1]:complex(0,m2*nx+1),
                 w[2]:w[3]:complex(0,m2*ny+1),
                 w[4]:w[5]:complex(0,m2*nz+1)] # (GD, NX+1, NY+1, NZ+1)
    I = I.reshape(3, -1)
    I = I[:, c2d] # (GD, NC, LDF)
    I = I.reshape(3, m2, nx, m2, ny, m2, nz, 8)
    J = np.c_[np.full(NC, 0.125), np.zeros(NC)] # (NC, 2)
    J[I0, 0] = 0
    J[I0, 1] = 0.125
    J = J.reshape(m2, nx, m2, ny, m2, nz, 2)
    I = np.einsum('tmpnqrsl, mpnqrsi, imnr -> mnrit', I, J, 1/J0) # (m2,m2,m2,2,GD)
    I = I.reshape(m2**3, 6)
    I -= I[I.shape[0]//2]
    
    S[-2*m2**3:-m2**3, 2:5] = -I[:, :3]
    S[-m2**3:, 5:] = -I[:, 3:]
    
    # Boudary condition
    phi = meshp.set_zero_dirichlet_bc(A, S)
    phi = phi[:-2*m2**3]
    
    # Macroscopic coefficient
    # Constraint to RVE
    S = m2 // 2
    I = np.arange(NN-2*m2**3)
    I = I.reshape(m2*nx+1, m2*ny+1, m2*nz+1)
    I = I[S*nx:(S+1)*nx+1, S*ny:(S+1)*ny+1, S*nz:(S+1)*nz+1]
    phi = phi[I.flat] # (NN, 8)
    
    c2d = meshp.cell_to_dof_para(nx, ny, nz)
    phi = phi[c2d] # (NC, LDF, 8)
    A = meshp.cell_stiff_matrix_varphi()
    Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, k[S,:,S,:,S].flat)
    
    bs = np.einsum('lmi, l, m -> i', phi[:,:,:2], f.flat, np.full(8, meshp.cellm/8))
    return Bs[:2,:2], Bs[2:,2:], bs

def phi_for_Rpm_mh_single(k, psi, f, w):
    """
    (2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
    """
    m2, nx, I0, ny, J0, nz = k.shape
    meshp = HexahedronMesh(w, m2*nx, m2*ny, m2*nz)
    
    # Constraint
    c2d = meshp.cell_to_dof()      # (NC, 8)
    I0 = psi.astype(np.float64)
    I0 = np.stack((1-I0, I0))      # (2, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
    J0 = np.sum(I0, axis=(2,4,6))  # (2, 2m+1, 2m+1, 2m+1)
    J0 = np.broadcast_to(J0[:,:,None,:,None,:,None], (2,m2,nx,m2,ny,m2,nz))
    J0 = J0.reshape(2, -1)
    NN = (m2*nx+1) * (m2*ny+1) * (m2*nz+1)
    S = np.full(c2d.shape, -0.125) # (NC, 8)
    I0, = np.where(psi.flat == 0)
    S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
    I0, = np.where(psi.flat == 1)
    S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])
    J1 = np.arange(NN, NN+m2**3).reshape(m2, m2, m2)
    J1 = np.broadcast_to(J1[:,None,:,None,:,None], (m2,nx,m2,ny,m2,nz))
    J1 = J1.reshape(-1)
    J1[I0] += m2**3
    J1 = np.broadcast_to(J1[:,None], S.shape)
    
    # Stiff matrix
    A = meshp.cell_stiff_matrix_varphi()
    A = np.einsum('l, ij -> lij', k.flat, A)
    I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
    J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
    
    A = np.r_[A.flat, S.flat, S.flat]
    I = np.r_[I.flat, c2d.flat, J1.flat]
    J = np.r_[J.flat, J1.flat, c2d.flat]
    NN += 2*m2**3
    A = csr_matrix((A, (I, J)), shape=(NN, NN))
    
    # Right hand side
    S = np.zeros((A.shape[0], 8))
    S[-2*m2**3:-m2**3, 0] = -1
    S[-m2**3:, 1] = -1
    
    NC = c2d.shape[0]
    I = np.sum(psi==0, axis=(1,3,5)) # (m2, m2, m2)
    J0 = np.stack((I, nx*ny*nz-I))   # (2, m2, m2, m2)
    I = np.mgrid[w[0]:w[1]:complex(0,m2*nx+1),
                 w[2]:w[3]:complex(0,m2*ny+1),
                 w[4]:w[5]:complex(0,m2*nz+1)] # (GD, NX+1, NY+1, NZ+1)
    I = I.reshape(3, -1)
    I = I[:, c2d] # (GD, NC, LDF)
    I = I.reshape(3, m2, nx, m2, ny, m2, nz, 8)
    J = np.c_[np.full(NC, 0.125), np.zeros(NC)] # (NC, 2)
    J[I0, 0] = 0
    J[I0, 1] = 0.125
    J = J.reshape(m2, nx, m2, ny, m2, nz, 2)
    I = np.einsum('tmpnqrsl, mpnqrsi, imnr -> mnrit', I, J, 1/J0) # (m2,m2,m2,2,GD)
    I = I.reshape(m2**3, 6)
    I -= I[I.shape[0]//2]
    
    S[-2*m2**3:-m2**3, 2:5] = -I[:, :3]
    S[-m2**3:, 5:] = -I[:, 3:]
    
    # Boudary condition
    phi = meshp.set_zero_dirichlet_bc(A, S)
    phi = phi[:-2*m2**3]
    
    # Macroscopic coefficient
    # Constraint to RVE
    S = m2 // 2
    I = np.arange(NN-2*m2**3)
    I = I.reshape(m2*nx+1, m2*ny+1, m2*nz+1)
    I = I[S*nx:(S+1)*nx+1, S*ny:(S+1)*ny+1, S*nz:(S+1)*nz+1]
    phi = phi[I.flat] # (NN, 8)
    
    c2d = meshp.cell_to_dof_para(nx, ny, nz)
    phi = phi[c2d] # (NC, LDF, 8)
    A = meshp.cell_stiff_matrix_varphi()
    Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, k[S,:,S,:,S].flat)
    return Bs[:2,:2], Bs[2:,2:], phi

def Fs_for_Ki(phi, f, cellm):
    bs = np.einsum('lmi, l, m -> i', phi, f.flat, np.full(8, cellm/8))
    return bs


class MulticontinuumHomogenization:
    def __init__(self, ps):
        self.ps = ps
        self.Omega = ps.Omega
        self.nxc = ps.nxc
        self.nyc = ps.nyc
        self.nzc = ps.nzc
        self.nxp = ps.nxp
        self.nyp = ps.nyp
        self.nzp = ps.nzp
        self.nxf = ps.nxf * 2
        self.nyf = ps.nyf * 2
        self.nzf = ps.nzf * 2
        self.m = ps.m
        
        self.ncc = self.nxc * self.nyc * self.nzc
        self.ncp = self.nxp * self.nyp * self.nzp
        self.nnc = (self.nxc+1) * (self.nyc+1) * (self.nzc+1)
        
        self.hxc = (ps.Omega[1] - ps.Omega[0]) / self.nxc
        self.hyc = (ps.Omega[3] - ps.Omega[2]) / self.nyc
        self.hzc = (ps.Omega[5] - ps.Omega[4]) / self.nzc
        self.cellmf = self.hxc*self.hyc*self.hzc / ps.nxf/ps.nyf/ps.nzf / 8
        
        self.m = ps.m
            
    def broadcast(self, a, I=1):
        
        nx, nxf, ny, nyf, nz, nzf = a.shape
        
        a = np.broadcast_to(a[:,:,None,:,:,None,:,:,None], (nx,nxf,2,ny,nyf,2,nz,nzf,2))
        a = a.reshape(nx, nxf*2, ny, nyf*2, nz, nzf*2)
        
        a = a.transpose(0, 2, 4, 1, 3, 5)
        a = a.reshape(-1, nxf*2, nyf*2, nzf*2)
        
        if I == 0:
            I = np.arange(self.ncc*self.ncp)
            I = I.reshape(self.nxc, self.nxp, self.nyc, self.nyp, self.nzc, self.nzp)
            I = I[:, self.nxp//2, :, self.nyp//2, :, self.nzp//2]
            I = I.reshape(-1)
            a = a[I] # (NCC, NXF, NYF, NZF)
        else:
            I = np.arange(self.ps.mxc*self.ps.myc*self.ps.mzc)
            I = I.reshape(self.ps.mxc, self.ps.myc, self.ps.mzc)
            J = I[:2*self.m+1, :2*self.m+1, :2*self.m+1]
            I = I[self.m:-self.m, self.m:-self.m, self.m:-self.m]
            I = I.reshape(self.nxc, self.nxp, self.nyc, self.nyp, self.nzc, self.nzp)
            I = I[:, self.nxp//2, :, self.nyp//2, :, self.nzp//2]
            I = I.reshape(-1)
            I -= I[0]
            I = I[:, np.newaxis] + J.reshape(-1)
            a = a[I] # (NCC, (2M+1)^3, NXF, NYF, NZF)
            a = a.reshape(self.ncc, 2*self.m+1, 2*self.m+1, 2*self.m+1, self.nxf, self.nyf, self.nzf)
            a = a.transpose(0, 1, 4, 2, 5, 3, 6) # (NCC, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
        return a
        
    def solve(self, k, Psi, f, way='t'):
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = f.reshape(self.nxc*self.nxp, self.nxf//2, self.nyc*self.nyp, 
                      self.nyf//2, self.nzc*self.nzp, self.nzf//2)
        f = self.broadcast(f, 0)
        
        Rpp = np.array([0, (2*self.m+1)*self.hxc/self.nxp, 
                        0, (2*self.m+1)*self.hyc/self.nyp,
                        0, (2*self.m+1)*self.hzc/self.nzp])
        
        ### Non-Periodic case
        if way == 'm':
            nc = mp.cpu_count() - 2
            pool = mp.Pool(processes=nc)
            Rpp = np.broadcast_to(Rpp[None,:], (self.ncc,6))
            args = list(zip(k, Psi, f, Rpp))
            res = pool.starmap(phi_for_Rpm_mh, args)
            Rs, Ds, Fs = map(list, zip(*res))
            Rs = np.array(Rs)
            Ds = np.array(Ds)
            Fs = np.array(Fs)
        else:
            Rs = np.zeros((self.ncc, 2, 2))
            Ds = np.zeros((self.ncc, 6, 6))
            Fs = np.zeros((self.ncc, 2))
            for i in range(self.ncc):
                Rs[i], Ds[i], Fs[i] = phi_for_Rpm_mh(k[i], Psi[i], f[i], Rpp)
        
        # ### Periodic case
        # Rs, Ds, phi = phi_for_Rpm_mh_single(k[0], Psi[0], f[0], Rpp)
        # Fs = np.zeros((self.ncc, 2))
        # for i in range(self.ncc):
        #     Fs[i] = Fs_for_Ki(phi[:,:,:2], f[i], self.cellmf/self.ncp)
        # Rs = np.broadcast_to(Rs[None,:,:], (self.ncc, 2, 2))
        # Ds = np.broadcast_to(Ds[None,:,:], (self.ncc, 6, 6))
        
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc, self.nzc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa, Rs, Ds, Fs
        
