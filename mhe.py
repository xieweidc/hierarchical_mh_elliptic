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
    
    def phi_for_Kim(self, ith, k, psi, f):
        xth, yth = divmod(ith, self.nyc)
        # print("ith: %d, xth: %d, yth: %d"%(ith, xth, yth))
        # k, (2m+1, NXF, 2m+1, NYF)
        # psi, (2m+1, NXF, 2m+1, NYF)
        m2, nx, I, ny = k.shape
        m = m2 // 2
        
        w = np.zeros(4)
        w[0] = (xth-m)*self.hxc + self.Omega[0]
        w[1] = w[0] + self.hxc*m2
        w[2] = (yth-m)*self.hyc + self.Omega[2]
        w[3] = w[2] + self.hyc*m2
        mesh = RectangleMesh(w, m2*nx, m2*ny)
        
        # Constraint
        c2d = mesh.cell_to_dof(m2*nx, m2*ny)
        I0 = psi.astype(np.float64)
        I0 = np.stack((1-I0, I0)) # (2, 2m+1, NXF, 2m+1, NYF)
        J0 = np.sum(I0, axis=(2,4)) # (2, 2m+1, 2m+1)
        J0 = np.broadcast_to(J0[:,:,None,:,None], (2,m2,nx,m2,ny))
        J0 = J0.reshape(2, -1)
        NN = (m2*nx+1) * (m2*ny+1)
        S = np.full(c2d.shape, -0.25) # (NC, 4)
        I0, = np.where(psi.flat == 0)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
        I0, = np.where(psi.flat == 1)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])
        
        # if ith == 0:
        #     print("np.min(S): ", np.min(np.abs(S)))
        #     print("np.max(S): ", np.max(np.abs(S)))
        
        # if ith == 0:
        #     # print("I0: ", I0)
        #     ks = np.zeros_like(k)
        #     ks = ks.reshape(-1)
        #     ks[I0] = 1
        #     ks = ks.reshape(m2*nx, m2*ny)
        #     np.save('ks.npy', ks)
        #     plt.imshow(ks.T, extent=w, origin='lower', cmap='viridis')
        #     colorbar = plt.colorbar()
        #     plt.title(r'$\kappa_{01}$')
        #     plt.show()
        
        J1 = np.arange(NN, NN+m2**2).reshape(m2, m2)
        J1 = np.broadcast_to(J1[:, None, :, None], (m2, nx, m2, ny))
        J1 = J1.reshape(-1)
        J1[I0] += m2**2
        J1 = np.broadcast_to(J1[:, None], S.shape)        
        # Stiff matrix
        A = mesh.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        # if ith == 0:
        #     M = csr_matrix((A.flat, (I.flat, J.flat)), (NN, NN))
        
        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*m2**2
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
        
        # if ith == 0:
        #     print("A.shape: ", A.shape)
        
        # if ith == 0:
        #     er = np.max(np.abs(A[:-2*m2**2, :-2*m2**2]-M))
        #     print("Stiff : ", er)
        #     print("Constraint: ", np.unique(A[:-2*m2**2, -2*m2**2:].A))
        #     print(A[:-2*m2**2, -2*m2**2:])
        #     np.save('C.npy', A[:-2*m2**2, -2*m2**2:].A)
        #     print("A.cond: ", np.linalg.cond(A.A))

        # Right hand side
        S = np.zeros((A.shape[0], 6))
        I = np.sum(psi==0, axis=(1, 3)) # (m2, m2)
        J0 = np.stack((I, nx*ny-I))     # (m2, m2)
        S[-2*m2**2:-m2**2, 0] = -1
        S[-m2**2:, 1] = -1
        # if ith == 0:
        #     print("mesh.cellm: ", mesh.cellm)
        NC = c2d.shape[0]
        I = np.mgrid[w[0]:w[1]:complex(0,m2*nx+1),
                     w[2]:w[3]:complex(0,m2*ny+1)] # (GD, NX+1, NY+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, LDF)
        I = I.reshape(2, m2, nx, m2, ny, 4)
        J = np.c_[np.full(NC, 0.25), np.zeros(NC)] # (NC, 2)
        # if ith == 0:
        #     print("I0: ", I0)
        #     print("I0.shape", I0.shape)
        J[I0, 0] = 0
        J[I0, 1] = 0.25
        J = J.reshape(m2, nx, m2, ny, 2)
        I = np.einsum('tmpnql, mpnqi, imn -> mnit', I, J, 1/J0) # (m2, m2, 2, GD)
        I = I.reshape(m2**2, 4)
        # if ith == 0:
        #     print(I)
        I -= I[I.shape[0]//2]
        S[-2*m2**2:-m2**2, 2] = -I[:, 0]
        S[-2*m2**2:-m2**2, 3] = -I[:, 1]
        S[-m2**2:, 4] = -I[:, 2]
        S[-m2**2:, 5] = -I[:, 3]
        
        # if ith == 0:
        #     print("S top: ", np.max(np.abs(S[:-2*m2**2])))
        
        # if ith == 0:
        #     print(I)
        #     print("S top: ", np.max(np.abs(S[:-2*m2**2])))
        #     print("S bottom: ", S[-2*m2**2:])
            
        #     print("BC A.cond: ", np.linalg.cond(A.A))
        # Boudary condition
        phi = mesh.set_zero_dirichlet(A, S)
        # if ith == 0:
        #     print(np.max(np.abs(phi[-2*m2**2:]), axis=0))
        
        # if ith == 0:
        #     for jth in range(6):
        #         a = phi[:-2*m2**2, jth]
        #         a = a.reshape(m2*nx+1, m2*ny+1)
        #         plt.imshow(a.T, extent=w, origin='lower', cmap='viridis')
        #         colorbar = plt.colorbar()
        #         plt.xlabel(r'$x_1$')
        #         plt.ylabel(r'$x_2$')
        #         plt.title(r'$\phi_{%d}$'%jth)
        #         plt.show()

        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh.ada(m*nx*(m2*ny+1)+m*ny, nx+1, ny+1, m2*ny+1)      
        
        # if ith == 0:
        #     varphi = phi.copy()
        #     # varphi = np.zeros_like(phi)
        #     print("I.shape: ", I.shape)
        #     varphi[I] = 1e4
        #     for jth in range(6):
        #         a = varphi[:-2*m2**2, jth]
        #         # print("np.max(varphi): ", np.max(np.abs(a)))
        #         a = a.reshape(m2*nx+1, m2*ny+1)
        #         plt.imshow(a.T, extent=w, origin='lower', cmap='viridis')
        #         colorbar = plt.colorbar()
        #         plt.xlabel(r'$x_1$')
        #         plt.ylabel(r'$x_2$')
        #         plt.title(r'$\psi_{%d}$'%jth)
        #         plt.show()
        
        phi = phi[I] # (NN, 6)
        
        # if ith == 0:
        #     ks = k.reshape(m2*nx, m2*ny)
        #     plt.imshow(ks.T, extent=w, origin='lower', cmap='viridis')
        #     colorbar = plt.colorbar()
        #     plt.xlabel(r'$x_1$')
        #     plt.ylabel(r'$x_2$')
        #     plt.title(r'$\kappa_0$')
        #     plt.show()
        #     # print(ks[0])
        #     for jth in range(6):
        #         a = phi[:, jth]
        #         a = a.reshape(nx+1, ny+1)
        #         plt.imshow(a.T, extent=w, origin='lower', cmap='viridis')
        #         colorbar = plt.colorbar()
        #         plt.title(r'$\varphi_{%d}$'%jth)
        #         plt.xlabel(r'$x_1$')
        #         plt.ylabel(r'$x_2$')
        #         plt.show()
        
        c2d = mesh.cell_to_dof(nx, ny)
        phi = phi[c2d] # (NC, 4, 6)
        A = mesh.cell_stiff_matrix_varphi()
        Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, k[m,:,m].flat)
        # if ith == 0:
        #     print("A: \n", A)
        #     print("Bs[:2,:2]: \n", Bs[:2,:2])
        #     print("Bs[2:,2:]: \n", Bs[2:,2:])
        
        # if ith == 0:
        #     ks = k[m, :, m]
        #     # print(ks[0])
        #     plt.imshow(ks.T, extent=w, origin='lower', cmap='viridis')
        #     colorbar = plt.colorbar()
        #     plt.xlabel(r'$x_1$')
        #     plt.ylabel(r'$x_2$')
        #     plt.title(r'$\kappa_0$')
        bs = np.einsum('lmi, l, m -> i', phi[:,:,:2], f.flat, np.full(4, mesh.cellm*0.25))
        
        # I = np.max(np.abs(Bs[:2, 2:]))
        # if I > 1e-7:
        #     print(r"$K_{%d}$ advection coefficient is "%ith, I)
        return Bs[:2,:2], Bs[2:,2:], bs
    
    def phi_for_Kim_single(self, k, psi):
        # k, (2m+1, NXF, 2m+1, NYF)
        # psi, (2m+1, NXF, 2m+1, NYF)
        m2, nx, I, ny = k.shape
        m = m2 // 2
        
        w = np.zeros(4)
        w[0] = -m*self.hxc + self.Omega[0]
        w[1] = w[0] + self.hxc*m2
        w[2] = -m*self.hyc + self.Omega[2]
        w[3] = w[2] + self.hyc*m2
        mesh = RectangleMesh(w, m2*nx, m2*ny)
        
        # Constraint
        c2d = mesh.cell_to_dof(m2*nx, m2*ny)
        I0 = psi.astype(np.float64)
        I0 = np.stack((1-I0, I0)) # (2, 2m+1, NXF, 2m+1, NYF)
        J0 = np.sum(I0, axis=(2,4)) # (2, 2m+1, 2m+1)
        J0 = np.broadcast_to(J0[:,:,None,:,None], (2,m2,nx,m2,ny))
        J0 = J0.reshape(2, -1)
        NN = (m2*nx+1) * (m2*ny+1)
        S = np.full(c2d.shape, -0.25) # (NC, 4)
        I0, = np.where(psi.flat == 0)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
        I0, = np.where(psi.flat == 1)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])
        
        J1 = np.arange(NN, NN+m2**2).reshape(m2, m2)
        J1 = np.broadcast_to(J1[:, None, :, None], (m2, nx, m2, ny))
        J1 = J1.reshape(-1)
        J1[I0] += m2**2
        J1 = np.broadcast_to(J1[:, None], S.shape)        
        # Stiff matrix
        A = mesh.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        
        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*m2**2
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
                    
        # Right hand side
        S = np.zeros((A.shape[0], 6))
        I = np.sum(psi==0, axis=(1, 3)) # (m2, m2)
        J0 = np.stack((I.flat, (nx*ny-I).flat))
        S[-2*m2**2:-m2**2, 0] = -1
        S[-m2**2:, 1] = -1
        NC = c2d.shape[0]
        I = np.mgrid[w[0]:w[1]:complex(0,m2*nx+1),
                      w[2]:w[3]:complex(0,m2*ny+1)] # (GD, NX+1, NY+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, LDF)
        I = I.reshape(2, m2, nx, m2, ny, 4)
        J = np.c_[np.full(NC, 0.25), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 0.25
        J = J.reshape(m2, nx, m2, ny, 2)
        I = np.einsum('tmpnql, mpnqi -> mnit', I, J) # (m2, m2, 2, GD)
        I = I.reshape(m2**2, 2, 2)
        I = np.einsum('lmn, ml -> lmn', I, 1/J0)
        I = I.reshape(m2**2, 4)
        I -= I[I.shape[0]//2]
        S[-2*m2**2:-m2**2, 2] = -I[:, 0]
        S[-2*m2**2:-m2**2, 3] = -I[:, 1]
        S[-m2**2:, 4] = -I[:, 2]
        S[-m2**2:, 5] = -I[:, 3]
        
        # Boudary condition
        phi = mesh.set_zero_dirichlet(A, S)
        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh.ada(m*nx*(m2*ny+1)+m*ny, nx+1, ny+1, m2*ny+1)      
        
        phi = phi[I] # (NN, 6)
        c2d = mesh.cell_to_dof(nx, ny)
        phi = phi[c2d] # (NC, 4, 6)
        A = mesh.cell_stiff_matrix_varphi()
        Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, k[m,:,m].flat)
        print("Reaction: \n", Bs[:2,:2])
        print("Diffusion: \n", Bs[2:,2:])
        return Bs[:2,:2], Bs[2:,2:], phi[:,:,:2]

    def Fs_for_Ki(self, phi, f):
        
        cellm = self.hxc * self.hyc / self.nxf / self.nyf / 4
        bs = np.einsum('lmi, l, m -> i', phi, f.flat, np.full(4, cellm))
        return bs


class MulticontinuumHomogenization(MicroPhi):
    
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
        ix, iy = np.meshgrid(ix, iy)
        ix += iy
        return ix.flatten('F')
    
    def broadcast(self, a):
        nx, nxf, ny, nyf = a.shape
        
        a = np.broadcast_to(a[:,:,None,:,:,None], (nx,nxf,4,ny,nyf,4))
        a = a.reshape(nx, nxf*4, ny, nyf*4)
        
        # a = a.reshape(nx, nxf, ny, nyf)
        
        a = a.swapaxes(1, 2)
        a = a.reshape(-1, self.nxf, self.nyf)
        return a
    
    def solve(self, k, Psi, f, way='m'):
        
        # Need broadcast the coefficient.
        f = f.reshape(self.nxc, self.nxf//4, self.nyc, self.nyf//4)
        # f = f.reshape(self.nxc, self.nxf, self.nyc, self.nyf)
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = self.broadcast(f)
        
        I = np.arange((self.nxc+2*self.m)*(self.nyc+2*self.m)).reshape(self.nxc+2*self.m, self.nyc+2*self.m)
        I = I[:-2*self.m, :-2*self.m]
        I = I.reshape(-1)
        J = self.ada(0, 2*self.m+1, 2*self.m+1, self.nyc+2*self.m)
        I = I[:, np.newaxis] + J
        
        k = k[I] # (NCC, (2M+1)^2, NXF, NYF)
        k = k.reshape(self.ncc, 2*self.m+1, 2*self.m+1, self.nxf, self.nyf)
        k = k.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
        
        Psi = Psi[I] # (NCC, (2M+1)^2, NXF, NYF)
        Psi = Psi.reshape(self.ncc, 2*self.m+1, 2*self.m+1, self.nxf, self.nyf)
        Psi = Psi.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
        
        if way == 'm':
            pool = mp.Pool(processes=self.nc)
            local = deepcopy(self)
            args = list(zip(range(self.ncc), k, Psi, f))
            res = pool.starmap(local.phi_for_Kim, args)
            Rs, Ds, Fs = map(list, zip(*res))
        else:
            Rs = np.zeros((self.ncc, 2, 2))
            Ds = np.zeros((self.ncc, 4, 4))
            Fs = np.zeros((self.ncc, 2))
            for i in range(self.ncc):
                Rs[i], Ds[i], Fs[i] = self.phi_for_Kim(i, k[i], Psi[i], f[i])
        
        Ds = Ds.reshape(self.ncc, 2, 2, 2, 2)
        # return Rs, Ds, Fs
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa, Rs, Ds, Fs

    def solve_periodic(self, k, Psi, f):
        
        # Need broadcast the coefficient.
        # f = f.reshape(self.nxc, self.nxf//4, self.nyc, self.nyf//4)
        f = f.reshape(self.nxc, self.nxf, self.nyc, self.nyf)
        f = f.swapaxes(1, 2)
        f = f.reshape(-1, self.nxf, self.nyf)
        
        k = k[0, :, 0]
        k = np.broadcast_to(k[None, :, None, :], (2*self.m+1, self.nxf, 2*self.m+1, self.nyf))
        Psi = Psi[0, :, 0]
        Psi = np.broadcast_to(Psi[None, :, None, :], (2*self.m+1, self.nxf, 2*self.m+1, self.nyf))
        
        Rs, Ds, phi = self.phi_for_Kim_single(k, Psi)
        Fs = np.zeros((self.ncc, 2))
        for i in range(self.ncc):
            Fs[i] = self.Fs_for_Ki(phi, f[i])
        Rs = np.broadcast_to(Rs[None,:,:], (self.ncc,2,2))
        Ds = np.broadcast_to(Ds[None,:,:], (self.ncc,4,4))
        Ds = Ds.reshape(self.ncc, 2, 2, 2, 2)
        # return Rs, Ds, Fs
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa, Rs, Ds, Fs

    def solve_single(self, k, Psi, f, way='m'):
        
        # Need broadcast the coefficient.
        f = f.reshape(self.nxc, self.nxf//4, self.nyc, self.nyf//4)
        # f = f.reshape(self.nxc, self.nxf, self.nyc, self.nyf)
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = self.broadcast(f)
        
        I = np.arange((self.nxc+2*self.m)*(self.nyc+2*self.m)).reshape(self.nxc+2*self.m, self.nyc+2*self.m)
        I = I[:-2*self.m, :-2*self.m]
        I = I.reshape(-1)
        J = self.ada(0, 2*self.m+1, 2*self.m+1, self.nyc+2*self.m)
        I = I[:, np.newaxis] + J
        
        k = k[I] # (NCC, (2M+1)^2, NXF, NYF)
        k = k.reshape(self.ncc, 2*self.m+1, 2*self.m+1, self.nxf, self.nyf)
        k = k.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
        
        Psi = Psi[I] # (NCC, (2M+1)^2, NXF, NYF)
        Psi = Psi.reshape(self.ncc, 2*self.m+1, 2*self.m+1, self.nxf, self.nyf)
        Psi = Psi.swapaxes(2, 3) # (NCC, 2m+1, NXF, 2m+1, NYF)
        
        i = 0
        self.phi_for_Kim(i, k[i], Psi[i], f[i])

