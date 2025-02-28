import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve as scsolve
from pypardiso import spsolve as pysolve


class RectangleMesh:
    def __init__(self, w, nx, ny):
        self.w = w
        self.nx = nx
        self.ny = ny
        
        self.hx = (w[1]-w[0]) / nx
        self.hy = (w[3]-w[2]) / ny
        self.cellm = self.hx * self.hy
        
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

    def cell_to_dof(self, nx, ny):
        """
        
        Returns
        -------
        v : numpy array
            array, (nx*ny, 4).

        """
        v = np.arange((nx+1)*(ny+1)).reshape(nx+1, ny+1)
        v = v[:-1, :-1]
        v = v.reshape(-1)
        v = v[:, np.newaxis] + np.array([0, 1, ny+1, ny+2])
        return v
    
    def cell_mass_matrix_varphi(self):
        """
        \int_{w} varphi_j \varphi_i

        Returns
        -------
        Mw : ndarray
            Cell mass matrix on fine grid, (4, 4).

        """
        Mw = np.array([[4, 2, 2, 1],
                       [2, 4, 1, 2],
                       [2, 1, 4, 2],
                       [1, 2, 2, 4]]) * (self.cellm/36)
        return Mw
    
    def cell_stiff_matrix_varphi(self):
        """
        \int_{w} \nabla varphi_j \cdot \nabla \varphi_i

        Returns
        -------
        Aw : ndarray
            Cell stiff matrix on fine grid, (4, 4).

        """
        # \int_{w} \nabla_{x_1} varphi_j \cdot \nabla_{x_1} \varphi_i
        Aw = np.array([[ 2,  1, -2, -1],
                       [ 1,  2, -1, -2],
                       [-2, -1,  2,  1],
                       [-1, -2,  1,  2]]) * (self.hy/self.hx/6)
        # \int_{w} \nabla_{x_2} varphi_j \cdot \nabla_{x_2} \varphi_i
        Aw += np.array([[ 2, -2,  1, -1],
                        [-2,  2, -1,  1],
                        [ 1, -1,  2, -2],
                        [-1,  1, -2,  2]]) * (self.hx/self.hy/6)
        return Aw

    def is_boundary_dof(self):
        """
        

        Parameters
        ----------
        nx : int
            division on x axis.
        ny : int
            division on y axis.

        Returns
        -------
        ibd : ndarray
            number of boundary node on rectangle.

        """
        ibd = np.zeros(2*(self.nx+self.ny), dtype=np.int_)
        ibd[:self.ny+1] = np.arange(self.ny+1)
        ibd[-(self.ny+1):] = np.arange(self.nx*(self.ny+1), (self.nx+1)*(self.ny+1))
        ibd[self.ny+1:-(self.ny+2):2] = np.arange(self.ny+1, self.nx*(self.ny+1), self.ny+1)
        ibd[self.ny+2:-(self.ny+1):2] = ibd[self.ny+1:-(self.ny+2):2] + self.ny
        return ibd
        
    def set_zero_dirichlet(self, A, F):
        
        # # # print("A.det: ", np.linalg.det(A.A))
        # if F.ndim == 1:
        #     bdI = self.is_boundary_dof()
        #     F[bdI] = 0
        #     I = np.zeros(A.shape[0], dtype=np.int_)
        #     I[bdI] = 1
        #     Tbd = spdiags(I, 0, A.shape[0], A.shape[0])
        #     T = spdiags(1-I, 0, A.shape[0], A.shape[0])
        #     A = T@A@T + Tbd
        
        bdI = self.is_boundary_dof()
        F[bdI] = 0
        I = np.zeros(A.shape[0], dtype=np.int_)
        I[bdI] = 1
        Tbd = spdiags(I, 0, A.shape[0], A.shape[0])
        T = spdiags(1-I, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd

        x = pysolve(A, F)
        # x = scsolve(A, F)
        
        er = np.sum(np.abs(A@x - F)> 1e-10)
        if er > 0:
            print(self.w, "er is ", er, np.max(np.abs(A@x - F)))
        return x
    
    
class MacroMesh:
    def __init__(self, Omega, nx, ny):
        self.Omega = Omega
        self.nx = nx
        self.ny = ny
        self.nn = (nx+1) * (ny+1)
        self.hx = (Omega[1] - Omega[0]) / self.nx 
        self.hy = (Omega[3] - Omega[2]) / self.ny
        self.cellm = self.hx * self.hy
        
    def cell_to_dof(self):
        """
        
        Returns
        -------
        v : numpy array
            array, (nx*ny, 4).

        """
        v = np.arange((self.nx+1)*(self.ny+1)).reshape(self.nx+1, self.ny+1)
        v = v[:-1, :-1]
        v = v.reshape(-1)
        v = v[:, np.newaxis] + np.array([0, 1, self.ny+1, self.ny+2])
        return v
    
    def Ki_mass_matrix(self):
        """
        

        Returns
        -------
        Mw : csr_matrix
            FEM mass matrix, 
                \int_{\Omega} \varphi_j \varphi_i.

        """
        Mw = np.array([[4, 2, 2, 1],
                       [2, 4, 1, 2],
                       [2, 1, 4, 2],
                       [1, 2, 2, 4]]) * (self.cellm/36)
        return Mw
    
    def Ki_stiff_matrix(self):
        """
        \int_{w} \nabla varphi_j \varphi_i

        Returns
        -------
        Aw : ndarray
            Cell stiff matrix on fine grid, (2, 2, 4, 4).

        """
        Aw = np.array([[[[ 2,  1, -2, -1],
                         [ 1,  2, -1, -2],
                         [-2, -1,  2,  1],
                         [-1, -2,  1,  2]],
                        
                         [[ 1,  1, -1, -1],
                          [-1, -1,  1,  1],
                          [ 1,  1, -1, -1],
                          [-1, -1,  1,  1]]],
                        
                        [[[ 1, -1,  1, -1],
                          [ 1, -1,  1, -1],
                          [-1,  1, -1,  1],
                          [-1,  1, -1,  1]],
                       
                        [[ 2, -2,  1, -1],
                         [-2,  2, -1,  1],
                         [ 1, -1,  2, -2],
                         [-1,  1, -2,  2]]]], dtype=np.float64)
        
        Aw[0, 0] = Aw[0, 0] * (self.hy/self.hx/6)
        Aw[0, 1] = Aw[0, 1] / 4
        Aw[1, 0] = Aw[1, 0] / 4
        Aw[1, 1] = Aw[1, 1] * (self.hx/self.hy/6)
        return Aw
    
    def is_boundary_dof(self):
        """
        

        Parameters
        ----------
        nx : int
            division on x axis.
        ny : int
            division on y axis.

        Returns
        -------
        ibd : ndarray
            number of boundary node on rectangle.

        """
        ibd = np.zeros(2*(self.nx+self.ny), dtype=np.int_)
        ibd[:self.ny+1] = np.arange(self.ny+1)
        ibd[-(self.ny+1):] = np.arange(self.nx*(self.ny+1), (self.nx+1)*(self.ny+1))
        ibd[self.ny+1:-(self.ny+2):2] = np.arange(self.ny+1, self.nx*(self.ny+1), self.ny+1)
        ibd[self.ny+2:-(self.ny+1):2] = ibd[self.ny+1:-(self.ny+2):2] + self.ny
        return ibd


class MHMacroMesh(MacroMesh):
    
    def solve(self, Rs, Ds, Fs):
        
        I = self.Ki_mass_matrix()
        A = np.einsum('lmn, ij -> mnlij', Rs, I)
        I = self.Ki_stiff_matrix()
        I = np.einsum('lmgnp, gpij -> mnlij', Ds, I)
        A += I
        c2d = self.cell_to_dof()
        c2d = np.stack((c2d, c2d+self.nn)) # (2, NC, 4)
        I = np.broadcast_to(c2d[:, None, :, :, None], A.shape)
        J = np.broadcast_to(c2d[None, :, :, None, :], A.shape)
        A = csr_matrix((A.flat, (I.flat,J.flat)), (self.nn*2,self.nn*2))
        
        f = np.einsum('lm, i -> mli', Fs, np.full(4, self.cellm/4))
        F = np.zeros(2*self.nn)
        np.add.at(F, c2d, f)
        
        # Boundary condition
        bdI = self.is_boundary_dof()
        bdI = np.r_[bdI, bdI+self.nn]
        F[bdI] = 0
        I = np.zeros(A.shape[0], dtype=np.int_)
        I[bdI] = 1
        Tbd = spdiags(I, 0, A.shape[0], A.shape[0])
        T = spdiags(1-I, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd

        UH = scsolve(A, F)
        # UH = pysolve(A, F)
        
        UH = UH.reshape(2, self.nn)
        UHa = UH[:, c2d[0]] # (2, NC, 4)
        UHa = np.einsum('tli, i -> tl', UHa, np.full(4, 0.25))
        UHa = UHa.reshape(2, self.nx, self.ny)
        UH = UH.reshape(2, self.nx+1, self.ny+1)
        return UH, UHa


class NHMacroMesh(MacroMesh):
    
    def solve(self, Ds, f):
        
        A = self.Ki_stiff_matrix()
        A = np.einsum('lgp, gpij -> lij', Ds, A)
        c2d = self.cell_to_dof()
        I = np.broadcast_to(c2d[:, :, None], A.shape)
        J = np.broadcast_to(c2d[:, None, :], A.shape)
        A = csr_matrix((A.flat, (I.flat,J.flat)), (self.nn,self.nn))
        
        f = np.einsum('l, i -> li', f.flat, np.full(4, self.cellm/4))
        F = np.zeros(self.nn)
        np.add.at(F, c2d, f)
        
        # Boundary condition
        bdI = self.is_boundary_dof()
        F[bdI] = 0
        I = np.zeros(A.shape[0], dtype=np.int_)
        I[bdI] = 1
        Tbd = spdiags(I, 0, A.shape[0], A.shape[0])
        T = spdiags(1-I, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
                
        UH = scsolve(A, F)
        # UH = pysolve(A, F)
        
        UHa = UH[c2d] # (NC, 4)
        UHa = np.einsum('li, i -> l', UHa, np.full(4, 0.25))
        UHa = UHa.reshape(self.nx, self.ny)
        UH = UH.reshape(self.nx+1, self.ny+1)
        return UH, UHa


