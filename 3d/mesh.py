import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve as scsolve
from pypardiso import spsolve as pysolve

import vtk
import vtk.util.numpy_support as vnp


def write_to_vtk_Hexahedron(fname, nodes, cells, nodedata, celldata):
    """
    Write 3D hexahedral mesh data to a VTK file.

    Parameters:
    - fname: str, the output file name (e.g., 'output.vtu').
    - cells: numpy array of shape (NC, 8), where NC is the number of cells and 8 corresponds to the indices of the cell's nodes.
    - nodes: numpy array of shape (NN, 3), where NN is the number of nodes and 3 corresponds to the node coordinates (x, y, z).
    """
    nodes = nodes.swapaxes(0, 1)
    cells = cells[:, [0, 1, 3, 2, 4, 5, 7, 6]]
    NN = nodes.shape[0]  # Number of nodes
    NC = cells.shape[0]  # Number of cells
    NV = cells.shape[1]  # Number of vertices per cell (should be 8 for hexahedra)

    # Ensure that nodes have 3 components (x, y, z)
    if nodes.shape[1] == 2:
        # Add a zero z-component if only x and y are provided
        nodes = np.c_[nodes, np.zeros(NN)]
    elif nodes.shape[1] != 3:
        raise ValueError("Nodes array must have exactly 3 components per point (x, y, z).")

    # Prepare cells: insert the number of vertices at the start of each cell row
    cells = cells.astype(np.int64)
    cell = np.c_[np.full(NC, NV), cells]  

    # Setup VTK points
    points = vtk.vtkPoints()
    points.SetData(vnp.numpy_to_vtk(nodes, deep=True, array_type=vtk.VTK_FLOAT))

    # Setup VTK cells
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell, deep=True))

    # Create the VTK unstructured grid
    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(points)
    mesh.SetCells(vtk.VTK_HEXAHEDRON, vtk_cells)

    # Add point data if available
    pdata = mesh.GetPointData()

    if nodedata is not None:
        for key, val in nodedata.items():
            if val is not None:
                # Extend data to 3D if it's 2D
                if len(val.shape) == 2 and val.shape[1] == 2:
                    shape = (val.shape[0], 3)
                    val1 = np.zeros(shape, dtype=val.dtype)
                    val1[:, 0:2] = val
                else:
                    val1 = val
                    
                if len(val1) != NN:
                    raise ValueError("Node value is not fullfilled!")

                # Convert boolean arrays to integer type for VTK compatibility
                if val1.dtype == np.bool_:
                    d = vnp.numpy_to_vtk(val1.astype(np.int_), deep=True)
                else:
                    d = vnp.numpy_to_vtk(val1[:], deep=True)
                d.SetName(key)
                pdata.AddArray(d)

    # Add cell data if available
    if celldata is not None:
        cdata = mesh.GetCellData()
        for key, val in celldata.items():
            if val is not None:
                # Extend data to 3D if it's 2D
                if len(val.shape) == 2 and val.shape[1] == 2:
                    shape = (val.shape[0], 3)
                    val1 = np.zeros(shape, dtype=val.dtype)
                    val1[:, 0:2] = val
                else:
                    val1 = val

                if len(val1) != NC:
                    raise ValueError("Cell value is not fullfilled!")

                # Convert boolean arrays to integer type for VTK compatibility
                if val1.dtype == np.bool_:
                    d = vnp.numpy_to_vtk(val1.astype(np.int_), deep=True)
                else:
                    d = vnp.numpy_to_vtk(val1[:], deep=True)

                d.SetName(key)
                cdata.AddArray(d)

    # Write the mesh to a VTK file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(mesh)
    writer.Write()

    
class HexahedronMesh:
    def __init__(self, w, nx, ny, nz):
        self.w = w
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.hx = (w[1]-w[0]) / nx
        self.hy = (w[3]-w[2]) / ny
        self.hz = (w[5]-w[4]) / nz
        self.cellm = self.hx * self.hy * self.hz
        
        self.nodedata = {}
        self.celldata = {}
        
    def cell_mass_matrix_varphi(self):
        Mw = np.array([[8, 4, 4, 2, 4, 2, 2, 1],
                       [4, 8, 2, 4, 2, 4, 1, 2],
                       [4, 2, 8, 4, 2, 1, 4, 2],
                       [2, 4, 4, 8, 1, 2, 2, 4],
                       [4, 2, 2, 1, 8, 4, 4, 2],
                       [2, 4, 1, 2, 4, 8, 2, 4],
                       [2, 1, 4, 2, 4, 2, 8, 4],
                       [1, 2, 2, 4, 2, 4, 4, 8]]) * (self.cellmf/216)
        return Mw
        
    def cell_stiff_matrix_varphi(self):
        r"""
        

        Returns
        -------
        Aw : ndarray
            \int_w \nabla_{x_m} \varphi_i \nabla_{x_m} \varphi_j.

        """
        # \int_w \nabla_x1 \varphi \nabla_x1 \phi
        Aw = np.array([[4, 2, 2, 1, -4, -2, -2, -1],
                       [2, 4, 1, 2, -2, -4, -1, -2],
                       [2, 1, 4, 2, -2, -1, -4, -2],
                       [1, 2, 2, 4, -1, -2, -2, -4],
                       [-4, -2, -2, -1, 4, 2, 2, 1],
                       [-2, -4, -1, -2, 2, 4, 1, 2],
                       [-2, -1, -4, -2, 2, 1, 4, 2],
                       [-1, -2, -2, -4, 1, 2, 2, 4]
                       ])  * (self.hy*self.hz/self.hx/36)

        # \int_w \nabla_x2 \varphi \nabla_x2 \phi
        Aw += np.array([[4, 2, -4, -2, 2, 1, -2, -1],
                        [2, 4, -2, -4, 1, 2, -1, -2],
                        [-4, -2, 4, 2, -2, -1, 2, 1],
                        [-2, -4, 2, 4, -1, -2, 1, 2],
                        [2, 1, -2, -1, 4, 2, -4, -2],
                        [1, 2, -1, -2, 2, 4, -2, -4],
                        [-2, -1, 2, 1, -4, -2, 4, 2],
                        [-1, -2, 1, 2, -2, -4, 2, 4]
                        ])  * (self.hx*self.hz/self.hy/36)
        
        # \int_w \nabla_x3 \varphi \nabla_x3 \phi
        Aw += np.array([[4, -4, 2, -2, 2, -2, 1, -1],
                        [-4, 4, -2, 2, -2, 2, -1, 1],
                        [2, -2, 4, -4, 1, -1, 2, -2],
                        [-2, 2, -4, 4, -1, 1, -2, 2],
                        [2, -2, 1, -1, 4, -4, 2, -2],
                        [-2, 2, -1, 1, -4, 4, -2, 2],
                        [1, -1, 2, -2, 2, -2, 4, -4],
                        [-1, 1, -2, 2, -2, 2, -4, 4]
                        ])  * (self.hx*self.hy/self.hz/36)
        return Aw
    
    def cell_to_dof(self):
        """
            (nx*ny*nz, 8)
        """
        NN = (self.nx+1) * (self.ny+1) * (self.nz+1)
        NC = self.nx * self.ny * self.nz
        I = np.arange(NN).reshape(self.nx+1, self.ny+1, self.nz+1)
        J = I[:2, :2, :2]
        I = I[:-1, :-1, :-1]
        I = I.reshape(-1)
        I = I[:, np.newaxis] + J.reshape(-1)
        return I

    def cell_to_dof_para(self, nx, ny, nz):
        """
            (nx*ny*nz, 8)
        """
        NN = (nx+1) * (ny+1) * (nz+1)
        NC = nx * ny * nz
        I = np.arange(NN).reshape(nx+1, ny+1, nz+1)
        J = I[:2, :2, :2]
        I = I[:-1, :-1, :-1]
        I = I.reshape(-1)
        I = I[:, np.newaxis] + J.reshape(-1)
        return I

    def get_nodes_from_grid(self):
        """
            (GD, (NXG+1)*(NYG+1)*(NZG+1))
        """
        
        node = np.mgrid[self.w[0]:self.w[1]:complex(0,self.nx+1), 
                        self.w[2]:self.w[3]:complex(0,self.ny+1),
                        self.w[4]:self.w[5]:complex(0,self.nz+1)]
        
        node = node.reshape(3, -1)   
        return node

    def is_boundary_dof(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        Idx = np.array([(ny+2)*(nz+1), (ny+2)*(nz+1)+nz])
        Idx = np.arange(0, (ny-1)*(nz+1), nz+1) + Idx[:, np.newaxis]
        Idx = np.r_[np.arange((ny+1)*(nz+1), (ny+2)*(nz+1)), Idx.flat,
                    np.arange((2*ny+1)*(nz+1), (2*ny+2)*(nz+1))]
        Idx = np.arange(0,(nx-1)*(ny+1)*(nz+1),(ny+1)*(nz+1)) + Idx[:, np.newaxis]
        Idx = np.r_[np.arange((ny+1)*(nz+1)), Idx.flat, 
                    np.arange(nx*(ny+1)*(nz+1), (nx+1)*(ny+1)*(nz+1))]
        return Idx     
        
    def set_zero_dirichlet_bc(self, A, F):
        # Set the zero dirichelt boundary condition.
        bdidx = self.is_boundary_dof()
        F[bdidx] = 0
        I = np.zeros(A.shape[0], dtype=np.int_)
        I[bdidx] = 1
        Tbd = spdiags(I, 0, A.shape[0], A.shape[0])
        T = spdiags(1-I, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        I = pysolve(A, F)
        
        er = np.max(np.abs(A@I-F))
        if er > 1e-8:
            print("er : %.2e"%er)
        return I
    
    def write_to_vtk(self, fname):
        
        nodes = self.get_nodes_from_grid()
        cells = self.cell_to_dof()
        nodedata = self.nodedata
        celldata = self.celldata
        write_to_vtk_Hexahedron(fname, nodes, cells, nodedata, celldata)
  
        
class MHMacroMesh:
    def __init__(self, w, nx, ny, nz):
        self.w = w
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nn = (nx+1) * (ny+1) * (nz+1)
        
        self.hx = (w[1]-w[0]) / nx
        self.hy = (w[3]-w[2]) / ny
        self.hz = (w[5]-w[4]) / nz
        self.cellm = self.hx * self.hy * self.hz
        
        self.nodedata = {}
        self.celldata = {}
        
    def Ki_mass_matrix(self):
        Mw = np.array([[8, 4, 4, 2, 4, 2, 2, 1],
                       [4, 8, 2, 4, 2, 4, 1, 2],
                       [4, 2, 8, 4, 2, 1, 4, 2],
                       [2, 4, 4, 8, 1, 2, 2, 4],
                       [4, 2, 2, 1, 8, 4, 4, 2],
                       [2, 4, 1, 2, 4, 8, 2, 4],
                       [2, 1, 4, 2, 4, 2, 8, 4],
                       [1, 2, 2, 4, 2, 4, 4, 8]]) * (self.cellm/216)
        return Mw
        
    def Ki_stiff_matrix(self):
        r"""
        \int_{w} \nabla_m varphi_j \nabla_n \varphi_i

        Returns
        -------
        Aw : ndarray
            Cell stiff matrix on fine grid, (GD, GD, LDF, LDF).

        """
        q = 2
        qs, ws = np.polynomial.legendre.leggauss(q)
        qs = (qs + 1) / 2          # (q, )
        ws = ws / 2                # (q, )
        ws = np.einsum('i, j, k -> ijk', ws, ws, ws).flatten() # (NQ, )
                
        qs = np.stack((1-qs, qs)) # (2, NQ)
        
        # \nabla [\varphi_0, \cdots, \varphi_7]
        gvarphi = np.stack((np.einsum('i, mp, nq, r -> imnrpq', 
                            np.array([-1/self.hx, 1/self.hx]), qs, qs, np.ones(2)),
                           
                            np.einsum('ir, m, nq, p -> imnrpq',
                            qs, np.array([-1/self.hy, 1/self.hy]), qs, np.ones(2)),
                            
                            np.einsum('ir, mp, n, q -> imnrpq',
                            qs, qs, np.array([-1/self.hz, 1/self.hz]), np.ones(2))))
        gvarphi = gvarphi.reshape(3, 8, -1) # (GD, LDF, NQ)
        
        A = np.einsum('tim, sjm, m, ts -> tsji', gvarphi, gvarphi, ws, np.full((3,3), self.cellm))        
        return A
    
    def cell_to_dof(self):
        """
            (nx*ny*nz, 8)
        """
        NN = (self.nx+1) * (self.ny+1) * (self.nz+1)
        NC = self.nx * self.ny * self.nz
        c2d = np.arange(NN).reshape(self.nx+1, self.ny+1, self.nz+1)
        c2d = c2d[:-1, :-1, :-1]
        c2d = c2d.reshape(-1)
        c2d = np.broadcast_to(c2d[:, None], (NC, 8))
        c2d = c2d.copy()
        c2d[:, 1] = c2d[:, 0] + 1
        c2d[:, 2] = c2d[:, 0] + self.nz + 1
        c2d[:, 3] = c2d[:, 2] + 1
        c2d[:, 4] = c2d[:, 0] + (self.ny+1)*(self.nz+1)
        c2d[:, 5] = c2d[:, 4] + 1
        c2d[:, 6] = c2d[:, 4] + self.nz + 1
        c2d[:, 7] = c2d[:, 6] + 1
        return c2d

    def get_nodes_from_grid(self):
        """
            (GD, (NXG+1)*(NYG+1)*(NZG+1))
        """
        
        node = np.mgrid[self.w[0]:self.w[1]:complex(0,self.nx+1), 
                        self.w[2]:self.w[3]:complex(0,self.ny+1),
                        self.w[4]:self.w[5]:complex(0,self.nz+1)]
        
        node = node.reshape(3, -1)   
        return node

    def is_boundary_dof(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        Idx = np.array([(ny+2)*(nz+1), (ny+2)*(nz+1)+nz])
        Idx = np.arange(0, (ny-1)*(nz+1), nz+1) + Idx[:, np.newaxis]
        Idx = np.r_[np.arange((ny+1)*(nz+1), (ny+2)*(nz+1)), Idx.flat,
                    np.arange((2*ny+1)*(nz+1), (2*ny+2)*(nz+1))]
        Idx = np.arange(0,(nx-1)*(ny+1)*(nz+1),(ny+1)*(nz+1)) + Idx[:, np.newaxis]
        Idx = np.r_[np.arange((ny+1)*(nz+1)), Idx.flat, 
                    np.arange(nx*(ny+1)*(nz+1), (nx+1)*(ny+1)*(nz+1))]
        return Idx     

    def solve(self, Rs, Ds, Fs):
        """
            Rs, (NC, 2, 2)
            Ds, (NC, 6, 6)
            Fs, (NC, 2)
        """
        
        I = self.Ki_mass_matrix() # (LDF, LDF)
        A = np.einsum('lmn, ij -> mnlij', Rs, I) # (2, 2, NC, LDF, LDF)
        # A = np.zeros(A.shape)
        I = self.Ki_stiff_matrix() # (GD, GD, LDF, LDF)
        Ds = Ds.reshape(-1, 2, 3, 2, 3)
        I = np.einsum('lmgnp, gpij -> mnlij', Ds, I) # (2, 2, NC, LDF, LDF)
        A += I
        c2d = self.cell_to_dof() # (NC, LDF)
        c2d = np.stack((c2d, c2d+self.nn)) # (2, NC, LDF)
        I = np.broadcast_to(c2d[:, None, :, :, None], A.shape)
        J = np.broadcast_to(c2d[None, :, :, None, :], A.shape)
        A = csr_matrix((A.flat, (I.flat,J.flat)), (self.nn*2,self.nn*2))
        
        f = np.einsum('lm, i -> mli', Fs, np.full(8, self.cellm/8))
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

        # UH = scsolve(A, F)
        UH = pysolve(A, F)
        
        UH = UH.reshape(2, self.nn)
        UHa = UH[:, c2d[0]] # (2, NC, LDF)
        UHa = np.mean(UHa, axis=2)
        # UHa = np.einsum('tli, i -> tl', UHa, np.full(8, 0.125))
        UHa = UHa.reshape(2, self.nx, self.ny, self.nz)
        UH = UH.reshape(2, self.nx+1, self.ny+1, self.nz+1)
        return UH, UHa
    
    def write_to_vtk(self, fname):
        
        nodes = self.get_nodes_from_grid()
        cells = self.cell_to_dof()
        nodedata = self.nodedata
        celldata = self.celldata
        write_to_vtk_Hexahedron(fname, nodes, cells, nodedata, celldata)
  