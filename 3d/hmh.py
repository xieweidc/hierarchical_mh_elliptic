import numpy as np
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve as scsolve
from pypardiso import spsolve as pysolve

from mesh import HexahedronMesh, MHMacroMesh


# Interpolation to fine-grid
# (LDC, q**GD, LDF), (8, 8, 8)
Q = np.array([[[1.   , 0.5  , 0.5  , 0.25 , 0.5  , 0.25 , 0.25 , 0.125],
               [0.5  , 0.   , 0.25 , 0.   , 0.25 , 0.   , 0.125, 0.   ],
               [0.5  , 0.25 , 0.   , 0.   , 0.25 , 0.125, 0.   , 0.   ],
               [0.25 , 0.   , 0.   , 0.   , 0.125, 0.   , 0.   , 0.   ],
               [0.5  , 0.25 , 0.25 , 0.125, 0.   , 0.   , 0.   , 0.   ],
               [0.25 , 0.   , 0.125, 0.   , 0.   , 0.   , 0.   , 0.   ],
               [0.25 , 0.125, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
               [0.125, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]],
        
              [[0.   , 0.5  , 0.   , 0.25 , 0.   , 0.25 , 0.   , 0.125],
               [0.5  , 1.   , 0.25 , 0.5  , 0.25 , 0.5  , 0.125, 0.25 ],
               [0.   , 0.25 , 0.   , 0.   , 0.   , 0.125, 0.   , 0.   ],
               [0.25 , 0.5  , 0.   , 0.   , 0.125, 0.25 , 0.   , 0.   ],
               [0.   , 0.25 , 0.   , 0.125, 0.   , 0.   , 0.   , 0.   ],
               [0.25 , 0.5  , 0.125, 0.25 , 0.   , 0.   , 0.   , 0.   ],
               [0.   , 0.125, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
               [0.125, 0.25 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]],
        
              [[0.   , 0.   , 0.5  , 0.25 , 0.   , 0.   , 0.25 , 0.125],
               [0.   , 0.   , 0.25 , 0.   , 0.   , 0.   , 0.125, 0.   ],
               [0.5  , 0.25 , 1.   , 0.5  , 0.25 , 0.125, 0.5  , 0.25 ],
               [0.25 , 0.   , 0.5  , 0.   , 0.125, 0.   , 0.25 , 0.   ],
               [0.   , 0.   , 0.25 , 0.125, 0.   , 0.   , 0.   , 0.   ],
               [0.   , 0.   , 0.125, 0.   , 0.   , 0.   , 0.   , 0.   ],
               [0.25 , 0.125, 0.5  , 0.25 , 0.   , 0.   , 0.   , 0.   ],
               [0.125, 0.   , 0.25 , 0.   , 0.   , 0.   , 0.   , 0.   ]],
        
              [[0.   , 0.   , 0.   , 0.25 , 0.   , 0.   , 0.   , 0.125],
               [0.   , 0.   , 0.25 , 0.5  , 0.   , 0.   , 0.125, 0.25 ],
               [0.   , 0.25 , 0.   , 0.5  , 0.   , 0.125, 0.   , 0.25 ],
               [0.25 , 0.5  , 0.5  , 1.   , 0.125, 0.25 , 0.25 , 0.5  ],
               [0.   , 0.   , 0.   , 0.125, 0.   , 0.   , 0.   , 0.   ],
               [0.   , 0.   , 0.125, 0.25 , 0.   , 0.   , 0.   , 0.   ],
               [0.   , 0.125, 0.   , 0.25 , 0.   , 0.   , 0.   , 0.   ],
               [0.125, 0.25 , 0.25 , 0.5  , 0.   , 0.   , 0.   , 0.   ]],
       
              [[0.   , 0.   , 0.   , 0.   , 0.5  , 0.25 , 0.25 , 0.125],
               [0.   , 0.   , 0.   , 0.   , 0.25 , 0.   , 0.125, 0.   ],
               [0.   , 0.   , 0.   , 0.   , 0.25 , 0.125, 0.   , 0.   ],
               [0.   , 0.   , 0.   , 0.   , 0.125, 0.   , 0.   , 0.   ],
               [0.5  , 0.25 , 0.25 , 0.125, 1.   , 0.5  , 0.5  , 0.25 ],
               [0.25 , 0.   , 0.125, 0.   , 0.5  , 0.   , 0.25 , 0.   ],
               [0.25 , 0.125, 0.   , 0.   , 0.5  , 0.25 , 0.   , 0.   ],
               [0.125, 0.   , 0.   , 0.   , 0.25 , 0.   , 0.   , 0.   ]],
        
              [[0.   , 0.   , 0.   , 0.   , 0.   , 0.25 , 0.   , 0.125],
               [0.   , 0.   , 0.   , 0.   , 0.25 , 0.5  , 0.125, 0.25 ],
               [0.   , 0.   , 0.   , 0.   , 0.   , 0.125, 0.   , 0.   ],
               [0.   , 0.   , 0.   , 0.   , 0.125, 0.25 , 0.   , 0.   ],
               [0.   , 0.25 , 0.   , 0.125, 0.   , 0.5  , 0.   , 0.25 ],
               [0.25 , 0.5  , 0.125, 0.25 , 0.5  , 1.   , 0.25 , 0.5  ],
               [0.   , 0.125, 0.   , 0.   , 0.   , 0.25 , 0.   , 0.   ],
               [0.125, 0.25 , 0.   , 0.   , 0.25 , 0.5  , 0.   , 0.   ]],
       
              [[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.25 , 0.125],
               [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.125, 0.   ],
               [0.   , 0.   , 0.   , 0.   , 0.25 , 0.125, 0.5  , 0.25 ],
               [0.   , 0.   , 0.   , 0.   , 0.125, 0.   , 0.25 , 0.   ],
               [0.   , 0.   , 0.25 , 0.125, 0.   , 0.   , 0.5  , 0.25 ],
               [0.   , 0.   , 0.125, 0.   , 0.   , 0.   , 0.25 , 0.   ],
               [0.25 , 0.125, 0.5  , 0.25 , 0.5  , 0.25 , 1.   , 0.5  ],
               [0.125, 0.   , 0.25 , 0.   , 0.25 , 0.   , 0.5  , 0.   ]],
        
              [[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.125],
               [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.125, 0.25 ],
               [0.   , 0.   , 0.   , 0.   , 0.   , 0.125, 0.   , 0.25 ],
               [0.   , 0.   , 0.   , 0.   , 0.125, 0.25 , 0.25 , 0.5  ],
               [0.   , 0.   , 0.   , 0.125, 0.   , 0.   , 0.   , 0.25 ],
               [0.   , 0.   , 0.125, 0.25 , 0.   , 0.   , 0.25 , 0.5  ],
               [0.   , 0.125, 0.   , 0.25 , 0.   , 0.25 , 0.   , 0.5  ],
               [0.125, 0.25 , 0.25 , 0.5  , 0.25 , 0.5  , 0.5  , 1.   ]]])
    

def linear_intepolation_hexmesh(q):
    # R, (LDC, q**GD, LDF)
    
    R = np.linspace(0, 1, q+1)
    R = np.c_[1-R, R] # (2, q+1)
        
    R = np.einsum('li, mj, nk -> ijklmn', R, R, R)
    R = R.reshape(8, q+1, q+1, q+1)
    R = np.stack((R[:, :-1, :-1, :-1], R[:, :-1, :-1, 1:],
                  R[:, :-1, 1:, :-1], R[:, :-1, 1:, 1:],
                  R[:, 1:, :-1, :-1], R[:, 1:, :-1, 1:],
                  R[:, 1:, 1:, :-1], R[:, 1:, 1:, 1:]), axis=-1)
    R = R.reshape(8, -1, 8)
    return R

def phi_for_Rpm_hmh(k, psi, f, Rpp):
    # k, (2^GD, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
    # f, (2^GD, NXF, NYF)
    D = np.zeros((8, 6, 6))
    R = np.zeros((8, 2, 2))
    F = np.zeros((8, 2))
    # The finest grid, T0
    R[7], D[7], F[7], phiRpm0 = phi_for_Rpm_T0(k[7], psi[7], f[7], Rpp)
    # T1 grid
    for i in range(7):
        R[i], D[i], F[i] = phi_for_Rpm_T1(k[i], psi[i], f[i], Rpp, phiRpm0)
    return R, D, F
    
def phi_for_Rpm_T0(k, psi, f, w):
    m2, nx, I0, ny, J0, nz = k.shape
    
    k = np.broadcast_to(k[:,:,None,:,:,None,:,:,None], (m2,nx,2,m2,ny,2,m2,nz,2))
    psi = np.broadcast_to(psi[:,:,None,:,:,None,:,:,None], (m2,nx,2,m2,ny,2,m2,nz,2))
    f = np.broadcast_to(f[:,None,:,None,:,None], (nx,2,ny,2,nz,2))
    nx *= 2
    ny *= 2
    nz *= 2
    k = k.reshape(m2, nx, m2, ny, m2, nz)
    psi = psi.reshape(m2, nx, m2, ny, m2, nz)
    f = f.reshape(nx, ny, nz)
    
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
    phiRpm = meshp.set_zero_dirichlet_bc(A, S)
    phi = phiRpm[:-2*m2**3]
    
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
    return Bs[:2,:2], Bs[2:,2:], bs, phiRpm

def phi_for_Rpm_T1(k, psi, f, w, phiRpm0):
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
    # Minus T0 phiRpm.
    mesh0 = HexahedronMesh(w, 2*m2*nx, 2*m2*ny, 2*m2*nz)
    c2d = mesh0.cell_to_dof()
    phiRpm0 = phiRpm0[c2d]                   # (NC, LDF, 8)
    phiRpm0 = phiRpm0.reshape(m2*nx, 2, m2*ny, 2, m2*nz, 2, 8, 8)
    phiRpm0 = phiRpm0.transpose(0, 2, 4, 1, 3, 5, 6, 7)
    phiRpm0 = phiRpm0.reshape(-1, 8, 8, 8)   # (m2*nx*m2*ny*m2*nz, r^3, LDF, 8)
    S = mesh0.cell_stiff_matrix_varphi()     # (LDF, LDF)
    I = np.einsum('l, lrjm, ij, tri -> ltm', 
                  k.flat, phiRpm0, -S, Q)    # (m2*nx*m2*ny*m2*nz, LDC, 8)
    c2d = meshp.cell_to_dof()   
    NC = c2d.shape[0]
    S = np.zeros((A.shape[0], 8))
    for i in range(6):
        np.add.at(S[:, i], c2d, I[:, :, i])
    # constraint part
    J = np.c_[np.full(NC, 1/64), np.zeros(NC)] # (NC, 2)
    J[I0, 0] = 0
    J[I0, 1] = 1/64
    J = J.reshape(m2, nx, m2, ny, m2, nz, 2)
    phiRpm0 = phiRpm0.reshape(m2, nx, m2, ny, m2, nz, 8, 8, 8) # (r^3, LDF, 8)
    J0 = np.sum(psi==0, axis=(1, 3, 5))       # (m2, m2, m2)
    J0 = np.stack((J0, (nx*ny*nz-J0)))        # (2, m2, m2, m2)
    J0 = 1 / J0                               # (2, m2, m2, m2)
    I = np.einsum('mxnykzrli, mxnykzj, jmnk -> jmnki',
                  phiRpm0, J, J0)             # (2, m2, m2, m2, 8)
    S[-2*m2**3:] = I.reshape(-1, 8)
    
    # Original part
    S[-2*m2**3:-m2**3, 0] -= 1
    S[-m2**3:, 1] -= 1
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
    
    S[-2*m2**3:-m2**3, 2:5] -= I[:, :3]
    S[-m2**3:, 5:] -= I[:, 3:]
    
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
    phi = phi.reshape(nx, ny, nz, 8, 8)
    phi = np.einsum('xyzci, crf -> xyzrfi', phi, Q)
    # (m2, nx, m2, ny, m2, nz, r^3, LDF, 8) -> (nx, ny, nz, r^3, LDF, 8) 
    phiRpm0 = phiRpm0[m2//2, :, m2//2, :, m2//2]
    phi = phi + phiRpm0
    phi = phi.reshape(-1, 8, 8, 8)

    A = mesh0.cell_stiff_matrix_varphi()
    Bs = np.einsum('lrmi, lrnj, mn, l -> ij', phi, phi, A, k[S,:,S,:,S].flat)
    bs = np.einsum('lrmi, l, m -> i', phi[:,:,:,:2], f.flat, np.full(8, mesh0.cellm/8))
    return Bs[:2,:2], Bs[2:,2:], bs

    
class HierarchicalMulticontinuumHomogenization:
    
    def __init__(self, ps):
        self.ps = ps
        self.Omega = ps.Omega
        self.nxc = ps.nxc
        self.nyc = ps.nyc
        self.nzc = ps.nzc
        self.nxp = ps.nxp
        self.nyp = ps.nyp
        self.nzp = ps.nzp
        self.nxf = ps.nxf
        self.nyf = ps.nyf
        self.nzf = ps.nzf
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
        
        a = a.transpose(0, 2, 4, 1, 3, 5)
        a = a.reshape(-1, nxf, nyf, nzf)
        
        if I == 0:
            I = np.arange(self.ncc*self.ncp)
            I = I.reshape(self.nxc, self.nxp, self.nyc, self.nyp, self.nzc, self.nzp)
            I = I[:, self.nxp//2, :, self.nyp//2, :, self.nzp//2]
            I = I.reshape(self.nxc//2, 2, self.nyc//2, 2, self.nzc//2, 2)
            I = I.transpose(0, 2, 4, 1, 3, 5)
            I = I.reshape(-1, 8)
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
            a = a.reshape(self.nxc//2, 2, self.nyc//2, 2, self.nzc//2, 2,
                          2*self.m+1, 2*self.m+1, 2*self.m+1, nxf, nyf, nzf)
            # (NXC//2, 2, NYC//2, 2, NZC//2, 2, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
            # -> (NCC//8, 2^GD, 2m+1, NXF, 2m+1, NYF, 2m+1, NZF)
            a = a.transpose(0, 2, 4, 1, 3, 5, 6, 9, 7, 10, 8, 11)
            a = a.reshape(self.ncc//8, 8, 2*self.m+1, nxf, 2*self.m+1, nyf, 2*self.m+1, nzf)
        return a
        
    def solve(self, k, Psi, f, way='t'):
        
        k = self.broadcast(k)
        Psi = self.broadcast(Psi)
        f = f.reshape(self.nxc*self.nxp, self.nxf, self.nyc*self.nyp, 
                      self.nyf, self.nzc*self.nzp, self.nzf)
        f = self.broadcast(f, 0)
        
        Rpp = np.array([0, (2*self.m+1)*self.hxc/self.nxp, 
                        0, (2*self.m+1)*self.hyc/self.nyp,
                        0, (2*self.m+1)*self.hzc/self.nzp])
        
        if way == 'm':
            nc = mp.cpu_count() - 2
            pool = mp.Pool(processes=nc)
            Rpp = np.broadcast_to(Rpp[None,:], (self.ncc,6))
            args = list(zip(k, Psi, f, Rpp))
            res = pool.starmap(phi_for_Rpm_hmh, args)
            Rs, Ds, Fs = map(list, zip(*res))
            Rs = np.array(Rs)
            Ds = np.array(Ds)
            Fs = np.array(Fs)
        else:        
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
        Ds = Ds.reshape(-1, 2, 2)
        # Source term coefficient
        Fs = Fs.reshape(self.nxc//2, self.nyc//2, self.nzc//2, 2, 2, 2, 2)
        Fs = Fs.transpose(0, 3, 1, 4, 2, 5, 6)
        Fs = Fs.reshape(-1, 2)
        
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc, self.nzc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa, Rs, Ds, Fs
        




