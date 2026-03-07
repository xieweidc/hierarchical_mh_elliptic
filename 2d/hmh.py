import numpy as np
from copy import deepcopy
import multiprocessing as mp
from scipy.sparse import csr_matrix

from mesh2d import RectangleMesh, MHMacroMesh


class MicroPhi:
    def __init__(self, Omega, nxc, nyc, nxf, nyf, m, L, nc=None):
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
        self.L = L
        self.nc=nc if nc is not None else mp.cpu_count()-2
        
    def get_local_domain(self, xth, yth):
        
        m2 = 2*self.m + 1
        x = np.arange(2**(self.L-1))*self.hxc 
        x += self.Omega[0] + (xth*2**(self.L-1)-self.m)*self.hxc
        y = np.arange(2**(self.L-1))*self.hyc 
        y += self.Omega[2] + (yth*2**(self.L-1)-self.m)*self.hyc
        x, y = np.meshgrid(x, y)
        ws = np.c_[x.flat, x.flat+m2*self.hxc, y.flat, y.flat+m2*self.hyc]
        return ws
        
    def linear_intepolation_quadmesh(self, r):
        # R, (LDC, r*r, LDF)
        if r == 2: # (4, 4, 4)
            R = np.array([[[1.  , 0.5 , 0.5 , 0.25],
                           [0.5 , 0.  , 0.25, 0.  ],
                           [0.5 , 0.25, 0.  , 0.  ],
                           [0.25, 0.  , 0.  , 0.  ]],
         
                          [[0.  , 0.5 , 0.  , 0.25],
                           [0.5 , 1.  , 0.25, 0.5 ],
                           [0.  , 0.25, 0.  , 0.  ],
                           [0.25, 0.5 , 0.  , 0.  ]],
         
                          [[0.  , 0.  , 0.5 , 0.25],
                           [0.  , 0.  , 0.25, 0.  ],
                           [0.5 , 0.25, 1.  , 0.5 ],
                           [0.25, 0.  , 0.5 , 0.  ]],
         
                          [[0.  , 0.  , 0.  , 0.25],
                           [0.  , 0.  , 0.25, 0.5 ],
                           [0.  , 0.25, 0.  , 0.5 ],
                           [0.25, 0.5 , 0.5 , 1.  ]]])
        elif r == 4: # (4, 16, 4)
            R = np.array([[[1.    , 0.75  , 0.75  , 0.5625],
                           [0.75  , 0.5   , 0.5625, 0.375 ],
                           [0.5   , 0.25  , 0.375 , 0.1875],
                           [0.25  , 0.    , 0.1875, 0.    ],
                           [0.75  , 0.5625, 0.5   , 0.375 ],
                           [0.5625, 0.375 , 0.375 , 0.25  ],
                           [0.375 , 0.1875, 0.25  , 0.125 ],
                           [0.1875, 0.    , 0.125 , 0.    ],
                           [0.5   , 0.375 , 0.25  , 0.1875],
                           [0.375 , 0.25  , 0.1875, 0.125 ],
                           [0.25  , 0.125 , 0.125 , 0.0625],
                           [0.125 , 0.    , 0.0625, 0.    ],
                           [0.25  , 0.1875, 0.    , 0.    ],
                           [0.1875, 0.125 , 0.    , 0.    ],
                           [0.125 , 0.0625, 0.    , 0.    ],
                           [0.0625, 0.    , 0.    , 0.    ]],
     
                          [[0.    , 0.25  , 0.    , 0.1875],
                           [0.25  , 0.5   , 0.1875, 0.375 ],
                           [0.5   , 0.75  , 0.375 , 0.5625],
                           [0.75  , 1.    , 0.5625, 0.75  ],
                           [0.    , 0.1875, 0.    , 0.125 ],
                           [0.1875, 0.375 , 0.125 , 0.25  ],
                           [0.375 , 0.5625, 0.25  , 0.375 ],
                           [0.5625, 0.75  , 0.375 , 0.5   ],
                           [0.    , 0.125 , 0.    , 0.0625],
                           [0.125 , 0.25  , 0.0625, 0.125 ],
                           [0.25  , 0.375 , 0.125 , 0.1875],
                           [0.375 , 0.5   , 0.1875, 0.25  ],
                           [0.    , 0.0625, 0.    , 0.    ],
                           [0.0625, 0.125 , 0.    , 0.    ],
                           [0.125 , 0.1875, 0.    , 0.    ],
                           [0.1875, 0.25  , 0.    , 0.    ]],
     
                          [[0.    , 0.    , 0.25  , 0.1875],
                           [0.    , 0.    , 0.1875, 0.125 ],
                           [0.    , 0.    , 0.125 , 0.0625],
                           [0.    , 0.    , 0.0625, 0.    ],
                           [0.25  , 0.1875, 0.5   , 0.375 ],
                           [0.1875, 0.125 , 0.375 , 0.25  ],
                           [0.125 , 0.0625, 0.25  , 0.125 ],
                           [0.0625, 0.    , 0.125 , 0.    ],
                           [0.5   , 0.375 , 0.75  , 0.5625],
                           [0.375 , 0.25  , 0.5625, 0.375 ],
                           [0.25  , 0.125 , 0.375 , 0.1875],
                           [0.125 , 0.    , 0.1875, 0.    ],
                           [0.75  , 0.5625, 1.    , 0.75  ],
                           [0.5625, 0.375 , 0.75  , 0.5   ],
                           [0.375 , 0.1875, 0.5   , 0.25  ],
                           [0.1875, 0.    , 0.25  , 0.    ]],
     
                          [[0.    , 0.    , 0.    , 0.0625],
                           [0.    , 0.    , 0.0625, 0.125 ],
                           [0.    , 0.    , 0.125 , 0.1875],
                           [0.    , 0.    , 0.1875, 0.25  ],
                           [0.    , 0.0625, 0.    , 0.125 ],
                           [0.0625, 0.125 , 0.125 , 0.25  ],
                           [0.125 , 0.1875, 0.25  , 0.375 ],
                           [0.1875, 0.25  , 0.375 , 0.5   ],
                           [0.    , 0.125 , 0.    , 0.1875],
                           [0.125 , 0.25  , 0.1875, 0.375 ],
                           [0.25  , 0.375 , 0.375 , 0.5625],
                           [0.375 , 0.5   , 0.5625, 0.75  ],
                           [0.    , 0.1875, 0.    , 0.25  ],
                           [0.1875, 0.375 , 0.25  , 0.5   ],
                           [0.375 , 0.5625, 0.5   , 0.75  ],
                           [0.5625, 0.75  , 0.75  , 1.    ]]])
        return R
    
    def phi_for_Kim_L2(self, ith, k, psi, f):
        # k, (2m+2, NXF, 2m+2, NYF)
        # f, (4, NXF, NYF)
        xth, yth = divmod(ith, self.nyc//2)
        m = self.m
        m2 = 2*m + 1
        
        ws = self.get_local_domain(xth, yth)
        
        D = np.zeros((4, 4, 4))
        R = np.zeros((4, 2, 2))
        F = np.zeros((4, 2))
                
        # The finest grid, T0
        phiKim0, R[3], D[3], F[3] = self.phi_for_Kim_T0(ws[3], k[1:,:,1:], psi[1:,:,1:], f[3])
        # T1 grid
        for i in range(3):
            xth, yth = divmod(i, 2)
            R[i], D[i], F[i] = self.phi_for_Kim_T1(ws[i], 
        k[xth:xth+m2,:,yth:yth+m2], psi[xth:xth+m2,:,yth:yth+m2], f[i], phiKim0)
        return R, D, F
    
    def phi_for_Kim_L3(self, ith, k, psi, f):
        # k, (2m+4, NXF, 2m+4, NYF)
        # f, (16, NXF, NYF)
        xth, yth = divmod(ith, self.nyc//4)
        m = self.m
        m2 = 2*m + 1
        
        ws = self.get_local_domain(xth, yth)
        
        D = np.zeros((16, 4, 4))
        R = np.zeros((16, 2, 2))
        F = np.zeros((16, 2))
        
        # The finest grid, T0
        phiKim0, R[10], D[10], F[10] = self.phi_for_Kim_T0(ws[10], k[2:-1,:,2:-1], psi[2:-1,:,2:-1], f[10])
        # T1 grid
        for i in [0, 2, 8]:
            xth, yth = divmod(i, 4)
            R[i], D[i], F[i] = self.phi_for_Kim_T1(ws[i], 
        k[xth:xth+m2,:,yth:yth+m2], psi[xth:xth+m2,:,yth:yth+m2], f[i], phiKim0)
        # T2 grid
        for i in [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]:
            xth, yth = divmod(i, 4)
            R[i], D[i], F[i] = self.phi_for_Kim_T2(ws[i],
        k[xth:xth+m2,:,yth:yth+m2], psi[xth:xth+m2,:,yth:yth+m2], f[i], phiKim0)
        return R, D, F
    
    def phi_for_Kim_T0(self, w0, k0, psi0, f):
        m2, nx, I, ny = k0.shape
        m = m2 // 2
        k0 = np.broadcast_to(k0[:,:,None,:,:,None], (m2,nx,4,m2,ny,4))
        psi0 = np.broadcast_to(psi0[:,:,None,:,:,None], (m2,nx,4,m2,ny,4))
        f = np.broadcast_to(f[:,None,:,None], (nx,4,ny,4))
        nx = nx * 4
        ny = ny * 4
        k0 = k0.reshape(m2, nx, m2, ny)
        psi0 = psi0.reshape(m2, nx, m2, ny)
        mesh0 = RectangleMesh(w0, m2*nx, m2*ny)
        
        I0 = psi0.astype(np.float64)
        I0 = np.stack((1-I0, I0)) # (2, 2m+1, NXF, 2m+1, NYF)
        J0 = np.sum(I0, axis=(2,4)) # (2, 2m+1, 2m+1)
        J0 = np.broadcast_to(J0[:,:,None,:,None], (2,m2,nx,m2,ny))
        J0 = J0.reshape(2, -1)
        
        # Constraint
        c2d = mesh0.cell_to_dof(m2*nx, m2*ny)
        NN = (m2*nx+1) * (m2*ny+1)
        S = np.full(c2d.shape, -0.25) # (NC, 4)
        I0, = np.where(psi0.flat == 0)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
        I0, = np.where(psi0.flat == 1)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])
        
        J1 = np.arange(NN, NN+m2**2).reshape(m2, m2)
        J1 = np.broadcast_to(J1[:, None, :, None], (m2, nx, m2, ny))
        J1 = J1.reshape(-1)
        J1[I0] += m2**2
        J1 = np.broadcast_to(J1[:, None], S.shape)        
        # Stiff matrix
        A = mesh0.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k0.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*m2**2
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
        
        # Right hand side
        S = np.zeros((A.shape[0], 6))
        J0 = np.sum(psi0==0, axis=(1, 3)) # (m2, m2)
        J0 = np.stack((J0, nx*ny-J0))
        S[-2*m2**2:-m2**2, 0] = -1
        S[-m2**2:, 1] = -1
        NC = c2d.shape[0]
        I = np.mgrid[w0[0]:w0[1]:complex(0,m2*nx+1),
                     w0[2]:w0[3]:complex(0,m2*ny+1)] # (GD, NX+1, NY+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, 4)
        I = I.reshape(2, m2, nx, m2, ny, 4)
        J = np.c_[np.full(NC, 0.25), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 0.25
        J = J.reshape(m2, nx, m2, ny, 2)
        I = np.einsum('tmpnql, mpnqi, imn -> mnit', I, J, 1/J0) # (m2, m2, 2, GD)
        I = I.reshape(m2**2, 4)
        I -= I[I.shape[0]//2]
        S[-2*m2**2:-m2**2, 2:4] = -I[:, :2]
        S[-m2**2:, 4:] = -I[:, 2:]
        
        # Boudary condition
        phiKim = mesh0.set_zero_dirichlet(A, S)
        
        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh0.ada(m*nx*(m2*ny+1)+m*ny, nx+1, ny+1, m2*ny+1)
        phi = phiKim[I] # (NN, 6)
        
        c2d = mesh0.cell_to_dof(nx, ny)
        phi = phi[c2d] # (NC, 4, 6)
        A = mesh0.cell_stiff_matrix_varphi()
        Bs = np.einsum('lmi, lnj, mn, l -> ij', phi, phi, A, k0[m,:,m].flat)
        bs = np.einsum('lmi, l, m -> i', phi[:,:,:2], f.flat, np.full(4, mesh0.cellm*0.25))
        return phiKim, Bs[:2,:2], Bs[2:,2:], bs
            
    def phi_for_Kim_T1(self, w1, k1, psi1, f, phiKim0):
        m2, nx, I, ny = k1.shape
        m = m2 // 2
        k1 = np.broadcast_to(k1[:,:,None,:,:,None], (m2,nx,2,m2,ny,2))
        psi1 = np.broadcast_to(psi1[:,:,None,:,:,None], (m2,nx,2,m2,ny,2))
        f = np.broadcast_to(f[:,None,:,None], (nx,2,ny,2))
        nx = nx * 2
        ny = ny * 2
        k1 = k1.reshape(m2, nx, m2, ny)
        psi1 = psi1.reshape(m2, nx, m2, ny)
        mesh1 = RectangleMesh(w1, m2*nx, m2*ny)
        
        # Constraint
        c2d = mesh1.cell_to_dof(m2*nx, m2*ny)
        I0 = psi1.astype(np.float64)
        I0 = np.stack((1-I0, I0)) # (2, 2m+1, NXF, 2m+1, NYF)
        J0 = np.sum(I0, axis=(2,4)) # (2, 2m+1, 2m+1)
        J0 = np.broadcast_to(J0[:,:,None,:,None], (2,m2,nx,m2,ny))
        J0 = J0.reshape(2, -1)
        NN = (m2*nx+1) * (m2*ny+1)
        S = np.full(c2d.shape, -0.25) # (NC, 4)
        I0, = np.where(psi1.flat == 0)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
        I0, = np.where(psi1.flat == 1)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])  
        J1 = np.arange(NN, NN+m2**2).reshape(m2, m2)
        J1 = np.broadcast_to(J1[:, None, :, None], (m2, nx, m2, ny))
        J1 = J1.reshape(-1)
        J1[I0] += m2**2
        J1 = np.broadcast_to(J1[:, None], S.shape)
        # Stiff matrix
        A = mesh1.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k1.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*m2**2
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
        
        # Right hand side
        # Minus T0 phiKim.
        c2d = mesh1.cell_to_dof(2*m2*nx, 2*m2*ny)
        phiKim0 = phiKim0[c2d] # (NC, LDF, 6)
        phiKim0 = phiKim0.reshape(m2*nx,2,m2*ny,2,4,6)
        phiKim0 = phiKim0.swapaxes(1, 2) # (m2*nx,m2*ny,2,2,LDF,6)   
        phiKim0 = phiKim0.reshape(-1, 4, 4, 6) # (m2*nx*m2*ny, r*r, LDF, 6)
        S = mesh1.cell_stiff_matrix_varphi() # Only the same in 2d case.
        I = self.linear_intepolation_quadmesh(2) # (LDC, r*r, LDF)
        I = np.einsum('l, lrjm, ij, tri -> ltm', 
                      k1.flat, phiKim0, -S, I) # (m2*nx*m2*ny, LDC, 6)
        c2d = mesh1.cell_to_dof(m2*nx, m2*ny)   
        NC = c2d.shape[0]     
        S = np.zeros((A.shape[0], 6))
        for i in range(6):
            np.add.at(S[:, i], c2d, I[:, :, i])
        # constraint part
        J = np.c_[np.full(NC, 1/16), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 1/16
        J = J.reshape(m2, nx, m2, ny, 2)
        phiKim0 = phiKim0.reshape(m2, nx, m2, ny, 4, 4, 6)
        J0 = np.sum(psi1==0, axis=(1, 3)) # (m2, m2)
        J0 = np.stack((J0, (nx*ny-J0)))   # (2, m2, m2)
        J0 = 1 / J0                       # (2, m2, m2)
        I = np.einsum('mxnyrli, mxnyj, jmn -> jmni', phiKim0, J, J0) # (2, m2, m2, 6)
        S[-2*m2**2:] = I.reshape(-1, 6)
        
        # Original part
        S[-2*m2**2:-m2**2, 0] -= 1
        S[-m2**2:, 1] -= 1
        I = np.mgrid[w1[0]:w1[1]:complex(0,m2*nx+1),
                     w1[2]:w1[3]:complex(0,m2*ny+1)] # (GD, NX+1, NY+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, 4)
        I = I.reshape(2, m2, nx, m2, ny, 4)
        J = np.c_[np.full(NC, 0.25), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 0.25
        J = J.reshape(m2, nx, m2, ny, 2)
        I = np.einsum('tmpnql, mpnqi, imn -> mnit', I, J, J0) # (m2, m2, 2, GD)
        I = I.reshape(m2**2, 4)
        I -= I[I.shape[0]//2]
        S[-2*m2**2:-m2**2, 2:4] -= I[:, :2]
        S[-m2**2:, 4:] -= I[:, 2:]

        # Boudary condition
        phi = mesh1.set_zero_dirichlet(A, S)
        
        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh1.ada(m*nx*(m2*ny+1)+m*ny, nx+1, ny+1, m2*ny+1)
        phi = phi[I] # (NN, 6)
        c2d = mesh1.cell_to_dof(nx, ny)
        phi = phi[c2d] # (nx, ny, LDC, 6)
        phi = phi.reshape(nx, ny, 4, 6)
        I = self.linear_intepolation_quadmesh(2) # (LDC, r*r, LDF)
        phi = np.einsum('xyci, crf -> xyrfi', phi, I)
        phiKim0 = phiKim0[m, :, m] # (nx, ny, r*r, LDF, 6)
        phi = phi + phiKim0
        phi = phi.reshape(-1, 4, 4, 6)
        
        A = mesh1.cell_stiff_matrix_varphi()
        Bs = np.einsum('lrmi, lrnj, mn, l -> ij', phi, phi, A, k1[m,:,m].flat)
        bs = np.einsum('lrmi, l, m -> i', phi[:,:,:,:2], f.flat, np.full(4, mesh1.cellm/16))
        return Bs[:2,:2], Bs[2:,2:], bs
        
    def phi_for_Kim_T2(self, w2, k2, psi2, f, phiKim0):
        m2, nx, I, ny = k2.shape
        m = m2 // 2
        mesh2 = RectangleMesh(w2, m2*nx, m2*ny)
        
        # Constraint
        c2d = mesh2.cell_to_dof(m2*nx, m2*ny)
        I0 = psi2.astype(np.float64)
        I0 = np.stack((1-I0, I0)) # (2, 2m+1, NXF, 2m+1, NYF)
        J0 = np.sum(I0, axis=(2,4)) # (2, 2m+1, 2m+1)
        J0 = np.broadcast_to(J0[:,:,None,:,None], (2,m2,nx,m2,ny))
        J0 = J0.reshape(2, -1)
        NN = (m2*nx+1) * (m2*ny+1)
        S = np.full(c2d.shape, -0.25) # (NC, 4)
        I0, = np.where(psi2.flat == 0)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[0,I0])
        I0, = np.where(psi2.flat == 1)
        S[I0] = np.einsum('li, l -> li', S[I0], 1/J0[1,I0])  
        J1 = np.arange(NN, NN+m2**2).reshape(m2, m2)
        J1 = np.broadcast_to(J1[:, None, :, None], (m2, nx, m2, ny))
        J1 = J1.reshape(-1)
        J1[I0] += m2**2
        J1 = np.broadcast_to(J1[:, None], S.shape)        
        # Stiff matrix
        A = mesh2.cell_stiff_matrix_varphi()
        A = np.einsum('l, ij -> lij', k2.flat, A)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        A = np.r_[A.flat, S.flat, S.flat]
        I = np.r_[I.flat, c2d.flat, J1.flat]
        J = np.r_[J.flat, J1.flat, c2d.flat]
        NN += 2*m2**2
        A = csr_matrix((A, (I, J)), shape=(NN, NN))
        
        # Right hand side
        # Minus T0 phiKim.
        c2d = mesh2.cell_to_dof(4*m2*nx, 4*m2*ny)
        phiKim0 = phiKim0[c2d] # (NC, LDF, 6)
        phiKim0 = phiKim0.reshape(m2*nx,4,m2*ny,4,4,6)
        phiKim0 = phiKim0.swapaxes(1, 2) # (m2*nx,m2*ny,r,r,LDF,6)   
        phiKim0 = phiKim0.reshape(-1, 16, 4, 6) # (m2*nx*m2*ny, r*r, LDF, 6)
        S = mesh2.cell_stiff_matrix_varphi() # Only the same in 2d case.
        I = self.linear_intepolation_quadmesh(4) # (LDC, r*r, LDF)
        I = np.einsum('l, lrjm, ij, tri -> ltm', 
                      k2.flat, phiKim0, -S, I) # (m2*nx*m2*ny, LDC, 6)
        c2d = mesh2.cell_to_dof(m2*nx, m2*ny)   
        NC = c2d.shape[0]
        S = np.zeros((A.shape[0], 6))
        for i in range(6):
            np.add.at(S[:, i], c2d, I[:, :, i])
        J = np.c_[np.full(NC, 1/64), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 1/64
        J = J.reshape(m2, nx, m2, ny, 2)
        phiKim0 = phiKim0.reshape(m2, nx, m2, ny, 16, 4, 6)
        J0 = np.sum(psi2==0, axis=(1, 3)) # (m2, m2)
        J0 = np.stack((J0, (nx*ny-J0))) # (2, m2, m2)
        J0 = 1 / J0                     # (2, m2, m2)
        I = np.einsum('mxnyrli, mxnyj, jmn -> jmni', phiKim0, J, J0) # (2, m2, m2, 6)
        S[-2*m2**2:] = I.reshape(-1, 6)
        
        # Original part.
        S[-2*m2**2:-m2**2, 0] -= 1
        S[-m2**2:, 1] -= 1
        I = np.mgrid[w2[0]:w2[1]:complex(0,m2*nx+1),
                     w2[2]:w2[3]:complex(0,m2*ny+1)] # (GD, NX+1, NY+1)
        I = I.reshape(2, -1)
        I = I[:, c2d] # (GD, NC, 4)
        I = I.reshape(2, m2, nx, m2, ny, 4)
        J = np.c_[np.full(NC, 0.25), np.zeros(NC)] # (NC, 2)
        J[I0, 0] = 0
        J[I0, 1] = 0.25
        J = J.reshape(m2, nx, m2, ny, 2)
        I = np.einsum('tmpnql, mpnqi, imn -> mnit', I, J, J0) # (m2, m2, 2, GD)
        I = I.reshape(m2**2, 4)
        I -= I[I.shape[0]//2]
        S[-2*m2**2:-m2**2, 2:4] -= I[:, :2]
        S[-m2**2:, 4:] -= I[:, 2:]
                
        # Boudary condition
        phi = mesh2.set_zero_dirichlet(A, S)
        
        # Macroscopic coefficient
        # Constraint to RVE
        I = mesh2.ada(m*nx*(m2*ny+1)+m*ny, nx+1, ny+1, m2*ny+1)
        phi = phi[I] # (NN, 6)
        c2d = mesh2.cell_to_dof(nx, ny)
        phi = phi[c2d] # (nx*ny, LDC, 6)
        phi = phi.reshape(nx, ny, 4, 6)
        I = self.linear_intepolation_quadmesh(4) # (LDC, r*r, LDF)
        phi = np.einsum('xyci, crf -> xyrfi', phi, I)
        # (m2, nx, m2, ny, 16, 4, 6) -> (nx, ny, r*r, LDF, 6)
        phiKim0 = phiKim0[m, :, m] # (nx, ny, r*r, LDF, 6)
        phi = phi + phiKim0
        phi = phi.reshape(-1, 16, 4, 6)
        
        A = mesh2.cell_stiff_matrix_varphi()
        Bs = np.einsum('lrmi, lrnj, mn, l -> ij', phi, phi, A, k2[m,:,m].flat)
        bs = np.einsum('lrmi, l, m -> i', phi[:,:,:,:2], f.flat, np.full(4, mesh2.cellm/64))
        
        return Bs[:2,:2], Bs[2:,2:], bs
        
    
class HierarchicalMultiHomogenization(MicroPhi):
    
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
    
    def reshape(self, a, xr, yr, I):
        # I, ()
        a = a.swapaxes(1, 2)
        a = a.reshape(-1, self.nxf, self.nyf)
        
        a = a[I] # (xr*yr, (2m+2^{L-1})^2, NXF, NYF)
        a = a.reshape(xr*yr, 2*self.m+2**(self.L-1), 2*self.m+2**(self.L-1), self.nxf, self.nyf)
        a = a.swapaxes(2, 3) # (XR*YR, 2m+2^{L-1}, NXF, 2m+2^{L-1}, NYF)
        return a
    
    def solve(self, ks, Psis, fs, way='m'):
        # k, (nxc+2m, nxf, nyc+2m, nyf)
        E = 2**(self.L-1)
        xr = self.nxc // E
        yr = self.nyc // E
        
        fs = fs.reshape(xr, E, self.nxf, yr, E, self.nyf)
        fs = fs.transpose(0, 3, 1, 4, 2, 5) # (XR, YR, 2^{L-1}, 2^{L-1}, NXF, NYF)
        fs = fs.reshape(-1, E**2, self.nxf, self.nyf)
        
        I = np.arange((self.nxc+2*self.m)*(self.nyc+2*self.m))
        I = I.reshape(self.nxc+2*self.m, self.nyc+2*self.m)
        J = I[:2*self.m+2**(self.L-1), :2*self.m+2**(self.L-1)]
        I = I[:self.nxc:2**(self.L-1), :self.nyc:2**(self.L-1)]
        I = I.reshape(-1)
        I = I[:, np.newaxis] + J.reshape(-1) # (xr*yr, (2m+2^{L-1})^2)
        
        ks = self.reshape(ks, xr, yr, I)
        Psis = self.reshape(Psis, xr, yr, I)
        
        if way == 'm':
            pool = mp.Pool(processes=self.nc)
            local = deepcopy(self)
            args = list(zip(range(self.ncc), ks, Psis, fs))
            if self.L == 2:
                res = pool.starmap(local.phi_for_Kim_L2, args)
            elif self.L == 3:
                res = pool.starmap(local.phi_for_Kim_L3, args)
            Rs, Ds, Fs = map(list, zip(*res))
            Rs = np.array(Rs)
            Ds = np.array(Ds)
            Fs = np.array(Fs)
        else:
            Rs = np.zeros((xr*yr, E**2, 2, 2))
            Ds = np.zeros((xr*yr, E**2, 4, 4))
            Fs = np.zeros((xr*yr, E**2, 2))
            for i in range(xr*yr):
                if self.L == 2:
                    Rs[i], Ds[i], Fs[i] = self.phi_for_Kim_L2(i, ks[i], Psis[i], fs[i])
                elif self.L == 3:
                    Rs[i], Ds[i], Fs[i] = self.phi_for_Kim_L3(i, ks[i], Psis[i], fs[i])
        
        Rs = Rs.reshape(xr, yr, E, E, 2, 2)
        Rs = Rs.swapaxes(1, 2)
        Rs = Rs.reshape(self.ncc, 2, 2)
        Ds = Ds.reshape(xr, yr, E, E, 2, 2, 2, 2)
        Ds = Ds.swapaxes(1, 2)
        Ds = Ds.reshape(self.ncc, 2, 2, 2, 2)
        Fs = Fs.reshape(xr, yr, E, E, 2)
        Fs = Fs.swapaxes(1, 2)
        Fs = Fs.reshape(self.ncc, 2)
        # return Rs, Ds, Fs
        meshc = MHMacroMesh(self.Omega, self.nxc, self.nyc)
        UH, UHa = meshc.solve(Rs, Ds, Fs)
        return UH, UHa

