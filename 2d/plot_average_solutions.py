import numpy as np
from math import ceil
import matplotlib.pyplot as plt


def plot_average(Z, fname):
    
    plt.figure()
    plt.imshow(Z.T, extent=Omega, origin='lower', cmap='viridis')
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=18)  # 刻度字体大小
    colorbar.ax.yaxis.offsetText.set_fontsize(18)  # 修改科学计数法的字体大小
    plt.axis('off')

    plt.savefig(adf+fname)
    
    if show:
        plt.show()
    else:
        plt.close()
            
def plot_pointwise(Z, fname):
    nx, ny = Z.shape
    X = np.linspace(Omega[0], Omega[1], nx)
    Y = np.linspace(Omega[2], Omega[3], ny)
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(X, Y, Z.T, shading='gouraud', vmin=Z.min(), vmax=Z.max())
    colorbar = fig.colorbar(pc)
    colorbar.ax.tick_params(labelsize=18)  # 刻度字体大小
    colorbar.ax.yaxis.offsetText.set_fontsize(18)  # 修改科学计数法的字体大小
    plt.axis('off')
    plt.gca().set_aspect('equal', 'box')
    
    plt.savefig(adf+fname)
    
    if show:
        plt.show()
    else:
        plt.close()
            
pn = 1

NX = NY = 240
nxc = nyc = 24
m = ceil(2*np.log(nxc))

# NX = NY = 96
# nxc = nyc = 4
# m = 3

ad = 'res/ex%d_'%pn
adf = 'figs/ex%d_'%pn
show = True

Omega = np.array([0, 1, 0, 1])

# # Plot coefficient
# k = np.load(ad+'k.npy')
# plot_average(k, 'k.pdf')

# Plot fine-grid solution
u = np.load(ad+'u.npy')
u = u.reshape(NX*4+1, NY*4+1)
plot_pointwise(u, 'u.pdf')

# Plot fine-grid average
ua = np.load(ad+'%d_%d_ua.npy'%(nxc, m))
plot_average(ua[0], 'ua1.pdf')
plot_average(ua[1], 'ua2.pdf')

# Plot multicontinuum homogenization average
ua = np.load(ad+'%d_%d_uh_mha.npy'%(nxc, m))
plot_average(ua[0], 'mha1.pdf')
plot_average(ua[1], 'mha2.pdf')

# Plot hierarchical multicontinuum homogenization average
ua = np.load(ad+'%d_%d_uh_hmha.npy'%(nxc, m))
plot_average(ua[0], 'hmha1.pdf')
plot_average(ua[1], 'hmha2.pdf')




