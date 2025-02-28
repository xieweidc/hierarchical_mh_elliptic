import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

n = 6
N = 4 * n
# N = 8 * n + 1
uh = np.zeros((N, N))
uh[::2, ::2] = 1
uh[2::4, 2::4] = 2
# uh[::8, ::8] = 3
box = np.array([0, 1, 0, 1])

# print("N: ", N)
# print("1/h: ", (N+1)*(4*r+1))

# 定义颜色映射
cmap = ListedColormap(['white', 'yellow', 'green'])
# cmap = ListedColormap(['white', 'pink', 'green', 'blue'])

plt.figure()
plt.imshow(uh, extent=box, origin='lower', cmap=cmap)
# plt.imshow(uh, extent=box, origin='lower', cmap='YlGn')

# 在图像中画出网格线
for i in range(1+N):
    plt.plot([0, 1], [i/N, i/N], color='black')
    plt.plot([i/N, i/N], [0, 1], color='black')

# 关闭坐标轴刻度
plt.xticks([])
plt.yticks([])

plt.savefig('hierarchy_grid_color.eps')
# plt.savefig('hierarchy_grid_color.png')
plt.show()

