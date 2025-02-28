import numpy as np
import matplotlib.pyplot as plt
import random
import math

SIN = np.sin
COS = np.cos
EXP = np.exp
PI = np.pi

# 创建一个 (N, N) 的空数组，初始化为 0
# NT = 1
NT = 1e5
tol = 1
t = 0

# while tol == 1:
#     # N = 256
#     N = 32
#     n = 8
#     ntol = max(1, int((N/n)**2/10))
#     array = np.zeros((N, N))
    
#     # # 设置裂缝的数量和长度
#     # num_cracks = 1500
#     # line_length = 20  # 每条裂缝的长度
    
#     # ### N=256, n=32
#     # # Case 1
#     # num_cracks = 3600
#     # line_length = 5  # 每条裂缝的长度
#     # # # Case 2
#     # # num_cracks = 1500
#     # # line_length = 20  # 每条裂缝的长度
    
#     # ### Case 3, N=128, n=16
#     # num_cracks = 1200
#     # line_length = 5  # 每条裂缝的长度
    
#     ### Case 4, N=32, n=18
#     num_cracks = 100
#     line_length = 5  # 每条裂缝的长度
    
    
    
#     # T = max(int(N / n / 2), 1)
#     T = 1
#     width = 1  # 裂缝的宽度
    
#     # 生成随机的直线裂缝
#     for _ in range(num_cracks):
#         # 随机选择直线的起点
#         x_start, y_start = random.randint(T, N-T), random.randint(T, N-T)
        
#         # 随机选择直线的角度（0 到 360 度）
#         angle = random.uniform(0, 2 * math.pi)
        
#         # 计算直线的终点（根据长度和角度）
#         x_end = int(x_start + line_length * math.cos(angle))
#         y_end = int(y_start + line_length * math.sin(angle))
        
#         # 确保终点在图像范围内
#         x_end = max(T, min(N-T, x_end))  # 这里使用 N-1
#         y_end = max(T, min(N-T, y_end))  # 这里使用 N-1
        
#         # 使用 Bresenham 算法绘制直线
#         x, y = x_start, y_start
#         dx = abs(x_end - x_start)
#         dy = abs(y_end - y_start)
#         sx = 1 if x_start < x_end else -1
#         sy = 1 if y_start < y_end else -1
#         err = dx - dy
    
#         while True:
#             # 标记主直线位置
#             array[x, y] = 1
            
#             # 标记宽度范围内的像素
#             for w in range(-width, width + 1):
#                 w = 0
#                 # 计算偏移后的坐标
#                 new_x = x + w * sy  # 垂直于线段的方向
#                 new_y = y + w * sx  # 垂直于线段的方向
                
#                 # 确保新坐标在范围内
#                 if 0 <= new_x < N and 0 <= new_y < N:
#                     array[new_x, new_y] = 1
    
#             if x == x_end and y == y_end:
#                 break
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 x += sx
#             if e2 < dx:
#                 err += dx
#                 y += sy
    
#     # array[:8, :8] = 1
#     # array = np.load('remote/npys/case1_k.npy')
#     # 可视化裂缝图案
#     plt.imshow(array, cmap='gray')
#     plt.title("Intersecting Crack Lines with Width")
#     plt.axis('off')
#     plt.show()
    
    
#     k1 = np.count_nonzero(array.reshape(n, N//n, n, N//n), (1, 3))
#     er1 = np.all(k1>=ntol)
#     er2 = np.all(k1<=int(k1.shape[1]**2-ntol))
    
#     # print("k1: ", k1)
#     if er1 & er2:
#         np.save('k.npy', array)
#         I = array.flatten()
#         I, = np.where(I!=0)
#         np.save('I.npy', I)
#         print("It's finished")
#         tol = 2
#     else:
#         I0, J0 = np.where(k1<ntol)
#         nc = I0.shape[0]
#         I0, J0 = np.where(k1>int(k1.shape[1]**2-ntol))
#         nc += I0.shape[0]
#         print("%d : %d/%d"%(t, nc, N**2))

#     t += 1
#     if t > NT:
#         tol = 2

N = 240
# N = 192
# N = N * 4
n = 24
# k = np.zeros((n, N//n, n, N//n))
# I = np.array([0, 0, 1, 1, 2, 2, 3, 3])
# J = np.array([1, 2, 0, 3, 0, 3, 1, 2])
# k[:, I, :, J] = 1
# k = k.reshape(N, N)


X, Y = np.mgrid[0:1:complex(0,N+1), 0:1:complex(0,N+1)]
X = X[1:, 1:] - 1/2/N
Y = Y[1:, 1:] - 1/2/N
eps = 1 / 12
X = X.flatten()
Y = Y.flatten()
k = np.zeros_like(X)
# Kappa 1
# I = np.where(np.abs(SIN(2*PI*X/eps))<0.4)

# I0, = np.where(np.abs(SIN(PI*X/eps)) > 0.95)
# I1, = np.where(np.abs(SIN(PI*Y/eps)) > 0.85)
# I = np.union1d(I0, I1)


# I0, = np.where(np.abs(SIN(PI*(X+0.1)/eps)) > 0.95)
# I1, = np.where(np.abs(SIN(PI*(Y+0.2)/eps)) > 0.85)
# I = np.union1d(I0, I1)

# I = np.where(np.abs(SIN(PI*(Y*(1-Y)+X)/eps)*SIN(PI*(X*(X-1)+Y)/eps))<0.4)

# I = np.where(np.abs(SIN(2*PI*((1/4+Y)*(1-Y)*(1/6+X))/eps))<0.7)
I = np.where(np.abs(SIN(2*PI*((1/2+Y)*X)/eps)*SIN(2*PI*(1/3-X)*(Y+1)/eps))<0.35)

# I = np.where(np.abs(SIN(2*PI*((0.1+Y))/eps)*SIN(2*PI*(0.2+Y+X/eps)))<0.3)

# I = np.where(SIN(2*PI*(Y-X*2)/eps)*SIN(2*PI*(X*2+Y)/eps)<0.3)
# I = np.where(np.abs(SIN(PI*((1-Y)*Y+X)/eps)*SIN(PI*(X**2+Y)/eps))<0.3)


k[I] = 1
# k = np.arange(N**2).reshape(N, N)
k = k.reshape(N, N)
plt.axis('off')
plt.xticks(N//n*np.arange(n+1), np.arange(n+1))
plt.yticks(N//n*np.arange(n+1), np.arange(n+1))
# plt.imshow(k)
plt.grid(True)
plt.imshow(k.T)
# plt.imshow((np.arange(9).reshape(3,3)).T)
plt.show()

k = k.reshape(n, N//n, n, N//n)
for i in range(n):
    for j in range(n):
        er = (np.max(k[i,:,j])<0.9) | (np.min(k[i,:,j])>0.2)
        if er:
            print("It will be singular")
            print(np.max(k[i,:,j]), np.min(k[i,:,j]))
            break



# X, Y = np.mgrid[0:1:complex(0,n+1), 0:1:complex(0,n+1)]
# X = X[1:, 1:] - 1/2/n
# Y = Y[1:, 1:] - 1/2/n
# f = 5*PI**2 * SIN(2*PI*X) * SIN(PI*Y)
# # f = f.reshape(n, n)
# # f = np.zeros((n, n))
# # f[0, 0] = 1
# # f[-1, -1] = -1
# f = np.broadcast_to(f[:,None,:,None], (n,N//n,n,N//n))
# f = f.reshape(N, N)
# plt.imshow(f.T, extent=np.array([0, 1, 0, 1]))
# # plt.xticks(N//n*np.arange(n+1), np.arange(n+1))
# # plt.yticks(N//n*np.arange(n+1), np.arange(n+1))
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.show()

# np.save('f.npy', f)

# np.save('k.npy', k)
# I = k.flatten()
# I, = np.where(I!=0)
# np.save('I.npy', I)


