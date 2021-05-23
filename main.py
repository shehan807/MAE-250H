"""
Created on May 2 2021
@author: Shehan M. Parmar
Main Navier-Stokes solver
"""
global nx, ny, u, v, p, dx, dy, q_size, p_size  

from numpy import linalg as LA

from init import *
import operators as op

xu = np.arange(dx, Lx, dx)
yu = np.arange(0.5*dy, Ly, dy)
print(xu, yu)
Xu, Yu = np.meshgrid(xu, yu)
Zxu = Yu*np.cos(Xu*Yu)
grad_x_ex = np.reshape(Zxu, (1, ny*(nx-1)))

xv = np.arange(0.5*dx, Lx, dx)
yv = np.arange(dy, Ly, dy)
Xv, Yv = np.meshgrid(xv, yv)
Zyv = Xv*np.cos(Xv*Yv)
grad_y_ex = np.reshape(Zyv, (1, nx*(ny-1)))

xp = np.arange(0.5*dx, Lx, dx)
yp = np.arange(0.5*dy, Ly, dy)
Xp, Yp = np.meshgrid(xp, yp)
Zp = np.sin(Xp*Yp)

print()


grad_ex = np.concatenate((grad_x_ex, grad_y_ex), axis=1)
g_test = np.reshape(Zp, (1,nx*ny))
g_test = g_test[0, 1:] # exclude pinned value
q = op.grad(g_test)
print(q)
print(grad_ex)
print(LA.norm(q-grad_ex))

