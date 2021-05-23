"""
Created on May 2 2021
@author: Shehan M. Parmar
Main Navier-Stokes solver
"""
global nx, ny, u, v, p, dx, dy, q_size, p_size  

from numpy import linalg as LA
from init import *
import operators as op

# Occasionally, np.arange will include the "stop" value due to rounding/floating
# point error, so a small corrector term (relative to grid spacing) ensures 
# arrays have correct length
corrX = 1e-6*dx
corrY = 1e-6*dy

xu = np.arange(dx, Lx-corrX, dx)
yu = np.arange(0.5*dy, Ly-corrY, dy)
Xu, Yu = np.meshgrid(xu, yu)
Zxu = Yu*np.cos(Xu*Yu)
grad_x_ex = np.reshape(Zxu, (1, ny*(nx-1)))

xv = np.arange(0.5*dx, Lx-corrX, dx)
yv = np.arange(dy, Ly-corrY, dy)
Xv, Yv = np.meshgrid(xv, yv)
Zyv = Xv*np.cos(Xv*Yv)
grad_y_ex = np.reshape(Zyv, (1, nx*(ny-1)))

xp = np.arange(0.5*dx, Lx-corrX, dx)
yp = np.arange(0.5*dy, Ly-corrY, dy)
Xp, Yp = np.meshgrid(xp, yp)
Zp = np.sin(Xp*Yp)

grad_ex = np.concatenate((grad_x_ex, grad_y_ex), axis=1)
g_test = np.reshape(Zp, (1,nx*ny))
g_test = g_test[0, 1:] # exclude pinned value
q = op.grad(g_test, pinned=True)

err = q-grad_ex
L2 = LA.norm(err) / len(err[0])
Linf = LA.norm(err, ord=np.inf)
print(err)
print(dx, dy, dx*dy, L2, Linf)
