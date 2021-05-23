"""
Created on May 2 2021
@author: Shehan M. Parmar
Discrete operators for Navier-Stokes solver. 
"""
import numpy as np

from init import *

def grad(g): # Gradient Operator
    
    q = np.zeros(q_size)
    
    # Be careful with p(0,0) for the pinned pressure location 
    
    # compute x-dir gradient, u
    for j in [0]:
        for i in [0]:
            q[u[i,j]] = (g[p[i+1,j]]            )/dx       # - g[p[0,0]]/dx = 0
        for i in range(1,nx-1):
            q[u[i,j]] = (g[p[i+1,j]] - g[p[i,j]])/dx       # 
    for j in range(1,ny):
        for i in range(0,nx-1):
            q[u[i,j]] = (g[p[i+1,j]] - g[p[i,j]])/dx       # 

    # compute y-dir gradient, v
    for j in [0]:
        for i in [0]:
            q[v[i,j]] = (g[p[i,j+1]]            )/dy       # - g[p[0,0]]/dy = 0
        for i in range(1,nx):
            q[v[i,j]] = (g[p[i,j+1]] - g[p[i,j]])/dy       # 
    for j in range(1,ny-1):
        for i in range(0,nx):
            q[v[i,j]] = (g[p[i,j+1]] - g[p[i,j]])/dy       # 


    return q
