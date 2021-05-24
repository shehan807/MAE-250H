"""
Created on May 2 2021
@author: Shehan M. Parmar
Initialize pointer arrays to ease coding of 
velocity and pressure variables matrices. 
"""
import numpy as np
from get_global import *

def init():
    
    get_global('inputs.txt') 
    
    u = np.zeros((nx-1, ny), dtype=np.int)
    v = np.zeros((nx, ny-1), dtype=np.int)
    p = np.zeros((nx, ny), dtype=np.int)
    
    # Create pointers for velocity, u, v
    ind = 0
    for j in range(0,ny):
        for i in range(0,nx-1):
            u[i,j] = ind
            ind += 1
    for j in range(0,ny-1):
        for i in range(0,nx):
            v[i,j] = ind
            ind += 1
    if ind != ((nx-1)*ny + nx*(ny-1)):
        raise IndexError('wrong velocity size')
    
    # create points for pressure, p
    ind = 0
    for j in range(0,ny):
        for i in range(0,nx):
            if (i==0) and (j==0):
                if pinned: 
                    pass # skip pinned pressure
                else:
                    p[i,j] = ind
                    ind += 1
            else:
                p[i,j] = ind
                ind += 1
    if ind != (nx*ny-1):
        raise IndexError('wrong pressure index')
