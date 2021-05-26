"""
Created on May 2 2021
@author: Shehan M. Parmar
Initialize pointer arrays to ease coding of 
velocity and pressure variables matrices. 
"""
import numpy as np

def init(nx, ny, pinned = True):
    
    u = np.ndarray((nx-1, ny), dtype=object)
    v = np.ndarray((nx, ny-1), dtype=object)
    p = np.ndarray((nx, ny)  , dtype=object)
    
    # Create pointers for velocity, u, v
    ind = int(0)
    for j in range(0,ny):
        for i in range(0,nx-1):
            u[i,j] = int(ind)
            ind += 1
    for j in range(0,ny-1):
        for i in range(0,nx):
            v[i,j] = int(ind)
            ind += 1
    if ind != ((nx-1)*ny + nx*(ny-1)):
        raise IndexError('wrong velocity size')

    # create points for pressure, p
    ind = 0
    for j in range(0,ny):
        for i in range(0,nx):
            if (i==0) and (j==0):
                if pinned: 
                    #p[i,j] = None 
                    pass # skip pinned pressure
                else:
                    p[i,j] = int(ind)
                    ind += 1
            else:
                p[i,j] = int(ind)
                ind += 1
    
    if ind != (nx*ny-1):
        if pinned:
            raise IndexError('wrong pressure index (pinned)')
        elif not pinned and (ind != (nx*ny)):
            raise IndexError('wrong pressure index (not pinned)')
    return u, v, p
