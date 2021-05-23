"""
Created on May 2 2021
@author: Shehan M. Parmar
Initialize pointer arrays to ease coding of 
velocity and pressure variables matrices. 
"""
def init():
    
    global nx, ny, u, v, p, dx, dy, nq, np

    u = np.zeros(nx-1, ny)
    v = np.zeros(nx, ny-1)
    p = np.zeros(nx, ny)

    # Create pointers for velocity, u, v
    ind = 0 
    for j in range(0,ny):
        for i in range(0,nx):
            u[i,j] = ind
            ind += 1
    for j in range(0,ny-1):
        for i in range(1,nx):
            v[i,j] = ind
            ind += 1
    if ind != ((nx-1)*ny + nx*(ny-1)):
        raise IndexError('wrong velocity size')
    # create points for pressure, p
    ind = 0
    for j in range(0,ny):
        for i in range(0,nx):
            if (i==1) and (j==1):
                pass # skip pinned pressure
            else:
                p[i,j] = ind
                ind += 1

    if ind != (nx*ny-1):
        raise IndexError('wrong pressure index')



