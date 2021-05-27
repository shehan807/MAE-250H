"""
Created on May 2 2021
@author: Shehan M. Parmar
Discrete operators for Navier-Stokes solver. 
"""
import numpy as np
#from numba import jit

def grad(g, u, v, p, dx, dy, nx, ny, q_size, pinned = True): # Gradient Operator
    
    q = np.zeros(q_size)
    
    # Be careful with p(0,0) for the pinned pressure location 
    
    # compute x-dir gradient, u
    for j in [0]:
        for i in [0]:
            if pinned:
                q[u[i,j]] = (g[p[i+1,j]]            )/dx       # - g[p[0,0]]/dx = 0
            else: 
                q[u[i,j]] = (g[p[i+1,j]] - g[p[0,0]])/dx       #  
        for i in range(1,nx-1):
            q[u[i,j]] = (g[p[i+1,j]] - g[p[i,j]])/dx       # 
    for j in range(1,ny):
        for i in range(0,nx-1):
            q[u[i,j]] = (g[p[i+1,j]] - g[p[i,j]])/dx       # 

    # compute y-dir gradient, v
    for j in [0]:
        for i in [0]:
            if pinned:
                q[v[i,j]] = (g[p[i,j+1]]            )/dy       # - g[p[0,0]]/dy = 0
            else: 
                q[v[i,j]] = (g[p[i,j+1]] - g[p[0,0]])/dy       #  
        for i in range(1,nx):
            q[v[i,j]] = (g[p[i,j+1]] - g[p[i,j]])/dy       # 
    for j in range(1,ny-1):
        for i in range(0,nx):
            q[v[i,j]] = (g[p[i,j+1]] - g[p[i,j]])/dy       # 


    return q

def div(q, u, v, p, dx, dy, nx, ny, p_size, pinned=True): # Divergence Operator
    
    if pinned:
        g = np.zeros(p_size)
    elif not pinned: 
        g = np.zeros(p_size+1)

    # Bottom Row of Grid
    for j in [0]:
        for i in range(1,nx-1):
            g[p[i,j]] = ( q[u[i,j]] - q[u[i-1, j]])/dx \
                      + ( q[v[i,j]]               )/dy      
                      #             - q[v[i,j-1]]  /dy
    # Bottom Right 
    for j in [0]:
        for i in [nx-1]:
            g[p[i,j]] = (           - q[u[i-1,j]])/dx  \
                      + ( q[v[i,j]]              )/dy       
                      #   q[u[i,j]]               /dx   
                      #             - q[v[i,j-1]] /dy
    # Left Wall
    for j in range(1, ny-1):
        for i in [0]:
            g[p[i,j]] = (           - q[u[i-1,j]])/dx \
                      + ( q[v[i,j]] - q[v[i,j-1]])/dy
                      #   q[u[i,j]]               /dx
    # Right Wall 
    for j in range(1,ny-1):
        for i in [nx-1]:
            g[p[i,j]] = (           - q[u[i-1,j]])/dx \
                      + ( q[v[i,j]] - q[v[i,j-1]])/dy
                      #   q[u[i,j]]               /dx
    # Top Wall 
    for j in [ny-1]:
        for i in range(1,nx-1):
            g[p[i,j]] = ( q[u[i,j]] - q[u[i-1,j]])/dx \
                      + (           - q[v[i,j-1]])/dy
                      #   q[v[i,j]]               /dy
    # Top Left Corner 
    for j in [ny-1]:
        for i in [0]:
            g[p[i,j]] = ( q[u[i,j]]              )/dx \
                      + (           - q[v[i,j-1]])/dy
                      #             - q[u[i-1,j]] /dx
                      #   q[v[i,j]]               /dy
    # Top Right Corner 
    for j in [ny-1]:
        for i in [nx-1]:
            g[p[i,j]] = (           - q[u[i-1,j]])/dx \
                      + (           - q[v[i,j-1]])/dy
                      #   q[u[i,j]]               /dx
                      #   q[v[i,j]]               /dy
    # Interior Points 
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            g[p[i,j]] = ( q[u[i,j]] - q[u[i-1,j]])/dx \
                      + ( q[v[i,j]] - q[v[i,j-1]])/dy
    return g

def bcdiv(qbc, u, v, p, dx, dy, nx, ny, p_size):
    """
    INPUTS: 
    ------
    qbc - dictionary with 8 keys (u and v 
    boundary conditions for each wall)
    """
    bcD = np.zeros(p_size)

    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]


    # Bottom
    for j in [0]:
        for i in range(1, nx-1):
            bcD[p[i,j]] = - vB[i]/dy
    # Bottom Right
    for j in [0]:
        for i in [nx-1]:
            bcD[p[i,j]] = uR[j]/dx - vB[i]/dy
    # Left Wall 
    for j in range(1,ny-1):
        for i in [0]:
            bcD[p[i,j]] = - uL[j]/dx
    # Right Wall
    for j in range(1, ny-1):
        for i in [nx-1]:
            bcD[p[i,j]] = uR[j]/dx
    # Top Wall 
    for j in [ny-1]:
        for i in range(1,nx-1):
            bcD[p[i,j]] = vT[i]/dy
    # Top Left Corner
    for j in [ny-1]:
        for i in [0]:
            bcD[p[i,j]] = -uL[j]/dx + vT[i]/dy
    # Top Right Corner 
    for j in [ny-1]:
        for i in [nx-1]:
            bcD[p[i,j]] = uR[j]/dx + vT[i]/dy
    # Interior Points (Zeroed to match q dimensions 
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            bcD[p[i,j]] = 0
    
    return bcD
