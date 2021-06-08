"""
Created on May 2 2021
@author: Shehan M. Parmar
Discrete operators for Navier-Stokes solver. 
"""
import numpy as np
from numpy import linalg as LA
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
            g[p[i,j]] = ( q[u[i,j]]              )/dx \
                      + ( q[v[i,j]] - q[v[i,j-1]])/dy
                      #             - q[u[i-1,j]] /dx
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

def bcdiv(qbc, u, v, p, dx, dy, nx, ny, p_size, pinned=True):
    """
    INPUTS: 
    ------
    qbc - dictionary with 8 keys (u and v 
    boundary conditions for each wall)
    """
    if pinned:
        bcD = np.zeros(p_size)
    elif not pinned:
        bcD = np.zeros(p_size+1)


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
    #for j in range(1,ny-1):
    #    for i in range(1,nx-1):
    #        bcD[p[i,j]] = 0
    
    return bcD

def laplace(q, u, v, p, dx, dy, nx, ny, q_size, pinned=True):
    
    Lq = np.zeros(q_size)

    # NOTE: coeff. = 3 are for ghost cell terms (e.g. (2*uBC - 3*u[i,1] + u[i,2]) / dy^2
    # U-COMPONENT
    # Bottom Row
    for j in [0]:
        for i in [0]:
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]]               ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]]               ) / dy**2
                       #                             + q[u[i-1,j]]   / dx**2
                       #                             + q[u[i,j-1]]   / dy**2
        for i in range(1,nx-2):
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]]               ) / dy**2
                       #                             + q[u[i,j-1]]   / dy**2
        for i in [nx-2]:
            Lq[u[i,j]] = (             - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]]               ) / dy**2
                       #   q[u[i+1,j]]                               / dx**2
                       #                             + q[u[i,j-1]]   / dy**2 
    # Top Row
    for j in [ny-1]:
        for i in [0]:
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]]               ) / dx**2 \
                       + (             - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
                       #                             + q[u[i-1,j]]   / dx**2
                       #   q[u[i,j+1]]                               / dy**2
        for i in range(1,nx-2):
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + (             - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
                       #   q[u[i,j+1]]                               / dy**2
        for i in [nx-2]:
            Lq[u[i,j]] = (             - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + (             - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
                       #   q[u[i+1,j]]                               / dx**2
                       #   q[u[i,j+1]]                               / dy**2 

    # Interior Points
    for j in range(1,ny-1):
        for i in [0]:
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]]               ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
                       #                             + q[u[i-1,j]]   / dx**2
        for i in range(1,nx-2):
            Lq[u[i,j]] = ( q[u[i+1,j]] - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
        for i in [nx-2]:
            Lq[u[i,j]] = (             - 2*q[u[i,j]] + q[u[i-1,j]] ) / dx**2 \
                       + ( q[u[i,j+1]] - 2*q[u[i,j]] + q[u[i,j-1]] ) / dy**2
                       #   q[u[i+1,j]]                               / dx**2
    
    # V-COMPONENT

    # Bottom Row
    for j in [0]:
        for i in [0]:
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]]               ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]]               ) / dy**2
                       #                             + q[v[i-1,j]]   / dx**2
                       #                             + q[v[i,j-1]]   / dy**2
        for i in range(1,nx-1):
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]]               ) / dy**2
                       #                             + q[v[i,j-1]]   / dy**2
        for i in [nx-1]:
            Lq[v[i,j]] = (             - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]]               ) / dy**2
                       #   q[v[i+1,j]]                               / dx**2
                       #                             + q[v[i,j-1]]   / dy**2 
    # Top Row
    for j in [ny-2]:
        for i in [0]:
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]]               ) / dx**2 \
                       + (             - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
                       #                             + q[v[i-1,j]]   / dx**2
                       #   q[v[i,j+1]]                               / dy**2
        for i in range(1,nx-1):
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + (             - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
                       #   q[v[i,j+1]]                               / dy**2
        for i in [nx-1]:
            Lq[v[i,j]] = (             - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + (             - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
                       #   q[v[i+1,j]]                               / dx**2
                       #   q[v[i,j+1]]                               / dy**2 
    # Interior Points
    for j in range(1,ny-2):
        for i in [0]:
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]]               ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
                       #                             + q[v[i-1,j]]   / dx**2
        for i in range(1,nx-1):
            Lq[v[i,j]] = ( q[v[i+1,j]] - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
        for i in [nx-1]:
            Lq[v[i,j]] = (             - 2*q[v[i,j]] + q[v[i-1,j]] ) / dx**2 \
                       + ( q[v[i,j+1]] - 2*q[v[i,j]] + q[v[i,j-1]] ) / dy**2
                       #   q[v[i+1,j]]                               / dx**2


    return Lq

def bclap(q, qbc, u, v, p, dx, dy, nx, ny, q_size, pinned=True):
    
    bcL = np.zeros(q_size)
    
    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]
    
    # U-COMPONENT

    # Bottom Row 
    for j in [0]:
        # BC + Ghost Cell
        for i in [0]:
            
            uB_ghost2 = (2*uB[i] - q[u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[u[i,j]] + q[u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[u[i,j]] + 5*q[u[i,j+1]] - q[u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uL[j] / dx**2 +  uB_ghost4 / dy**2
        
        # Ghost Cell
        for i in range(1,nx-2):
            
            uB_ghost2 = (2*uB[i] - q[u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[u[i,j]] + q[u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[u[i,j]] + 5*q[u[i,j+1]] - q[u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uB_ghost4 / dy**2
        
        # BC + Ghost Cell
        for i in [nx-2]:
            
            uB_ghost2 = (2*uB[i] - q[u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[u[i,j]] + q[u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[u[i,j]] + 5*q[u[i,j+1]] - q[u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uR[j] / dx**2 + uB_ghost4 / dy**2
    
    # Top Row
    for j in [ny-1]:
        # BC + Ghost Cell
        for i in [0]:
            
            uT_ghost2 = (2*uT[i] - q[u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[u[i,j]] + q[u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[u[i,j]] + 5*q[u[i,j-1]] - q[u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uL[j] / dx**2 + uT_ghost4 / dy**2
        # Ghost Cell
        for i in range(1,nx-2):
            
            uT_ghost2 = (2*uT[i] - q[u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[u[i,j]] + q[u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[u[i,j]] + 5*q[u[i,j-1]] - q[u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uT_ghost4 / dy**2
        # BC + Ghost Cell
        for i in [nx-2]:
            
            uT_ghost2 = (2*uT[i] - q[u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[u[i,j]] + q[u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[u[i,j]] + 5*q[u[i,j-1]] - q[u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[u[i,j]] = uR[j] / dx**2 + uT_ghost4 / dy**2
    
    # Interior Nodes (DONE)
    for j in range(1,ny-1):
        # BC
        for i in [0]:
            bcL[u[i,j]] = uL[j] / dx**2;
        for i in range(1,nx-2):
            bcL[u[i,j]] = 0
        # BC
        for i in [nx-2]:
            bcL[u[i,j]] = uR[j] / dx**2; 
    
    # V-COMPONENT

    # Bottom Row 
    for j in [0]:
        # BC + Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[v[i,j]] + q[v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[v[i,j]] + 5*q[v[i+1,j]] - q[v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] = vL_ghost4 / dx**2 + vB[i] / dy**2;
        # BC
        for i in range(1,nx-1):
            bcL[v[i,j]] = vB[i] / dy**2;
        # BC + Ghost Cell
        for i in [nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[v[i,j]] + q[v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[v[i,j]] + 5*q[v[i-1,j]] - q[v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] = vR_ghost4 / dx**2 + vB[i] / dy**2;
    
    # Top Row 
    for j in [ny-2]:
        # BC + Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[v[i,j]] + q[v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[v[i,j]] + 5*q[v[i+1,j]] - q[v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] = vL_ghost4  / dx**2 + vT[i] / dy**2;
        # BC
        for i in range(1,nx-1):
            bcL[v[i,j]] = vT[i] / dy**2
        # BC + Ghost Cell
        for i in [nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[v[i,j]] + q[v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[v[i,j]] + 5*q[v[i-1,j]] - q[v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] = vR_ghost4  / dx**2 + vT[i] / dy**2;
    
    # Interior Nodes
    for j in range(1,ny-2):
        # Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[v[i,j]] + q[v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[v[i,j]] + 5*q[v[i+1,j]] - q[v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] =  vL_ghost4 / dx**2;
        
        for i in range(1,nx-1):
            bcL[v[i,j]] =  0
        # Ghost Cell
        for i in [nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[v[i,j]] + q[v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[v[i,j]] + 5*q[v[i-1,j]] - q[v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] =  vR_ghost4 / dx**2;

    return bcL

def adv(q, qbc, u, v, p, dx, dy, nx, ny, q_size, pinned=True):
    
    advq = np.zeros(q_size)
    
    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]

    # Nx(i,j) -> u
    # Interpolation Operations, _uy_vx (cell vertices) and _ux_ux (cell centers)
    # Difference Operations, del_x, del_y
    for j in range(0, ny):
        for i in range(0, nx-1): # Interior
            
            if i == 0: # Left Wall
                _ux_ux_ = -(0.5*(uL[j]       + q[u[i,j]]))**2  \
                        +  (0.5*(q[u[i,j]]   + q[u[i+1,j]]))**2 
            elif i == nx-2: # Right Wall    
                _ux_ux_ = -(0.5*(q[u[i-1,j]] + q[u[i,j]]))**2  \
                        +  (0.5*(q[u[i,j]]   + uR[j]))**2 
            else: # Interior
                _ux_ux_ = -(0.5*(q[u[i-1,j]] + q[u[i,j]]))**2  \
                        +  (0.5*(q[u[i,j]]   + q[u[i+1,j]]))**2 
            
            if j == 0: # Bottom Wall
                
                uB_ghost2 = 2*uB[i] - q[u[i,j]] # 2-pt stencil
                uB_ghost3 = (8*uB[i] - 6*q[u[i,j]] + q[u[i,j+1]]) / 3. # 3-pt stencil
                uB_ghost4 = (16*uB[i] - 15*q[u[i,j]] + 5*q[u[i,j+1]] - q[u[i,j+2]]) / 5. # 4-pt stencil
                
                _vx_uy_ = -0.5*(vB[i] + vB[i+1])             * 0.5*(uB_ghost4   + q[u[i,j]]) \
                        +  0.5*(q[v[i,j]] + q[v[i+1,j]])     * 0.5*(q[u[i,j]]   + q[u[i,j+1]]) 
            
            elif j == ny-1: # Top Wall
                
                uT_ghost2 = 2*uT[i] - q[u[i,j]] # 2-pt stencil
                uT_ghost3 = (8*uT[i] - 6*q[u[i,j]] + q[u[i,j-1]]) / 3. # 3-pt stencil
                uT_ghost4 = (16*uT[i] - 15*q[u[i,j]] + 5*q[u[i,j-1]] - q[u[i,j-2]]) / 5. # 4-pt stencil
                
                _vx_uy_ = -0.5*(q[v[i,j-1]] + q[v[i+1,j-1]]) * 0.5*(q[u[i,j-1]] + q[u[i,j]]) \
                        +  0.5*(vT[i] + vT[i+1])             * 0.5*(q[u[i,j]]   + uT_ghost4)
                
            else: # Interior
                _vx_uy_ = -0.5*(q[v[i,j-1]] + q[v[i+1,j-1]]) * 0.5*(q[u[i,j-1]] + q[u[i,j]]) \
                        +  0.5*(q[v[i,j]]   + q[v[i+1,j]])   * 0.5*(q[u[i,j]]   + q[u[i,j+1]]) 
            
            del_y_vx_uy = _vx_uy_ / dy
            del_x_ux_ux = _ux_ux_ / dx
            
            advq[u[i,j]] = del_x_ux_ux + del_y_vx_uy
        

    # Ny(i,j) -> v
    # Interpolation Operations, _uy_vx (cell vertices) and _vy_vy (cell centers)
    for j in range(0, ny-1):
        for i in range(0, nx):
            
            if i == 0: # Left Wall
                
                vL_ghost2 = 2*vL[j] - q[v[i,j]] # 2-pt stencil
                vL_ghost3 = (8*vL[j] - 6*q[v[i,j]] + q[v[i+1,j]]) / 3. # 3-pt stencil
                vL_ghost4 = (16*vL[j] - 15*q[v[i,j]] + 5*q[v[i+1,j]] - q[v[i+2,j]]) / 5. # 4-pt stencil
                
                _uy_vx_ = -0.5*(uL[j]       + uL[j+1])       * 0.5*(vL_ghost4 + q[v[i,j]]) \
                        +  0.5*(q[u[i,j]]   + q[u[i,j+1]])   * 0.5*(q[v[i,j]]   + q[v[i+1,j]]) 
            
            elif i == nx-1: # Right Wall
                
                vR_ghost2 = 2*vR[j] - q[v[i,j]] # 2-pt stencil
                vR_ghost3 = (8*vR[j] - 6*q[v[i,j]] + q[v[i-1,j]]) / 3. # 3-pt stencil
                vR_ghost4 = (16*vR[j] - 15*q[v[i,j]] + 5*q[v[i-1,j]] - q[v[i-2,j]]) / 5. # 4-pt stencil

                _uy_vx_ = -0.5*(q[u[i-1,j]] + q[u[i-1,j+1]]) * 0.5*(q[v[i-1,j]] + q[v[i,j]]) \
                        +  0.5*(uR[j] + uR[j+1])             * 0.5*(q[v[i,j]]   + vR_ghost4) 
                
            else: 
                _uy_vx_ = -0.5*(q[u[i-1,j]] + q[u[i-1,j+1]]) * 0.5*(q[v[i-1,j]] + q[v[i,j]]) \
                        +  0.5*(q[u[i,j]]   + q[u[i,j+1]])   * 0.5*(q[v[i,j]]   + q[v[i+1,j]]) 
            
            if j == 0: # Bottom Wall
                _vy_vy_ = -(0.5*(vB[i]       + q[v[i,j]]))**2  \
                        +  (0.5*(q[v[i,j]]   + q[v[i,j+1]]))**2 
            elif j == ny-2: # Top Wall
                _vy_vy_ = -(0.5*(q[v[i,j-1]] + q[v[i,j]]))**2  \
                        +  (0.5*(q[v[i,j]]   + vT[i]))**2 
            else: # Interior
                _vy_vy_ = -(0.5*(q[v[i,j-1]] + q[v[i,j]]))**2  \
                        +  (0.5*(q[v[i,j]]   + q[v[i,j+1]]))**2 
            
            del_x_uy_vx = _uy_vx_ / dx
            del_y_vy_vy = _vy_vy_ / dy

            advq[v[i,j]] = del_x_uy_vx + del_y_vy_vy

    return advq

def S(q, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True):
    
    Lq = laplace(q, u, v, p, dx, dy, nx, ny, q_size, pinned=False)
    a = alpha*nu*dt
    I = np.ones(Lq.shape)
    Sq = np.add(q, np.multiply(a, Lq))

    return Sq

def R(q, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True):
    
    Lq = laplace(q, u, v, p, dx, dy, nx, ny, q_size, pinned=False)
    a = alpha*nu*dt
    I = np.ones(Lq.shape)
    Rq = np.subtract(q, np.multiply(a, Lq))
    
    return Lq, a, I, Rq

