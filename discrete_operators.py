"""
Created on May 2 2021
@author: Shehan M. Parmar
Discrete operators for Navier-Stokes solver. 
"""
import numpy as np
import config as cfg

def grad(g, pinned = True): # Gradient Operator
    
    q = np.zeros(cfg.q_size)
    
    # Be careful with p(0,0) for the pinned pressure location 
    
    # compute x-dir gradient, u
    for j in [0]:
        for i in [0]:
            if pinned:
                q[cfg.u[i,j]] = (g[cfg.p[i+1,j]]            )/cfg.dx       # - g[cfg.p[0,0]]/cfg.dx = 0
            else: 
                q[cfg.u[i,j]] = (g[cfg.p[i+1,j]] - g[cfg.p[0,0]])/cfg.dx       # 
        for i in range(1,cfg.nx-1):
            q[cfg.u[i,j]] = (g[cfg.p[i+1,j]] - g[cfg.p[i,j]])/cfg.dx       # 
    for j in range(1,cfg.ny):
        for i in range(0,cfg.nx-1):
            q[cfg.u[i,j]] = (g[cfg.p[i+1,j]] - g[cfg.p[i,j]])/cfg.dx       # 

    # compute y-dir gradient, v
    for j in [0]:
        for i in [0]:
            if pinned:
                q[cfg.v[i,j]] = (g[cfg.p[i,j+1]]            )/cfg.dy       # - g[cfg.p[0,0]]/cfg.dy = 0
            else: 
                q[cfg.v[i,j]] = (g[cfg.p[i,j+1]] - g[cfg.p[0,0]])/cfg.dy       #  
                
        for i in range(1,cfg.nx):
            q[cfg.v[i,j]] = (g[cfg.p[i,j+1]] - g[cfg.p[i,j]])/cfg.dy       # 
    
    for j in range(1,cfg.ny-1):
        for i in range(0,cfg.nx):
            q[cfg.v[i,j]] = (g[cfg.p[i,j+1]] - g[cfg.p[i,j]])/cfg.dy       # 
    
    return q

def div(q, pinned=True): # Divergence Operator
    
    if pinned:
        g = np.zeros(cfg.p_size)
    elif not pinned: 
        g = np.zeros(cfg.p_size+1)

    # Bottom Row of Grid
    for j in [0]:
        for i in range(1,cfg.nx-1):
            g[cfg.p[i,j]] = ( q[cfg.u[i,j]] - q[cfg.u[i-1, j]])/cfg.dx \
                      + ( q[cfg.v[i,j]]               )/cfg.dy      
                      #             - q[cfg.v[i,j-1]]  /cfg.dy
    # Bottom Right 
    for j in [0]:
        for i in [cfg.nx-1]:
            g[cfg.p[i,j]] = (           - q[cfg.u[i-1,j]])/cfg.dx  \
                      + ( q[cfg.v[i,j]]              )/cfg.dy       
                      #   q[cfg.u[i,j]]               /cfg.dx   
                      #             - q[cfg.v[i,j-1]] /cfg.dy
    # Left Wall
    for j in range(1, cfg.ny-1):
        for i in [0]:
            g[cfg.p[i,j]] = ( q[cfg.u[i,j]]              )/cfg.dx \
                      + ( q[cfg.v[i,j]] - q[cfg.v[i,j-1]])/cfg.dy
                      #             - q[cfg.u[i-1,j]] /cfg.dx
    # Right Wall 
    for j in range(1,cfg.ny-1):
        for i in [cfg.nx-1]:
            g[cfg.p[i,j]] = (           - q[cfg.u[i-1,j]])/cfg.dx \
                      + ( q[cfg.v[i,j]] - q[cfg.v[i,j-1]])/cfg.dy
                      #   q[cfg.u[i,j]]               /cfg.dx
    # Top Wall 
    for j in [cfg.ny-1]:
        for i in range(1,cfg.nx-1):
            g[cfg.p[i,j]] = ( q[cfg.u[i,j]] - q[cfg.u[i-1,j]])/cfg.dx \
                      + (           - q[cfg.v[i,j-1]])/cfg.dy
                      #   q[cfg.v[i,j]]               /cfg.dy
    # Top Left Corner 
    for j in [cfg.ny-1]:
        for i in [0]:
            g[cfg.p[i,j]] = ( q[cfg.u[i,j]]              )/cfg.dx \
                      + (           - q[cfg.v[i,j-1]])/cfg.dy
                      #             - q[cfg.u[i-1,j]] /cfg.dx
                      #   q[cfg.v[i,j]]               /cfg.dy
    # Top Right Corner 
    for j in [cfg.ny-1]:
        for i in [cfg.nx-1]:
            g[cfg.p[i,j]] = (           - q[cfg.u[i-1,j]])/cfg.dx \
                      + (           - q[cfg.v[i,j-1]])/cfg.dy
                      #   q[cfg.u[i,j]]               /cfg.dx
                      #   q[cfg.v[i,j]]               /cfg.dy
    # Interior Points 
    for j in range(1,cfg.ny-1):
        for i in range(1,cfg.nx-1):
            g[cfg.p[i,j]] = ( q[cfg.u[i,j]] - q[cfg.u[i-1,j]])/cfg.dx \
                      + ( q[cfg.v[i,j]] - q[cfg.v[i,j-1]])/cfg.dy
    return g

def bcdiv(qbc, pinned=True):
    """
    INPUTS: 
    ------
    qbc - dictionary with 8 keys (u and v 
    boundary conditions for each wall)
    """
    if pinned:
        bcD = np.zeros(cfg.p_size)
    elif not pinned:
        bcD = np.zeros(cfg.p_size+1)


    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]


    # Bottom
    for j in [0]:
        for i in range(1, cfg.nx-1):
            bcD[cfg.p[i,j]] = - vB[i]/cfg.dy
    # Bottom Right
    for j in [0]:
        for i in [cfg.nx-1]:
            bcD[cfg.p[i,j]] = uR[j]/cfg.dx - vB[i]/cfg.dy
    # Left Wall 
    for j in range(1,cfg.ny-1):
        for i in [0]:
            bcD[cfg.p[i,j]] = - uL[j]/cfg.dx
    # Right Wall
    for j in range(1, cfg.ny-1):
        for i in [cfg.nx-1]:
            bcD[cfg.p[i,j]] = uR[j]/cfg.dx
            
    # Top Wall 
    for j in [cfg.ny-1]:
        for i in range(1,cfg.nx-1):
            bcD[cfg.p[i,j]] = vT[i]/cfg.dy
    # Top Left Corner
    for j in [cfg.ny-1]:
        for i in [0]:
            bcD[cfg.p[i,j]] = -uL[j]/cfg.dx + vT[i]/cfg.dy
    # Top Right Corner 
    for j in [cfg.ny-1]:
        for i in [cfg.nx-1]:
            bcD[cfg.p[i,j]] = uR[j]/cfg.dx + vT[i]/cfg.dy
    # Interior Points (Zeroed to match q dimensions 
    #for j in range(1,cfg.ny-1):
    #    for i in range(1,cfg.nx-1):
    #        bcD[cfg.p[i,j]] = 0
    
    return bcD

def laplace(q, pinned=True):
    
    Lq = np.zeros(cfg.q_size)

    # NOTE: coeff. = 3 are for ghost cell terms (e.g. (2*uBC - 3*u[i,1] + u[i,2]) / cfg.dy^2
    # U-COMPONENT
    # Bottom Row
    for j in [0]:
        for i in [0]:
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]]               ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]]               ) / cfg.dy**2
                       #                             + q[cfg.u[i-1,j]]   / cfg.dx**2
                       #                             + q[cfg.u[i,j-1]]   / cfg.dy**2
        for i in range(1,cfg.nx-2):
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]]               ) / cfg.dy**2
                       #                             + q[cfg.u[i,j-1]]   / cfg.dy**2
        for i in [cfg.nx-2]:
            Lq[cfg.u[i,j]] = (             - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]]               ) / cfg.dy**2
                       #   q[cfg.u[i+1,j]]                               / cfg.dx**2
                       #                             + q[cfg.u[i,j-1]]   / cfg.dy**2 
    # Top Row
    for j in [cfg.ny-1]:
        for i in [0]:
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]]               ) / cfg.dx**2 \
                       + (             - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
                       #                             + q[cfg.u[i-1,j]]   / cfg.dx**2
                       #   q[cfg.u[i,j+1]]                               / cfg.dy**2
        for i in range(1,cfg.nx-2):
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + (             - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.u[i,j+1]]                               / cfg.dy**2
        for i in [cfg.nx-2]:
            Lq[cfg.u[i,j]] = (             - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + (             - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.u[i+1,j]]                               / cfg.dx**2
                       #   q[cfg.u[i,j+1]]                               / cfg.dy**2 

    # Interior Points
    for j in range(1,cfg.ny-1):
        for i in [0]:
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]]               ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
                       #                             + q[cfg.u[i-1,j]]   / cfg.dx**2
        for i in range(1,cfg.nx-2):
            Lq[cfg.u[i,j]] = ( q[cfg.u[i+1,j]] - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
        for i in [cfg.nx-2]:
            Lq[cfg.u[i,j]] = (             - 2*q[cfg.u[i,j]] + q[cfg.u[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.u[i,j+1]] - 2*q[cfg.u[i,j]] + q[cfg.u[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.u[i+1,j]]                               / cfg.dx**2
    
    # V-COMPONENT

    # Bottom Row
    for j in [0]:
        for i in [0]:
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]]               ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]]               ) / cfg.dy**2
                       #                             + q[cfg.v[i-1,j]]   / cfg.dx**2
                       #                             + q[cfg.v[i,j-1]]   / cfg.dy**2
        for i in range(1,cfg.nx-1):
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]]               ) / cfg.dy**2
                       #                             + q[cfg.v[i,j-1]]   / cfg.dy**2
        for i in [cfg.nx-1]:
            Lq[cfg.v[i,j]] = (             - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]]               ) / cfg.dy**2
                       #   q[cfg.v[i+1,j]]                               / cfg.dx**2
                       #                             + q[cfg.v[i,j-1]]   / cfg.dy**2 
    # Top Row
    for j in [cfg.ny-2]:
        for i in [0]:
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]]               ) / cfg.dx**2 \
                       + (             - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
                       #                             + q[cfg.v[i-1,j]]   / cfg.dx**2
                       #   q[cfg.v[i,j+1]]                               / cfg.dy**2
        for i in range(1,cfg.nx-1):
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + (             - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.v[i,j+1]]                               / cfg.dy**2
        for i in [cfg.nx-1]:
            Lq[cfg.v[i,j]] = (             - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + (             - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.v[i+1,j]]                               / cfg.dx**2
                       #   q[cfg.v[i,j+1]]                               / cfg.dy**2 
    # Interior Points
    for j in range(1,cfg.ny-2):
        for i in [0]:
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]]               ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
                       #                             + q[cfg.v[i-1,j]]   / cfg.dx**2
        for i in range(1,cfg.nx-1):
            Lq[cfg.v[i,j]] = ( q[cfg.v[i+1,j]] - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
        for i in [cfg.nx-1]:
            Lq[cfg.v[i,j]] = (             - 2*q[cfg.v[i,j]] + q[cfg.v[i-1,j]] ) / cfg.dx**2 \
                       + ( q[cfg.v[i,j+1]] - 2*q[cfg.v[i,j]] + q[cfg.v[i,j-1]] ) / cfg.dy**2
                       #   q[cfg.v[i+1,j]]                               / cfg.dx**2


    return Lq

def bclap(q, qbc, pinned=True):
    
    bcL = np.zeros(cfg.q_size)
    

    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]
    
    # U-COMPONENT

    # Bottom Row 
    for j in [0]:
        # BC + Ghost Cell
        for i in [0]:
            
            uB_ghost2 = (2*uB[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j+1]] - q[cfg.u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uL[j] / cfg.dx**2 +  uB_ghost4 / cfg.dy**2
        
        # Ghost Cell
        for i in range(1,cfg.nx-2):
            
            uB_ghost2 = (2*uB[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j+1]] - q[cfg.u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uB_ghost4 / cfg.dy**2
        
        # BC + Ghost Cell
        for i in [cfg.nx-2]:
            
            uB_ghost2 = (2*uB[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uB_ghost3 = (8*uB[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j+1]]) / 3. # 3-pt. stencil
            uB_ghost4 = (16*uB[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j+1]] - q[cfg.u[i,j+2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uR[j] / cfg.dx**2 + uB_ghost4 / cfg.dy**2
    
    # Top Row
    for j in [cfg.ny-1]:
        # BC + Ghost Cell
        for i in [0]:
            
            uT_ghost2 = (2*uT[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j-1]] - q[cfg.u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uL[j] / cfg.dx**2 + uT_ghost4 / cfg.dy**2
        # Ghost Cell
        for i in range(1,cfg.nx-2):
            
            uT_ghost2 = (2*uT[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j-1]] - q[cfg.u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uT_ghost4 / cfg.dy**2
        # BC + Ghost Cell
        for i in [cfg.nx-2]:
            
            uT_ghost2 = (2*uT[i] - q[cfg.u[i,j]]) # 2-pt. stencil
            uT_ghost3 = (8*uT[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j-1]]) / 3. # 3-pt. stencil
            uT_ghost4 = (16*uT[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j-1]] - q[cfg.u[i,j-2]]) / 5. # 4-pt. stencil
            
            bcL[cfg.u[i,j]] = uR[j] / cfg.dx**2 + uT_ghost4 / cfg.dy**2
    
    # Interior Nodes (DONE)
    for j in range(1,cfg.ny-1):
        # BC
        for i in [0]:
            bcL[cfg.u[i,j]] = uL[j] / cfg.dx**2;
        for i in range(1,cfg.nx-2):
            bcL[cfg.u[i,j]] = 0
        # BC
        for i in [cfg.nx-2]:
            bcL[cfg.u[i,j]] = uR[j] / cfg.dx**2; 
    
    # V-COMPONENT

    # Bottom Row 
    for j in [0]:
        # BC + Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i+1,j]] - q[cfg.v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] = vL_ghost4 / cfg.dx**2 + vB[i] / cfg.dy**2;
        # BC
        for i in range(1,cfg.nx-1):
            bcL[cfg.v[i,j]] = vB[i] / cfg.dy**2;
        # BC + Ghost Cell
        for i in [cfg.nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i-1,j]] - q[cfg.v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] = vR_ghost4 / cfg.dx**2 + vB[i] / cfg.dy**2;
    
    # Top Row 
    for j in [cfg.ny-2]:
        # BC + Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i+1,j]] - q[cfg.v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] = vL_ghost4  / cfg.dx**2 + vT[i] / cfg.dy**2;
        # BC
        for i in range(1,cfg.nx-1):
            bcL[cfg.v[i,j]] = vT[i] / cfg.dy**2
        # BC + Ghost Cell
        for i in [cfg.nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i-1,j]] - q[cfg.v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] = vR_ghost4  / cfg.dx**2 + vT[i] / cfg.dy**2;
    
    # Interior Nodes
    for j in range(1,cfg.ny-2):
        # Ghost Cell
        for i in [0]:
            
            vL_ghost2 = (2*vL[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vL_ghost3 = (8*vL[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i+1,j]]) / 3. # 3-pt. stencil
            vL_ghost4 = (16*vL[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i+1,j]] - q[cfg.v[i+2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] =  vL_ghost4 / cfg.dx**2;
        
        for i in range(1,cfg.nx-1):
            bcL[cfg.v[i,j]] =  0
        # Ghost Cell
        for i in [cfg.nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[cfg.v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i-1,j]] - q[cfg.v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[cfg.v[i,j]] =  vR_ghost4 / cfg.dx**2;

    return bcL

def adv(q, qbc, pinned=True):
    
    advq = np.zeros(cfg.q_size)
    
    uB, uL, uR, uT = qbc["uB"], qbc["uL"], qbc["uR"], qbc["uT"]
    vB, vL, vR, vT = qbc["vB"], qbc["vL"], qbc["vR"], qbc["vT"]

    # Nx(i,j) -> u
    # Interpolation Operations, _uy_vx (cell vertices) and _ux_ux (cell centers)
    # Difference Operations, del_x, del_y
    for j in range(0, cfg.ny):
        for i in range(0, cfg.nx-1): # Interior
            
            if i == 0: # Left Wall
                _ux_ux_ = -(0.5*(uL[j]       + q[cfg.u[i,j]]))**2  \
                        +  (0.5*(q[cfg.u[i,j]]   + q[cfg.u[i+1,j]]))**2 
            elif i == cfg.nx-2: # Right Wall    
                _ux_ux_ = -(0.5*(q[cfg.u[i-1,j]] + q[cfg.u[i,j]]))**2  \
                        +  (0.5*(q[cfg.u[i,j]]   + uR[j]))**2 
            else: # Interior
                _ux_ux_ = -(0.5*(q[cfg.u[i-1,j]] + q[cfg.u[i,j]]))**2  \
                        +  (0.5*(q[cfg.u[i,j]]   + q[cfg.u[i+1,j]]))**2 
            
            if j == 0: # Bottom Wall
                
                uB_ghost2 = 2*uB[i] - q[cfg.u[i,j]] # 2-pt stencil
                uB_ghost3 = (8*uB[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j+1]]) / 3. # 3-pt stencil
                uB_ghost4 = (16*uB[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j+1]] - q[cfg.u[i,j+2]]) / 5. # 4-pt stencil
                
                _vx_uy_ = -0.5*(vB[i] + vB[i+1])             * 0.5*(uB_ghost4   + q[cfg.u[i,j]]) \
                        +  0.5*(q[cfg.v[i,j]] + q[cfg.v[i+1,j]])     * 0.5*(q[cfg.u[i,j]]   + q[cfg.u[i,j+1]]) 
            
            elif j == cfg.ny-1: # Top Wall
                
                uT_ghost2 = 2*uT[i] - q[cfg.u[i,j]] # 2-pt stencil
                uT_ghost3 = (8*uT[i] - 6*q[cfg.u[i,j]] + q[cfg.u[i,j-1]]) / 3. # 3-pt stencil
                uT_ghost4 = (16*uT[i] - 15*q[cfg.u[i,j]] + 5*q[cfg.u[i,j-1]] - q[cfg.u[i,j-2]]) / 5. # 4-pt stencil
                
                _vx_uy_ = -0.5*(q[cfg.v[i,j-1]] + q[cfg.v[i+1,j-1]]) * 0.5*(q[cfg.u[i,j-1]] + q[cfg.u[i,j]]) \
                        +  0.5*(vT[i] + vT[i+1])             * 0.5*(q[cfg.u[i,j]]   + uT_ghost4)
                
            else: # Interior
                _vx_uy_ = -0.5*(q[cfg.v[i,j-1]] + q[cfg.v[i+1,j-1]]) * 0.5*(q[cfg.u[i,j-1]] + q[cfg.u[i,j]]) \
                        +  0.5*(q[cfg.v[i,j]]   + q[cfg.v[i+1,j]])   * 0.5*(q[cfg.u[i,j]]   + q[cfg.u[i,j+1]]) 
            
            del_y_vx_uy = _vx_uy_ / cfg.dy
            del_x_ux_ux = _ux_ux_ / cfg.dx
            
            advq[cfg.u[i,j]] = del_x_ux_ux + del_y_vx_uy
        

    # Ny(i,j) -> v
    # Interpolation Operations, _uy_vx (cell vertices) and _vy_vy (cell centers)
    for j in range(0, cfg.ny-1):
        for i in range(0, cfg.nx):
            
            if i == 0: # Left Wall
                
                vL_ghost2 = 2*vL[j] - q[cfg.v[i,j]] # 2-pt stencil
                vL_ghost3 = (8*vL[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i+1,j]]) / 3. # 3-pt stencil
                vL_ghost4 = (16*vL[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i+1,j]] - q[cfg.v[i+2,j]]) / 5. # 4-pt stencil
                
                _uy_vx_ = -0.5*(uL[j]       + uL[j+1])       * 0.5*(vL_ghost4 + q[cfg.v[i,j]]) \
                        +  0.5*(q[cfg.u[i,j]]   + q[cfg.u[i,j+1]])   * 0.5*(q[cfg.v[i,j]]   + q[cfg.v[i+1,j]]) 
            
            elif i == cfg.nx-1: # Right Wall
                
                vR_ghost2 = 2*vR[j] - q[cfg.v[i,j]] # 2-pt stencil
                vR_ghost3 = (8*vR[j] - 6*q[cfg.v[i,j]] + q[cfg.v[i-1,j]]) / 3. # 3-pt stencil
                vR_ghost4 = (16*vR[j] - 15*q[cfg.v[i,j]] + 5*q[cfg.v[i-1,j]] - q[cfg.v[i-2,j]]) / 5. # 4-pt stencil

                _uy_vx_ = -0.5*(q[cfg.u[i-1,j]] + q[cfg.u[i-1,j+1]]) * 0.5*(q[cfg.v[i-1,j]] + q[cfg.v[i,j]]) \
                        +  0.5*(uR[j] + uR[j+1])             * 0.5*(q[cfg.v[i,j]]   + vR_ghost4) 
                
            else: 
                _uy_vx_ = -0.5*(q[cfg.u[i-1,j]] + q[cfg.u[i-1,j+1]]) * 0.5*(q[cfg.v[i-1,j]] + q[cfg.v[i,j]]) \
                        +  0.5*(q[cfg.u[i,j]]   + q[cfg.u[i,j+1]])   * 0.5*(q[cfg.v[i,j]]   + q[cfg.v[i+1,j]]) 
            
            if j == 0: # Bottom Wall
                _vy_vy_ = -(0.5*(vB[i]       + q[cfg.v[i,j]]))**2  \
                        +  (0.5*(q[cfg.v[i,j]]   + q[cfg.v[i,j+1]]))**2 
            elif j == cfg.ny-2: # Top Wall
                _vy_vy_ = -(0.5*(q[cfg.v[i,j-1]] + q[cfg.v[i,j]]))**2  \
                        +  (0.5*(q[cfg.v[i,j]]   + vT[i]))**2 
            else: # Interior
                _vy_vy_ = -(0.5*(q[cfg.v[i,j-1]] + q[cfg.v[i,j]]))**2  \
                        +  (0.5*(q[cfg.v[i,j]]   + q[cfg.v[i,j+1]]))**2 
            
            del_x_uy_vx = _uy_vx_ / cfg.dx
            del_y_vy_vy = _vy_vy_ / cfg.dy

            advq[cfg.v[i,j]] = del_x_uy_vx + del_y_vy_vy

    return advq

def S(q, alpha = 0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    a = (alpha*cfg.dt)/cfg.Re
    I = np.ones(Lq.shape)
    Sq = np.add(q, np.multiply(a, Lq))

    return Sq

def R(q, alpha=0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    a = (alpha*cfg.dt)/cfg.Re
    I = np.ones(Lq.shape)
    Rq = np.subtract(q, np.multiply(a, Lq))
    
    return Rq

def Rinv(q, alpha = 0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    Lq2 = laplace(Lq, pinned=False)
    a = (alpha*cfg.dt)/cfg.Re
    a2 = a**2
    I = np.ones(Lq.shape)
    
    # Taylor Series Expansion
    term1 = np.multiply(I, q)
    term2 = np.multiply(a, Lq)
    term3 = np.multiply(a2, Lq2)
    Rinvq = np.add(np.add(term1, term2), term3)

    return Rinvq

