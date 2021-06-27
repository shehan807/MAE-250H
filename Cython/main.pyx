import numpy as np
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import time 
from numba import jit 
import matplotlib.ticker as ticker
import os
import numpy.linalg as LA
from numpy.random import rand 
from numpy.random import seed 
import Cython

def main():
    # INITIALIZE SIMULATION DOMAIN
    
    init('inputsMAIN.txt')
    # U Positions
    xu = dx*(1. + np.arange(0, nx-1))
    yu = dy*(0.5 + np.arange(0, ny)) 
    Xu, Yu = np.meshgrid(xu, yu)
    
    # V Positions 
    xv = dx*(0.5 + np.arange(0, nx))
    yv = dy*(1.0 + np.arange(0, ny-1))
    Xv, Yv = np.meshgrid(xv, yv)
    
    # IC U, V @(x,y,t=0) 
    q_nm1 = np.zeros(q_size) 
    qBC_nm1 = {}
    qBC = {}
    
    # TIME INTEGRATION SETTINGS
    CN = 0.5 # alpha value for crank-nicholson method
    
    # BC FOR n = 0
    
    # Top Wall BC
    qBC_nm1["uT"] = np.ones(xu.shape)
    qBC_nm1["vT"] = xv*0
    # Bottom Wall BC
    qBC_nm1["uB"] = xu*0
    qBC_nm1["vB"] = xv*0
    # Left Wall BC
    qBC_nm1["uL"] = yu*0
    qBC_nm1["vL"] = yv*0
    # Right Wall BC
    qBC_nm1["uR"] = yu*0
    qBC_nm1["vR"] = yv*0
    
    # SOLVE FOR u(x,y,tn) 
    
    # BC FOR n 
    q_n = q_nm1 
    qBC = qBC_nm1
    bcL_n = bclap(q_n, qBC) 
    
    # BEGIN TIME STEPPING-
    start_time = time.time()
    prev_time = start_time
    Nt = int(T/dt)
    for tn in range(1, Nt+1): 
        
        # BC FOR n + 1
        qBC_np1 = qBC
        bcL_np1 = bclap(q_n, qBC_np1)
        
        # MOMENTUM EQUATION
        bcL = np.multiply((CN*dt)/Re, np.add(bcL_n, bcL_np1))
        Sq_n = S(q_n) 
        Aq_nm1 = adv(q_nm1, qBC_nm1)
        Aq_n = adv(q_n, qBC)
        ADV = np.multiply(-CN*dt, np.subtract(np.multiply(3, Aq_n), Aq_nm1))
        b = Sq_n + bcL + ADV
        [q_F, Rq_np1, iterMo] = Atimes(np.zeros(q_n.shape), b, 3)
        
        # PRESSURE POISSON EQUATION
        Du_F = div(q_F) + bcdiv(qBC) 
        
        ppe_rhs = np.multiply(1./dt, Du_F)
        b2 = -ppe_rhs 
    
        [P_np1, Ax_PPE, iterPPE] = Atimes(np.zeros(p_size), b2, 2)
        
        # PROJECTION STEP
        GP_np1 = grad(P_np1) 
        RinvGP_np1 = Rinv(GP_np1)
        q_np1 = np.subtract(q_F, np.multiply(dt, RinvGP_np1)) 
    
        q_nm1 = q_n
        qBC_nm1 = qBC
        q_n = q_np1
        bcL_n = bcL_np1
        
        # UPDATE LOG FILE AND SAVE DATA
        [X, Y, U, V] = getFrameData(q_n, qBC, tn*dt)
        
        if tn == 1:
            Udata = np.vstack(([U], [np.zeros(np.shape(U))]))
            Vdata = np.vstack(([V], [np.zeros(np.shape(V))]))
        
        elif tn == Nt: 
            Udata = np.vstack((Udata, [U]))
            Vdata = np.vstack((Vdata, [V]))
            
            np.save(outputPath+'X_Data_dt_{:.3e}'.format(dt).replace('.','p'),X)
            np.save(outputPath+'Y_Data_dt_{:.3e}'.format(dt).replace('.','p'),Y)
            np.save(outputPath+'U_Data_dt_{:.3e}'.format(dt).replace('.','p'), Udata)    
            np.save(outputPath+'V_Data_dt_{:.3e}'.format(dt).replace('.','p'), Vdata)    
        else:
            Udata = np.vstack((Udata, [U]))
            Vdata = np.vstack((Vdata, [V]))
        
        updated_time = time.time() 
        if (tn % 10 == 0) or (tn == 1): # Print outputs every 10% of simulation or at t = 0
            with open(outputPath + 'output.log', 'a+') as log:
                log.write('\r%(comp).1F%% complete:' %{'comp': (tn/Nt)*100})
                log.write('\rSimultation Time: %.3f sec (dt = %.3e)' % (tn*dt, dt))
                log.write('\r%(iter).d Iterations for CGS Convergence (Mo. Eq.)' %{'iter': iterMo})
                log.write('\r%(iter).d Iterations for CGS Convergence (PP Eq.)' %{'iter': iterPPE})
                log.write('\rWall Clock Time: %.3f sec\n' % (updated_time - prev_time))
                prev_time = updated_time
            
            #plot1DProfile(X, Y, U, V, tn*dt)
            #plot2DStreamPlot(X, Y, U, V, tn*dt)
    
        if tn == Nt: 
            sim_time = time.time() - start_time
            if sim_time < 60:
                time_units = 'sec'
            elif sim_time > 60 and sim_time < 3600:
                sim_time = sim_time / 60
                time_units = 'min'
            elif sim_time > 3600:
                sim_time = sim_time / 3600
                time_units = 'hrs'
    
            with open(outputPath + 'output.log', 'a+') as log:
                log.write('\nSimulation Completed in %.3f %s\n' % (sim_time, time_units))

def grad(g, pinned = True): # Gradient Operator
    
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

def div(q, pinned=True): # Divergence Operator
    
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

def bcdiv(qbc, pinned=True):
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

def laplace(q, pinned=True):
    
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

def bclap(q, qbc, pinned=True):
    
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
        #for i in range(1,nx-2):
        #    bcL[u[i,j]] = 0
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
        
        #for i in range(1,nx-1):
        #    bcL[v[i,j]] =  0
        # Ghost Cell
        for i in [nx-1]:
            
            vR_ghost2 = (2*vR[j] - q[v[i,j]]) # 2-pt. stencil
            vR_ghost3 = (8*vR[j] - 6*q[v[i,j]] + q[v[i-1,j]]) / 3. # 3-pt. stencil
            vR_ghost4 = (16*vR[j] - 15*q[v[i,j]] + 5*q[v[i-1,j]] - q[v[i-2,j]]) / 5. # 4-pt. stencil
            
            bcL[v[i,j]] =  vR_ghost4 / dx**2;

    return bcL

def adv(q, qbc, pinned=True):
    
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

def S(q, alpha = 0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    a = (alpha*dt)/Re
    I = np.ones(Lq.shape)
    Sq = np.add(q, np.multiply(a, Lq))

    return Sq

def R(q, alpha=0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    a = (alpha*dt)/Re
    I = np.ones(Lq.shape)
    Rq = np.subtract(q, np.multiply(a, Lq))
    
    return Rq

def Rinv(q, alpha = 0.5, pinned=True):
    
    Lq = laplace(q, pinned=False)
    Lq2 = laplace(Lq, pinned=False)
    a = (alpha*dt)/Re
    a2 = a**2
    I = np.ones(Lq.shape)
    
    # Taylor Series Expansion
    term1 = np.multiply(I, q)
    term2 = np.multiply(a, Lq)
    term3 = np.multiply(a2, Lq2)
    Rinvq = np.add(np.add(term1, term2), term3)

    return Rinvq

def Atimes(x, b, eqn, pinned=False, Atest = []):
    
    i = 1
    imax = 5000
    eps = 1e-6
    
    if eqn == 0: # Test Matrix
        if not Atest:
            raise("Must specify matrix variable 'A'.")
        #A = kwargs["A"]
        Ax = np.dot(Atest, x)
    elif eqn == 1: # Momentum Eq.
        Ax = R(x)
    elif eqn == 2: # Pressure Poisson Eq.
        GP_np1 = grad(x) 
        RinvGP_np1 = Rinv(GP_np1)
        DRinvGP_np1 = div(RinvGP_np1)
        Ax = np.multiply(-1., DRinvGP_np1)
    elif eqn == 3: # Diffusion Eq.
        Ax = R(x, pinned=False)
    
    r = np.subtract(b, Ax)
    d = r
    del_new = np.dot(r.T, r)
    del0 = del_new
    
    del_new_vals = []
    del_new_vals.append(del_new)
    
    while (i < imax) and (del_new > eps**2*del0):
        
        if (i % 500) == 0:
            print('Iteration No: %d' % (i))
            print('del_new = %.3e' % (del_new))

        if eqn == 0:
            q = np.dot(Atest, d)
        elif eqn == 1: # Mo. Eq.
            Ad = R(d)
            q = Ad
        elif eqn == 2: # PP Eq.
            GP_np1 = grad(d) 
            RinvGP_np1 = Rinv(GP_np1)
            DRinvGP_np1 = div(RinvGP_np1)
            Ad = np.multiply(-1., DRinvGP_np1)
            q = Ad
        elif eqn == 3: # Diff. Eq.
            Ad = R(d, pinned=False)
            q = Ad

        alpha_cg = np.divide( del_new , np.dot(d.T, q) )
        x = np.add(x , np.multiply(alpha_cg,d))
         
        if (i % 50) == 0:
            if eqn == 0: # Test Matrix
                r = np.subtract(b, np.dot(Atest, x))
            elif eqn == 1: # Mo. Eq.
                Ax = R(x)
                r = np.subtract(b, Ax)
            elif eqn == 2: # PP Eq.
                GP_np1 = grad(x) 
                RinvGP_np1 = Rinv(GP_np1)
                DRinvGP_np1 = div(RinvGP_np1)
                Ax = np.multiply(-1., DRinvGP_np1)
                r = np.subtract(b, Ax)
            elif eqn == 3: # Diff. Eq.
                Ax = R(x, pinned=False)
                r = np.subtract(b, Ax)
        else:
            r = np.subtract(r , np.multiply(alpha_cg,q))
        del_old = del_new
        del_new = np.dot(r.T, r)
        del_new_vals.append(del_new)
        beta = del_new / del_old
        
        d = np.add(r , beta*d)
        i += 1
     
    if eqn == 0: # Test Matrix
        Ax = np.dot(Atest, x)
    elif eqn == 1: # Mo. Eq.
        Ax = R(x) 
    elif eqn == 2: # PP Eq.
        GP_np1 = grad(x) 
        RinvGP_np1 = Rinv(GP_np1)
        DRinvGP_np1 = div(RinvGP_np1)
        Ax = np.multiply(-1., DRinvGP_np1)
    elif eqn == 3: # Diff. Eq.
        Ax = R(x, pinned=False)
    
    #if 'convIter' in kwargs:
    #    pass
    #   return [i, Ax]
    #else:   
        #plt.scatter(list(range(0,len(del_new_vals))), del_new_vals, marker='o')
        #plt.show()
        #print('CGS cnverged in %d iterations.' % (i))
        #return [x, Ax, i]
    return [x, Ax, i]

def plotL2vsGridSize(linReg, dxdy, error, outFile, oprtr, save=False):
    """
    INPUTS:
    ------
    linReg - linear regression data from linregress function
    dxdy   - array of spatial grid sizes (x-axis)
    error  - array of error values (y-axis)
    oprtr  - string value name of the operator being tested (for title)
    outFile- name of output file for figure
    """
    figFilePath = "./Figures/"
     
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.rc('grid', c='0.5', ls='-', alpha=0.5, lw=0.5)
    
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlabel(r'$\Delta$ $t$', fontsize=16)
    ax.set_xlabel(r'$\Delta$ $x$, $\Delta$ $y$', fontsize=16)
    ax.set_ylabel(r'$L^{\infty}$ Norm, $||x||_{\infty}$', fontsize=16)
    #ax.set_title(r"Temporal Convergence", fontsize=20) 
    ax.set_title(r"Spatial Convergence of " + oprtr + " Operator", fontsize=20) 
    ax.annotate(r"Log-Log Slope = $%.2f$" % (linReg.slope), 
            xy=(0.75, 0.05), 
            xycoords="axes fraction",
            size=16,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round4", fc="aqua", ec="k", alpha=0.7))

    plt.loglog(dxdy, error, 'bo', mfc="none", markersize=8, label=oprtr + ' Operator Tests')
    plt.loglog(dxdy, 10**(linReg.slope*np.log10(dxdy)+linReg.intercept), '-r', label='Fitted Line',linewidth=2)
    plt.legend(prop={'size':14})
    plt.grid(True, which="both")
    if save:
        plt.savefig(figFilePath + outFile.split('.')[0])
    plt.show()

    return

def getFrameData(q, qBC, time_n):
    
    U = np.zeros((nx+1, ny+1))
    V = np.zeros((nx+1, ny+1))
     
    # Staggered Grid Velocities
    u_stg = np.reshape(q[0:ny*(nx-1)], (ny  , nx-1))
    v_stg = np.reshape(q[ny*(nx-1):],  (ny-1, nx))
    
    U_vert = 0.5*(u_stg[0:-1,:] + u_stg[1:,:]) 
    V_vert = 0.5*(v_stg[:,0:-1] + v_stg[:,1:])
    
    U[1:nx, 1:ny] = U_vert
    V[1:nx, 1:ny] = V_vert
    
    U[0, :] = 0         #qBC["uB"]
    U[:, 0] = 0         #qBC["uL"]
    U[:, nx] = 0    #qBC["uR"]
    U[ny, :] = 1    #qBC["uT"]
    
    V[0, :] = 0         #qBC["vB"]
    V[ny, :] = 0    #qBC["vT"]
    V[:, 0] = 0         #qBC["vL"]
    V[:, nx] = 0    #qBC["vR"]
    
    x = dx*np.arange(0,nx+1)
    y = dy*np.arange(0,ny+1)
    X, Y = np.meshgrid(x, y)
    
    return X, Y, U, V

def plot1DProfile(X, Y, U, V, time_n):
    
    newFigPath = outputPath + '/Figures/'
    if not os.path.isdir(newFigPath):
        os.mkdir(newFigPath)
    
    # Read in Ghia Data for Validation
    df = pd.read_csv('Validation/Ghia1982_uData.csv', dtype='float')
    uGhia = df.to_dict(orient='list')
    df = pd.read_csv('Validation/Ghia1982_vData.csv', dtype='float')
    vGhia = df.to_dict(orient='list')
    
    u_ce_Ghia = uGhia[str(Re)]
    y_ce_Ghia = uGhia['y']
    
    v_ce_Ghia = vGhia[str(Re)]
    x_ce_Ghia = vGhia['x']
    
    plotSettings()
    
    # 1D VELOCITY PROFILES U(x = 0.5)
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    
    plt.scatter(Y[:,round(0.5*(nx+1))], U[:,round(0.5*(nx+1))], marker='o', c='b', label= 'Parmar 2021 (Re = ' + str(Re) + ')')
    plt.scatter(y_ce_Ghia, u_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = ' + str(Re) + ')')
    plt.legend(prop={"size":14})
    
    ax1.set_xlabel(r'$y$ position @ $x = 0.5$', fontsize=16)
    ax1.set_ylabel(r'$u$ velocity', fontsize=16)
    ax1.set_title(r"$u$ Velocity Profile along $x = 0.5$ at t = {:.3f}".format(time_n), fontsize=20) 
    
    plt.savefig(newFigPath \
            + "t_{:.3f}_".format(time_n).replace('.','p') \
            + "Re_" + str(Re) \
            + "dx_{:.3f}".format(dx).replace('.','p')\
            + '_uVALIDATION')

    # 1D VELOCITY PROFILES V(y = 0.5)
    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)
    
    plt.scatter(X[round(0.5*(ny+1)), :], V[round(0.5*(ny+1)), :], marker='o', c='b', label='Parmar 2021 (Re = '+str(Re) + ')')
    plt.scatter(x_ce_Ghia, v_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = '+str(Re) + ')')
    plt.legend(prop={"size":14})
    
    ax2.set_title(r"$v$ Velocity Profile along $y = 0.5$ at t = {:.3f}".format(time_n), fontsize=20) 
    ax2.set_xlabel(r'$x$ position @ $y = 0.5$', fontsize=16)
    ax2.set_ylabel(r'$v$ velocity', fontsize=16)
    
    plt.savefig(newFigPath \
            + "t_{:.3f}_".format(time_n).replace('.','p') \
            + "Re_" + str(Re) \
            + "dx_{:.3f}".format(dx).replace('.','p')\
            + '_vVALIDATION')
    plt.clf()
    plt.close('all')

def plot2DStreamPlot(X, Y, U, V, time_n, quiverOn = True, streamOn = True, vortOn = False, save = True):
    
    newFigPath = outputPath + '/Figures/'
    if not os.path.isdir(newFigPath):
        os.mkdir(newFigPath)
    
    plotSettings()

    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(1,1,1)
    
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_xlabel(r'$X$', fontsize=16)
    ax3.set_ylabel(r'$Y$', fontsize=16)
    ax3.set_title(r"Velocity Distribution at t = {:.3f}".format(time_n), fontsize=20) 
    
    levels = np.linspace(0,1,1000)
    cntrf = ax3.contourf(X, Y, np.sqrt(U**2 + V**2), levels=levels, cmap=cm.viridis)
    cbar = plt.colorbar(cntrf, format='%.2f')
    cbar.set_label('Velocity Magnitude', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    
    if quiverOn:
        quiv = plt.quiver(X, Y, U, V, color='white')
    if streamOn:
        strm = plt.streamplot(X, Y, U, V, color='white', linewidth=.5)
    if save:
        plt.savefig(newFigPath \
                + "t_{:.3f}_".format(time_n).replace('.','p') \
                + "Re_" + str(Re) \
                + "dx_{:.3f}".format(dx).replace('.','p'))
    
    if vortOn:
        w = (V[:,1:] - V[:,0:-1])/(dx) - (U[1:, :] - U[0:-1, :])/(dy)
        fig_vorti = plt.figure()
        vort = plt.contour(X, Y, w)
    plt.clf()
    plt.close('all')

def plotSettings():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)

def init(filename):
    import numpy as np
    import os
    import datetime
    global u, v, p, nx, ny, dx, dy, Lx, Ly, q_size, p_size, dt, T, Re, outputPath
    
    inpFilePath = './SIM_DATA/InputFiles/'
    

    pinned = True # Hard-coded and should stay true for NS solver
    # Read in Vriables from Input File
    with open(inpFilePath + filename, 'r') as inp:     
        inputs = {}
        for line in inp:
            key = line.split('=')[0].strip()
            attr = line.split('=')[1].strip()
            if ',' in attr: # applies only for nx, ny, or dt with multiple values
                attr = attr.split(',')
                if ("nx" == key) or ("ny" == key):
                    attr = np.array([int(entry) for entry in attr])
                    inputs[key] = attr
                elif "dt" == key:
                    attr = np.array([float(entry) for entry in attr])
                    inputs[key] = attr
                continue
            
            inputs[key] = float(attr)
      
        nx = int(inputs["nx"])
        Lx = float(inputs["Lx"])
        dx = Lx/(nx)
        
        ny = int(inputs["ny"])
        Ly = float(inputs["Ly"])
        dy = Ly/(ny)
        
        q_size = (nx-1)*ny + nx*(ny-1) 
        p_size = nx*ny-1 # subtract one only for pinned pressure values
    
        dt = inputs["dt"]
        T  = int(inputs["T"])
        Re = int(inputs["Re"])

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
                        p[i,j] = None 
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

        outputPath = './SIM_DATA/Re_' + str(Re) \
                + '_T_{:d}sec'.format(T) \
                + '_nx_' + str(nx) + '_ny_' + str(ny) + '/'

        if not os.path.isdir(outputPath):
            os.mkdir(outputPath)

        with open(outputPath + 'output.log', 'a+') as log:
            log.write(40*'-'+'\nSIMULATION INPUTS\n'\
                    + str(datetime.datetime.now()) + '\n' \
                    + 40*'-'+2*'\n')
            for key in inputs:
                log.write('%s = %s\n' % (key, inputs[key]))
            log.write('CFL Number = %s\n' % (dt/dx))
            log.write(40*'-'+'\nSIMULATION OUTPUTS\n' + 40*'-'+2*'\n')

#if __name__ == "__main__":
#    main()

