import numpy as np
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import time 

import config as cfg
import discrete_operators as op
from matrix_solvers import Atimes
import visualization as vis

# ---------- Initialize Simulation Domain ------------------

cfg.init('inputsMAIN.txt')
# U Positions
xu = cfg.dx*(1. + np.arange(0, cfg.nx-1))
yu = cfg.dy*(0.5 + np.arange(0, cfg.ny)) 
Xu, Yu = np.meshgrid(xu, yu)

# V Positions 
xv = cfg.dx*(0.5 + np.arange(0, cfg.nx))
yv = cfg.dy*(1.0 + np.arange(0, cfg.ny-1))
Xv, Yv = np.meshgrid(xv, yv)

# IC U, V @(x,y,t=0) 
q_nm1 = np.zeros(cfg.q_size) 
qBC_nm1 = {}
qBC = {}

# Other Settings
CN = 0.5 # alpha value for crank-nicholson method

# ---------- Set Boundary Conditions for n - 0-------------

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

# ---------- SOLVE FOR u(x,y,tn) WHERE n  ------------

# ---------- Set Boundary Conditions for n ------------
q_n = q_nm1 
qBC = qBC_nm1
bcL_n = op.bclap(q_n, qBC) 

# ---------- Begin Time-Stepping ---
start_time = time.time()
Nt = int(cfg.T/cfg.dt)
for tn in range(1, Nt+1): 
    
    # ---------- Set Boundary Conditions for n+1 ------------
    qBC_np1 = qBC
    bcL_np1 = op.bclap(q_n, qBC_np1)
    
    # ---------- Momentum Eq.  ----------------------------------------
    bcL = np.multiply((CN*cfg.dt)/cfg.Re, np.add(bcL_n, bcL_np1))
    Sq_n = op.S(q_n) 
    Aq_nm1 = op.adv(q_nm1, qBC_nm1)
    Aq_n = op.adv(q_n, qBC)
    adv = np.multiply(-CN*cfg.dt, np.subtract(np.multiply(3, Aq_n), Aq_nm1))
    b = Sq_n + bcL + adv
    
    [q_F, Rq_np1, iterMo] = Atimes(np.zeros(q_n.shape), b, 3)
    
    # ---------- Pressure Poisson Eq. ----------------------------------------
    Du_F = op.div(q_F) + op.bcdiv(qBC) 
    
    ppe_rhs = np.multiply(1./cfg.dt, Du_F)
    b2 = -ppe_rhs 

    [P_np1, Ax_PPE, iterPPE] = Atimes(np.zeros(cfg.p_size), b2, 2)
    
    # ---------- Projection Step ----------------------------------------
    GP_np1 = op.grad(P_np1) 
    RinvGP_np1 = op.Rinv(GP_np1)
    q_np1 = np.subtract(q_F, np.multiply(cfg.dt, RinvGP_np1)) 

    q_nm1 = q_n
    qBC_nm1 = qBC
    q_n = q_np1
    bcL_n = bcL_np1
    
    # ---------- Output to Log File  ----------------------------------------
    if ((tn % int(0.1*Nt)) == 0) or (tn == 1): # Print outputs every 10% of simulation or at t = 0
        with open(cfg.outputPath + 'output.log', 'a+') as log:
            log.write('\r%(comp).1F%% complete:' %{'comp': (tn/Nt)*100})
            log.write('\r%(iter).d Iterations for CGS Convergence (Mo. Eq.)' %{'iter': iterMo})
            log.write('\r%(iter).d Iterations for CGS Convergence (PP Eq.)\n' %{'iter': iterPPE})
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

        with open(cfg.outputPath + 'output.log', 'a+') as log:
            log.write('\nSimulation Completed in %.3f %s\n' % (sim_time, time_units))
        
    #break
