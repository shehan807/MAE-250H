from get_global import * 
from init import *
import operators as op
import operator_verf as opf
import matplotlib.pyplot as plt
from matplotlib import cm
from cgs import *
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import cg
import visualization as vis
import csv
import pandas as pd 
from ma import *

plotCurrent = False

dxdy = []
L2 = []
Linf = []
acc = 0
qBC_nm1 = {}
qBC = {}

dt = 5e-3
T = 10
Nt = int(T/dt)
print('Nt = %d' % (Nt))
t = np.linspace(0, Nt*dt, Nt)
alpha = .5 # Crank-Nicholson 
Re = 100
nu = 1./Re
a = 2

grid = zip(dx, dy, nx, ny, q_size, p_size)
for dxi, dyi, nxi, nyi, q_sizei, g_sizei in grid:
    
    # ---------- Initialize Simulation Domain ------------------
    
    [ui, vi, pi] = init(nxi, nyi)
    
    # U Positions
    xu = dxi*(1. + np.arange(0, nxi-1))
    yu = dyi*(0.5 + np.arange(0, nyi)) 
    Xu, Yu = np.meshgrid(xu, yu)
    
    # V Positions 
    xv = dxi*(0.5 + np.arange(0, nxi))
    yv = dyi*(1.0 + np.arange(0, nyi-1))
    Xv, Yv = np.meshgrid(xv, yv)
    
    # IC U, V @(x,y,t=0) 
    q_nm1 = np.zeros(q_sizei) 
    
    # ---------- Set Boundary Conditions -----------------------
    
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
    
     
    # ---------- SOLVE FOR u(x,y,tn) WHERE n = 1 ------------
    # ---------- Set Boundary Conditions for n+1 ------------
    
    q_n = q_nm1 
    
    # Top Wall BC
    qBC["uT"] = np.ones(xu.shape)
    qBC["vT"] = xv*0
    # Bottom Wall BC
    qBC["uB"] = xu*0
    qBC["vB"] = xv*0
    # Left Wall BC
    qBC["uL"] = yu*0
    qBC["vL"] = yv*0
    # Right Wall BC
    qBC["uR"] = yu*0
    qBC["vR"] = yv*0
    
    bcL_n = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei) 

    # ---------- Plot Initial U ------------------
    plotInit = False
    if plotInit:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        q_u = np.reshape(q_n[0:nyi*(nxi-1)], (Xu.shape)) 
        
        surf = ax.plot_surface(Xu, Yu, q_u, rstride=1, cstride=1,\
                cmap=cm.viridis, linewidth=0, antialiased=True)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('$xu$')
        ax.set_ylabel('$yu$')
        ax.view_init(30, 45)
        plt.show()
    
    X = np.reshape(q_nm1[0:nyi*(nxi-1)], (Xu.shape))

    # ---------- Begin Time-Stepping ---
    for tn in range(1, Nt+1):
    
        # ---------- Set Boundary Conditions for n+1 ------------
    
        # Top Wall BC
        qBC["uT"] = np.ones(xu.shape)
        qBC["vT"] = xv*0
        # Bottom Wall BC
        qBC["uB"] = xu*0
        qBC["vB"] = xv*0
        # Left Wall BC
        qBC["uL"] = yu*0
        qBC["vL"] = yv*0
        # Right Wall BC
        qBC["uR"] = yu*0
        qBC["vR"] = yv*0
    
        bcL_np1 = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei)

        # ---------- Momentum Eq.  ----------------------------------------
        bcL = np.multiply(0.5*dt*nu, np.add(bcL_n, bcL_np1))
        Sq_n = op.S(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt)
        
        Aq_nm1 = op.adv(q_nm1, qBC_nm1, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei)
        Aq_n = op.adv(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei)
        adv = np.multiply(-0.5*dt, np.subtract(np.multiply(3, Aq_n), Aq_nm1))
        
        b = Sq_n + bcL + adv
        
        [q_F, Rq_np1] = Atimes(np.zeros(q_n.shape), b, 3, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, g_sizei, alpha, nu, dt, pinned=True)
        
        # ---------- Pressure Poisson Eq. ----------------------------------------
        Du_F = op.div(q_F, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei)\
             + op.bcdiv(qBC, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei) 
        
        ppe_rhs = np.multiply(1./dt, Du_F)
        b2 = -ppe_rhs 

        [P_np1, Ax_PPE] = Atimes(np.zeros(g_sizei), b2, 2, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, g_sizei, alpha, nu, dt)
        
        # ---------- Projection Step ----------------------------------------
        GP_np1 = op.grad(P_np1, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei) 
        RinvGP_np1 = op.Rinv(GP_np1, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=True)
        q_np1 = np.subtract(q_F, np.multiply(dt, RinvGP_np1)) 

        q_nm1 = q_n
        qBC_nm1 = qBC
        q_n = q_np1
        bcL_n = bcL_np1
        
        # ---------- Visualization & Save Data ------------------------------
        #vis.plotVelocity(q_n, qBC, xu, xv, yu, yv, nxi, nyi, dt*tn, Re, drawNow = True, quiverOn = True)

        #if (tn % 5) == 0:
        #    vis.plotVelocity(q_n, qBC, xu, xv, yu, yv, nxi, nyi, dt*tn, Re, drawNow = False, quiverOn = False)
        #    print('Time = %f' % ((tn+1)*dt))
        #    #plotCurrent = True
       
        U_data = np.reshape(q_n[0:nyi*(nxi-1)], (Xu.shape)) 
        X = np.concatenate((X,U_data))
        if (tn % 5) == 0:
            modal_analysis(X, Xu, Yu)

        # ---------- Save X-Data at y = 0.5 ------------------
        plotXTime = False
        if plotXTime:
            q_u = np.reshape(q_n[0:nyi*(nxi-1)], (Xu.shape)) 
            U_data.append(q_u[5])
            time.append(tn*dt)
            #plt.plot(xu, q_u[5])

        if plotCurrent:
            # Current Simulation
            levels = np.linspace(-0.3, 1, 1000)
            fig, ax = plt.subplots()
            q_u = np.reshape(q_n[0:(nyi*(nxi-1))], (Xu.shape)) 
            CS = ax.contourf(Xu, Yu, q_u, levels=levels, cmap=cm.viridis)
            fig.colorbar(CS)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            plotCurrent = False
    
