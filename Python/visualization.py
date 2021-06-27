"""
Created on May 27 2021
@author S. M. Parmar
Various visualization routines for 
verification and flow visualization.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os

import config as cfg 

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
    
    U = np.zeros((cfg.nx+1, cfg.ny+1))
    V = np.zeros((cfg.nx+1, cfg.ny+1))
     
    # Staggered Grid Velocities
    u_stg = np.reshape(q[0:cfg.ny*(cfg.nx-1)], (cfg.ny  , cfg.nx-1))
    v_stg = np.reshape(q[cfg.ny*(cfg.nx-1):],  (cfg.ny-1, cfg.nx))
    
    U_vert = 0.5*(u_stg[0:-1,:] + u_stg[1:,:]) 
    V_vert = 0.5*(v_stg[:,0:-1] + v_stg[:,1:])
    
    U[1:cfg.nx, 1:cfg.ny] = U_vert
    V[1:cfg.nx, 1:cfg.ny] = V_vert
    
    U[0, :] = 0         #qBC["uB"]
    U[:, 0] = 0         #qBC["uL"]
    U[:, cfg.nx] = 0    #qBC["uR"]
    U[cfg.ny, :] = 1    #qBC["uT"]
    
    V[0, :] = 0         #qBC["vB"]
    V[cfg.ny, :] = 0    #qBC["vT"]
    V[:, 0] = 0         #qBC["vL"]
    V[:, cfg.nx] = 0    #qBC["vR"]
    
    x = cfg.dx*np.arange(0,cfg.nx+1)
    y = cfg.dy*np.arange(0,cfg.ny+1)
    X, Y = np.meshgrid(x, y)
    
    return X, Y, U, V

def plot1DProfile(X, Y, U, V, time_n):
    
    newFigPath = cfg.outputPath + '/Figures/'
    if not os.path.isdir(newFigPath):
        os.mkdir(newFigPath)
    
    # Read in Ghia Data for Validation
    df = pd.read_csv('Validation/Ghia1982_uData.csv', dtype='float')
    uGhia = df.to_dict(orient='list')
    df = pd.read_csv('Validation/Ghia1982_vData.csv', dtype='float')
    vGhia = df.to_dict(orient='list')
    
    u_ce_Ghia = uGhia[str(cfg.Re)]
    y_ce_Ghia = uGhia['y']
    
    v_ce_Ghia = vGhia[str(cfg.Re)]
    x_ce_Ghia = vGhia['x']
    
    plotSettings()
    
    # 1D VELOCITY PROFILES U(x = 0.5)
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    
    plt.scatter(Y[:,round(0.5*(cfg.nx+1))], U[:,round(0.5*(cfg.nx+1))], marker='o', c='b', label= 'Parmar 2021 (Re = ' + str(cfg.Re) + ')')
    plt.scatter(y_ce_Ghia, u_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = ' + str(cfg.Re) + ')')
    plt.legend(prop={"size":14})
    
    ax1.set_xlabel(r'$y$ position @ $x = 0.5$', fontsize=16)
    ax1.set_ylabel(r'$u$ velocity', fontsize=16)
    ax1.set_title(r"$u$ Velocity Profile along $x = 0.5$ at t = {:.3f}".format(time_n), fontsize=20) 
    
    plt.savefig(newFigPath \
            + "t_{:.3f}_".format(time_n).replace('.','p') \
            + "Re_" + str(cfg.Re) \
            + "dx_{:.3f}".format(cfg.dx).replace('.','p')\
            + '_uVALIDATION')

    # 1D VELOCITY PROFILES V(y = 0.5)
    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)
    
    plt.scatter(X[round(0.5*(cfg.ny+1)), :], V[round(0.5*(cfg.ny+1)), :], marker='o', c='b', label='Parmar 2021 (Re = '+str(cfg.Re) + ')')
    plt.scatter(x_ce_Ghia, v_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = '+str(cfg.Re) + ')')
    plt.legend(prop={"size":14})
    
    ax2.set_title(r"$v$ Velocity Profile along $y = 0.5$ at t = {:.3f}".format(time_n), fontsize=20) 
    ax2.set_xlabel(r'$x$ position @ $y = 0.5$', fontsize=16)
    ax2.set_ylabel(r'$v$ velocity', fontsize=16)
    
    plt.savefig(newFigPath \
            + "t_{:.3f}_".format(time_n).replace('.','p') \
            + "Re_" + str(cfg.Re) \
            + "dx_{:.3f}".format(cfg.dx).replace('.','p')\
            + '_vVALIDATION')
    plt.clf()
    plt.close('all')

def plot2DStreamPlot(X, Y, U, V, time_n, quiverOn = True, streamOn = True, vortOn = False, save = True):
    
    newFigPath = cfg.outputPath + '/Figures/'
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
                + "Re_" + str(cfg.Re) \
                + "dx_{:.3f}".format(cfg.dx).replace('.','p'))
    
    if vortOn:
        w = (V[:,1:] - V[:,0:-1])/(cfg.dx) - (U[1:, :] - U[0:-1, :])/(cfg.dy)
        fig_vorti = plt.figure()
        vort = plt.contour(X, Y, w)
    plt.clf()
    plt.close('all')

def plotSettings():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)


