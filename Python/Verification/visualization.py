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

def plotVelocity(q, qBC, xu, xv, yu, yv, nx, ny, time, Re, drawNow, strmOn = True, quiverOn = False, save=True):
   


    figFilePath = "./Figures/"
    subDir = "Re" + str(Re) + "/"
    

    u = q[0:ny*(nx-1)]
    v = q[ny*(nx-1):]
    if (len(u) != ny*(nx-1)) or (len(v) != nx*(ny-1)):
        raise("Velocity Components have inccorect length")
    
    U = np.reshape(u, (ny, nx-1))
    V = np.reshape(v, (ny-1, nx))
    
    U_vert = 0.5*(U[0:-1,:] + U[1:,:])
    #U_vert = U_vert.T
    V_vert = 0.5*(V[:,0:-1] + V[:,1:])
    #V_vert = V_vert.T
    X, Y = np.meshgrid(xu, yv)

    # Geometric Center (x = 0.5, y = 0.5)
    u_ce = U[:, int((nx-1)/2)]
    v_ce = V[int((ny-1)/2), :]
    
    # Read in Ghia Data for Validation
    df = pd.read_csv('Ghia1982_uData.csv', dtype='float')
    uGhia = df.to_dict(orient='list')
    df = pd.read_csv('Ghia1982_vData.csv', dtype='float')
    vGhia = df.to_dict(orient='list')
    
    u_ce_Ghia = uGhia[str(Re)]
    y_ce_Ghia = uGhia['y']
    
    v_ce_Ghia = vGhia[str(Re)]
    x_ce_Ghia = vGhia['x']

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    
    # ---------- Velocity Profiles 1D ------------------------------
    
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    plt.scatter(yu, u_ce, marker='o', c='b', label='Parmar 2021 (Re = '+str(Re) + ')')
    plt.scatter(y_ce_Ghia, u_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = '+str(Re) + ')')
    ax1.set_xlabel(r'$y$ position @ $x = 0.5$', fontsize=16)
    ax1.set_ylabel(r'$u$ velocity', fontsize=16)
    plt.legend(prop={"size":14})
    ax1.set_title(r"$u$ Velocity Profile along $x = 0.5$ at t = {:.3f}".format(time), fontsize=20) 
    plt.savefig(figFilePath + subDir \
            + "t_{:.3f}_".format(time).replace('.','p') \
            + "Re_" + str(Re) \
            + "dx_{:.3f}".format(xu[1]-xu[0]).replace('.','p')\
            + '_uVALIDATION')

    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)
    plt.scatter(xv, v_ce, marker='o', c='b', label='Parmar 2021 (Re = '+str(Re) + ')')
    plt.scatter(x_ce_Ghia, v_ce_Ghia, marker='s', c='r', label='Ghia 1982 (Re = '+str(Re) + ')')
    ax2.set_title(r"$v$ Velocity Profile along $y = 0.5$ at t = {:.3f}".format(time), fontsize=20) 
    ax2.set_xlabel(r'$x$ position @ $y = 0.5$', fontsize=16)
    ax2.set_ylabel(r'$v$ velocity', fontsize=16)
    plt.legend(prop={"size":14})
    plt.savefig(figFilePath + subDir \
            + "t_{:.3f}_".format(time).replace('.','p') \
            + "Re_" + str(Re) \
            + "dx_{:.3f}".format(xu[1]-xu[0]).replace('.','p')\
            + '_vVALIDATION')

    # ---------- Velocity Profiles 2D ------------------------------
    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_xlabel(r'$X$', fontsize=16)
    ax3.set_ylabel(r'$Y$', fontsize=16)
    ax3.set_title(r"Velocity Profile at t = {:.3f}".format(time), fontsize=20) 
    
    levels = np.linspace(0,1,1000)
    cntrf = ax3.contourf(X, Y, np.sqrt(U_vert**2 + V_vert**2), levels=levels, cmap=cm.viridis)
    cbar = plt.colorbar(cntrf, format='%.2f')
    cbar.set_label('Velocity Magnitude', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    
    if quiverOn:
        quiv = plt.quiver(X, Y, U_vert, V_vert, color='white')
    
    # ---------- Streamplots 2D ------------------------------
    if strmOn:
        strm = plt.streamplot(X, Y, U_vert, V_vert, color='white', linewidth=.5)
    if save:
        plt.savefig(figFilePath + subDir \
                + "t_{:.3f}_".format(time).replace('.','p') \
                + "Re_" + str(Re) \
                + "dx_{:.3f}".format(xu[1]-xu[0]).replace('.','p'))
    if drawNow:
        plt.show()
    vorticity = True
    if vorticity:
        w = (V[:,1:] - V[:,0:-1])/(xu[1]-xu[0]) - (U[1:, :] - U[0:-1, :])/(yv[1]-yv[0])
        fig_vorti = plt.figure()
        vort = plt.contour(X, Y, w)
        plt.show()
