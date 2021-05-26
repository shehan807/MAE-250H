"""
Created on May 12 2021
@author: S. M. Parmar
Verify discrete operators for Navier-Stokes solver
with known exact solutions. 
"""
import numpy as np
from scipy.stats import linregress
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import mpltex

import operators as op
from init import *

def test_grad(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True):
    
    # Choose function with known analytic solution for gradient
    f1 = lambda x, y : np.sin(x*y) 
    dfx1 = lambda x, y : y*np.cos(x*y)
    dfy1 = lambda x, y : x*np.cos(x*y)

    dxdy = []
    err = []
    acc = 0
    grid = zip(dx, dy, nx, ny, q_size)

    for dxi, dyi, nxi, nyi, q_sizei in grid:
        

        [ui, vi, pi] = init(nxi, nyi, pinned=False)

        # Occasionally, np.arange will include the "stop" value due to rounding/floating
        # point error, so a small corrector term (relative to grid spacing) ensures 
        # arrays have correct length
        corrX = 1e-6*dxi
        corrY = 1e-6*dyi
        
        xu = np.arange(dxi, Lx-corrX, dxi)
        yu = np.arange(0.5*dyi, Ly-corrY, dyi)
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = dfx1(Xu, Yu) #Yu*np.cos(Xu*Yu) 
        grad_x_ex = np.reshape(Zxu, (1, nyi*(nxi-1)))
        
        xv = np.arange(0.5*dxi, Lx-corrX, dxi)
        yv = np.arange(dyi, Ly-corrY, dyi)
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = dfy1(Xv, Yv) #Xv*np.cos(Xv*Yv) 
        grad_y_ex = np.reshape(Zyv, (1, nxi*(nyi-1)))
        
        xp = np.arange(0.5*dxi, Lx-corrX, dxi)
        yp = np.arange(0.5*dyi, Ly-corrY, dyi)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = f1(Xp,Yp) #np.sin(Xp*Yp) 
        
        grad_ex = np.concatenate((grad_x_ex, grad_y_ex), axis=1)
        g_test = np.reshape(Zp, (1,nxi*nyi))
        g_test = g_test[0]
        q = op.grad(g_test, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        
        err.append(  LA.norm(q-grad_ex) / len(g_test) ) 
        dxdy.append(dxi*dyi)
        Linf = LA.norm(err, ord=np.inf)
        
    
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if  plots:
        figFilePath = "./Figures/"
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        plt.rc('grid', c='0.5', ls='-', alpha=0.5, lw=0.5)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r'$\Delta$ $x$*$\Delta$ $y$', fontsize=16)
        ax.set_ylabel(r'L2 Norm, $||x||_{2}$', fontsize=16)
        ax.set_title(r"Spatial Convergence of Gradient Operator, $G$", fontsize=20) 
        ax.annotate(r"Log-Log Slope = $%.2f$" % (acc), 
                xy=(0.75, 0.05), 
                xycoords="axes fraction",
                size=16,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round4", fc="aqua", ec="k", alpha=0.7))

        plt.loglog(dxdy, err, 'bo', mfc="none", markersize=8, label='Gradient Operator Tests')
        plt.loglog(dxdy, 10**(lin.slope*np.log10(dxdy)+lin.intercept), '-r', label='Fitted Line',linewidth=2)
        plt.legend(prop={'size':14})
        plt.grid(True, which="both")
        plt.savefig(figFilePath + outFile.split('.')[0])
        plt.show()

    return dxdy, err, acc
    
