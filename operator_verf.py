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
import matplotlib.ticker as ticker
import mpltex

import operators as op
from init import *

def test_grad(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True):
    
    # Choose function with known analytic solution for gradient
    functions = {
            "f1"   : lambda x, y : np.sin(x*y), 
            "dfx1" : lambda x, y : y*np.cos(x*y),
            "dfy1" : lambda x, y : x*np.cos(x*y),
            "f2"   : lambda x, y : x**2*y**2,
            "dfx2" : lambda x, y : 2*x*y**2,
            "dfy2" : lambda x, y : 2*y*x**2,
            "f3"   : lambda x, y : x*np.cos(y) + y,
            "dfx3" : lambda x, y : np.cos(y),
            "dfy3" : lambda x, y : -x*np.sin(y) + 1,
            "f4"   : lambda x, y : np.sin(x)*np.sin(y), 
            "dfx4" : lambda x, y : np.sin(y)*np.cos(x),
            "dfy4" : lambda x, y : np.sin(x)*np.cos(y)
            }

    f = functions["f4"]
    dfx = functions["dfx4"] 
    dfy = functions["dfy4"] 

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
        Zxu = dfx(Xu, Yu) #Yu*np.cos(Xu*Yu) 
        grad_x_ex = np.reshape(Zxu, (1, nyi*(nxi-1)))
        
        xv = np.arange(0.5*dxi, Lx-corrX, dxi)
        yv = np.arange(dyi, Ly-corrY, dyi)
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = dfy(Xv, Yv) #Xv*np.cos(Xv*Yv) 
        grad_y_ex = np.reshape(Zyv, (1, nxi*(nyi-1)))
        
        xp = np.arange(0.5*dxi, Lx-corrX, dxi)
        yp = np.arange(0.5*dyi, Ly-corrY, dyi)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = f(Xp,Yp) #np.sin(Xp*Yp) 
        
        grad_ex = np.concatenate((grad_x_ex, grad_y_ex), axis=1)
        g_test = np.reshape(Zp, (1,nxi*nyi))
        g_test = g_test[0]
        q = op.grad(g_test, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        
        err.append(  LA.norm(q-grad_ex) / len(q) ) 
        
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
        #plt.savefig(figFilePath + outFile.split('.')[0])
        plt.show()

    return dxdy, err, acc
   
def test_div(dx, dy, nx, ny, Lx, Ly, g_size, outFile, plots=True):
    
    # Choose function with known analytic solution for divergence
    functions = {
            "fx1"   : lambda x, y : -y, 
            "fy1"   : lambda x, y : x*y,
            "divf1" : lambda x, y : x,
            "fx2"   : lambda x, y : np.sin(x)*np.cos(y), 
            "fy2"   : lambda x, y : -np.cos(x)*np.sin(y),
            "divf2" : lambda x, y : 0.
            }

    fx = functions["fx1"]
    fy = functions["fy1"] 
    divf = functions["divf1"] 
    

    dxdy = []
    err = []
    acc = 0
    grid = zip(dx, dy, nx, ny, g_size)
    qBC = {}

    for dxi, dyi, nxi, nyi, g_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=False)

        # Occasionally, np.arange will include the "stop" value due to rounding/floating
        # point error, so a small corrector term (relative to grid spacing) ensures 
        # arrays have correct length
        corrX = 1e-6*dxi
        corrY = 1e-6*dyi
        
        xu = np.arange(dxi, Lx-corrX, dxi)
        yu = np.arange(0.5*dyi, Ly-corrY, dyi)
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = fx(Xu, Yu) 
        q_test_x = np.reshape(Zxu, (1, nyi*(nxi-1)))
        
        xv = np.arange(0.5*dxi, Lx-corrX, dxi)
        yv = np.arange(dyi, Ly-corrY, dyi)
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = fy(Xv, Yv) 
        q_test_y = np.reshape(Zyv, (1, nxi*(nyi-1)))
        
        xp = np.arange(0.5*dxi, Lx-corrX, dxi)
        yp = np.arange(0.5*dyi, Ly-corrY, dyi)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = divf(Xp,Yp) 
        divf_ex = np.reshape( Zp, (1,nxi*nyi)) 
        
        q_test = np.concatenate((q_test_x, q_test_y), axis=1)
        q_test = q_test[0]
        g_ex = divf_ex[0]
        
        # Top Wall BC
        qBC["uT"] = fx(xu,Ly)
        qBC["vT"] = fx(xv,Ly)
        # Bottom Wall BC
        qBC["uB"] = fx(xu,0)
        qBC["vB"] = fx(xv,0)
        # Left Wall BC
        qBC["uL"] = fx(0,yu)
        qBC["vL"] = fx(0,yv)
        # Right Wall BC
        qBC["uR"] = fx(Lx,yu)
        qBC["vR"] = fx(Lx,yv)
         
        g = op.div(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=False) \
          + op.bcdiv(qBC, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=False) 
        



    return dxdy, err, acc
