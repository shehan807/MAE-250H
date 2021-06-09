"""
Created on May 12 2021
@author: S. M. Parmar
Verify discrete operators for Navier-Stokes solver
with known exact solutions. 
"""
import numpy as np
from scipy.stats import linregress
from numpy import linalg as LA
from matplotlib import cm
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import mpltex

import operators as op
from init import *
import visualization as vis

def test_grad(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True, save=False):
    
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
            "dfy4" : lambda x, y : np.sin(x)*np.cos(y),
            "f5"   : lambda x, y : np.sin(y)+np.cos(x), 
            "dfy5" : lambda x, y : np.cos(y),
            "dfx5" : lambda x, y : -np.sin(x)
            }

    f = functions["f4"]
    dfx = functions["dfx4"] 
    dfy = functions["dfy4"] 

    dxdy = [] 
    L2 = []
    Linf = []
    acc = 0

    grid = zip(dx, dy, nx, ny, q_size)
    for dxi, dyi, nxi, nyi, q_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=False)
        
        xu = dxi*(1. + np.arange(0, nxi-1))
        yu = dyi*(0.5 + np.arange(0, nyi)) 
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = dfx(Xu, Yu) 
        grad_x_ex = np.reshape(Zxu, (1, nyi*(nxi-1))) 

        xv = dxi*(0.5 + np.arange(0, nxi))
        yv = dyi*(1.0 + np.arange(0, nyi-1))
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = dfy(Xv, Yv) 
        grad_y_ex = np.reshape(Zyv, (1, nxi*(nyi-1)))
        
        grad_ex = np.concatenate((grad_x_ex, grad_y_ex), axis=1)
        grad_ex = grad_ex[0]
        
        xp = dxi*(0.5+np.arange(0, nxi))
        yp = dyi*(0.5+np.arange(0, nyi))
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = f(Xp,Yp)  
        g_test = np.reshape(Zp, (1,nxi*nyi))
        g_test = g_test[0]

        # Alternative Approach that also works: 
        grad_ex2 = np.zeros( nyi*(nxi-1) + nxi*(nyi-1) )
        for j in range(0,nyi):
            for i in range(0,nxi-1):
                grad_ex2[ui[i,j]] = dfx( (i+1.)*dxi , (j+0.5)*dyi  )
        for j in range(0,nyi-1):
            for i in range(0,nxi):
                grad_ex2[vi[i,j]] = dfy( (i+0.5)*dxi , (j+1.)*dyi  )

        g = np.zeros(nxi*nyi)
        for j in range(0,nyi):
            for i in range(0,nxi):
                g[pi[i,j]] = f( (i+0.5)*dxi, (j+0.5)*dyi )

        q = op.grad(g_test, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
     
        diff = np.abs(q-grad_ex)
        
        dxdy.append(dxi)
        L2.append(  LA.norm(diff) / len(diff) )  
        Linf.append(LA.norm(diff, ord=np.inf))
    
    err = Linf
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if  plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Gradient', save=save)

    return dxdy, err, acc
   
def test_div(dx, dy, nx, ny, Lx, Ly, g_size, outFile, plots=True, save=False):
    
    # Choose function with known analytic solution for divergence
    functions = {
            "fx1"   : lambda x, y : -y + x*0, 
            "fy1"   : lambda x, y : x*y,
            "divf1" : lambda x, y : x + y*0,
            "fx2"   : lambda x, y : np.sin(x)*np.cos(y), 
            "fy2"   : lambda x, y : -np.cos(x)*np.sin(y),
            "fxy2"  : lambda x, y : np.cos(x)*np.cos(y) - np.cos(x)*np.cos(y),
            "divf2" : lambda x, y : x*0. + y*0.
            }

    fx = functions["fx2"]
    fy = functions["fy2"]
    fxy = functions["fxy2"]
    divf = functions["divf2"] 
    

    dxdy = []
    L2 = []
    Linf = []
    acc = 0
    qBC = {}
    
    grid = zip(dx, dy, nx, ny, g_size)
    for dxi, dyi, nxi, nyi, g_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=False)
        
        xu = dxi*(1. + np.arange(0, nxi-1))
        yu = dyi*(0.5 + np.arange(0, nyi)) 
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = fx(Xu, Yu) 
        q_test_x = np.reshape(Zxu, (1, nyi*(nxi-1)))
        
        xv = dxi*(0.5 + np.arange(0, nxi))
        yv = dyi*(1.0 + np.arange(0, nyi-1))
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = fy(Xv, Yv) 
        q_test_y = np.reshape(Zyv, (1, nxi*(nyi-1)))
        
        q_test = np.concatenate((q_test_x, q_test_y), axis=1)
        q_test = q_test[0]
        
        xp = dxi*(0.5+np.arange(0, nxi))
        yp = dyi*(0.5+np.arange(0, nyi))
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = divf(Xp,Yp) 
        divf_ex = np.reshape( Zp, (1,nxi*nyi)) 
        divf_ex = divf_ex[0]
        
        # Top Wall BC
        qBC["uT"] = fx(xu,Ly)
        qBC["vT"] = fy(xv,Ly)
        # Bottom Wall BC
        qBC["uB"] = fx(xu,0)
        qBC["vB"] = fy(xv,0)
        # Left Wall BC
        qBC["uL"] = fx(0,yu)
        qBC["vL"] = fy(0,yv)
        # Right Wall BC
        qBC["uR"] = fx(Lx,yu)
        qBC["vR"] = fy(Lx,yv)
         
        gDiv = op.div(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=False) 
        gBC  =  op.bcdiv(qBC, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=False) 
        g = gDiv + gBC 
        dxdy.append(dxi)
        L2.append( LA.norm(g-divf_ex) / len(g) ) 
        Linf.append(LA.norm(g-divf_ex, np.inf))
    
    err = Linf
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Divergence', save=save)
        
    return dxdy, err, acc

def test_laplace(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True, save=False):
    
    # Choose function with known analytic solution for divergence
    functions = {
            "fx1"   : lambda x, y : x**2 + np.sin(y), 
            "fy1"   : lambda x, y : x**2 + np.sin(y),
            "Lfx1" : lambda x, y : 2. + x*0. - np.sin(y),
            "Lfy1" : lambda x, y : 2. + x*0. - np.sin(y),
            
            "fx2"   : lambda x, y : x**2 + y**2, 
            "fy2"   : lambda x, y : x**2 + y**2,
            "Lfx2" : lambda x, y : 4. + x*0. + y*0,
            "Lfy2" : lambda x, y : 4. + x*0. + y*0,

            "fx3"   : lambda x, y : x**2 * y**2, 
            "fy3"   : lambda x, y : x**2 * y**2,
            "Lfx3" : lambda x, y : 2. * (x**2 + y**2),
            "Lfy3" : lambda x, y : 2. * (x**2 + y**2),
            
            "fx4"   : lambda x, y : (np.sin(x)/np.sin(3*np.pi)) + (np.sinh(y)/np.sinh(np.pi)), 
            "fy4"   : lambda x, y : (np.sin(x)/np.sin(3*np.pi)) + (np.sinh(y)/np.sinh(np.pi)),
            "Lfx4" : lambda x, y : x*0 + y*0,
            "Lfy4" : lambda x, y : x*0 + y*0,
            
            "fx5"   : lambda x, y : (np.exp(-0.5*np.pi*x) * np.sin(0.5*np.pi*y)), 
            "fy5"   : lambda x, y : (np.exp(-0.5*np.pi*x) * np.sin(0.5*np.pi*y)),
            "Lfx5" : lambda x, y : x*0 + y*0,
            "Lfy5" : lambda x, y : x*0 + y*0,
            
            "fx6"   : lambda x, y : np.sin(np.pi*x)*np.sin(np.pi*y), 
            "fy6"   : lambda x, y : np.sin(np.pi*x)*np.sin(np.pi*y),
            "Lfx6" : lambda x, y : -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y),
            "Lfy6" : lambda x, y : -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
            }

    fx = functions["fx6"]
    Lfx = functions["Lfx6"] 
    
    fy = functions["fy6"] 
    Lfy = functions["Lfy6"]  

    dxdy = []
    L2 = []
    Linf = []
    acc = 0
    qBC = {}

    grid = zip(dx, dy, nx, ny, q_size)
    for dxi, dyi, nxi, nyi, q_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=False)

        xu = dxi*(1. + np.arange(0, nxi-1))
        yu = dyi*(0.5 + np.arange(0, nyi)) 
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = fx(Xu, Yu) 
        Zxu_ex = Lfx(Xu, Yu)
        q_test_x = np.reshape(Zxu, (1, nyi*(nxi-1)))
        q_test_x_ex = np.reshape(Zxu_ex, (1, nyi*(nxi-1)))

        xv = dxi*(0.5 + np.arange(0, nxi))
        yv = dyi*(1.0 + np.arange(0, nyi-1))
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = fy(Xv, Yv) 
        Zyv_ex = Lfy(Xv, Yv)
        q_test_y = np.reshape(Zyv, (1, nxi*(nyi-1)))
        q_test_y_ex = np.reshape(Zyv_ex, (1, nxi*(nyi-1)))
        
        q_test = np.concatenate((q_test_x, q_test_y), axis=1)
        q_test_ex = np.concatenate((q_test_x_ex, q_test_y_ex), axis=1)
        
        q_test = q_test[0]
        q_test_ex = q_test_ex[0]
        
        # Top Wall BC
        qBC["uT"] = fx(xu,Ly)
        qBC["vT"] = fy(xv,Ly)
        # Bottom Wall BC
        qBC["uB"] = fx(xu,0)
        qBC["vB"] = fy(xv,0) 
        # Left Wall BC
        qBC["uL"] = fx(0,yu)
        qBC["vL"] = fy(0,yv)
        # Right Wall BC
        qBC["uR"] = fx(Lx,yu)
        qBC["vR"] = fy(Lx,yv)
        
        Lq = op.laplace(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 
        LqBC  =  op.bclap(q_test, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 
        q = Lq + LqBC 
        
        checkL_T = False
        if checkL_T: 
            A = np.diag(q)
            #LA.norm(A-A.T, np.inf) -> results in segmentation faults for larger cases

        # ------------------Plot U or V--------------------------
        plotting = False
        if plotting:
            qu = q[0:nyi*(nxi-1)]
            QU = np.reshape(qu, (Xu.shape))
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(Xu, Yu, QU, rstride=1, cstride=1,\
                    cmap=cm.plasma, linewidth=0, antialiased=True)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()

            qu_ex = q_test_ex[0:nyi*(nxi-1)]
            QU_ex = np.reshape(qu_ex, (Xu.shape))
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(Xu, Yu, QU_ex, rstride=1, cstride=1,\
                    cmap=cm.magma, linewidth=0, antialiased=True)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()



        diff = q-q_test_ex
        dxdy.append(dxi)
        L2.append( LA.norm(diff) / len(q) ) 
        Linf.append(LA.norm(diff, np.inf))
    err = Linf
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Laplace', save=save)
        
    return dxdy, err, acc

def test_adv(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True, save=False):
    
    # Choose function with known analytic solution for divergence
    functions = {
            "fu1"   : lambda x, y : np.sin(x)*np.sin(y), 
            "fv1"   : lambda x, y : np.cos(x)*np.cos(y),
            "Nx1"   : lambda x, y :   np.cos(x)*np.sin(x)*np.cos(y)**2 + np.cos(x)*np.sin(x)*np.sin(y)**2,
            "Ny1"   : lambda x, y : - np.cos(y)*np.sin(y)*np.cos(x)**2 - np.cos(y)*np.sin(y)*np.sin(x)**2,
            "fu2"   : lambda x, y :   np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y), 
            "fv2"   : lambda x, y :   x*np.exp(-y/2),
            "Nx2"   : lambda x, y :   2*(np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y))*(np.cos(x)*np.sin(y) \
                                    - np.cos(y)*np.sin(x)) - (x*np.exp(-y/2)*(np.cos(x)*np.cos(y) \
                                    + np.sin(x)*np.sin(y)))/2 - x*np.exp(-y/2)*(np.cos(x)*np.sin(y) \
                                    - np.cos(y)*np.sin(x)),
            "Ny2"   : lambda x, y : np.exp(-y/2)*(np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y)) \
                                    - x**2*np.exp(-y) + x*np.exp(-y/2)*(np.cos(x)*np.sin(y) \
                                    - np.cos(y)*np.sin(x))
            }

    fu = functions["fu1"]
    fv = functions["fv1"]  
    Nx = functions["Nx1"]  
    Ny = functions["Ny1"]  

    dxdy = []
    L2 = []
    Linf = []
    acc = 0
    qBC = {}

    grid = zip(dx, dy, nx, ny, q_size)
    for dxi, dyi, nxi, nyi, q_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=False)

        xu = dxi*(1. + np.arange(0, nxi-1))
        yu = dyi*(0.5 + np.arange(0, nyi)) 
        Xu, Yu = np.meshgrid(xu, yu)
        Zxu = fu(Xu, Yu) 
        Nx_ex = Nx(Xu, Yu)
        
        q_test_x = np.reshape(Zxu, (1, nyi*(nxi-1)))
        q_test_x_ex = np.reshape(Nx_ex, (1, nyi*(nxi-1)))

        xv = dxi*(0.5 + np.arange(0, nxi))
        yv = dyi*(1.0 + np.arange(0, nyi-1))
        Xv, Yv = np.meshgrid(xv, yv)
        Zyv = fv(Xv, Yv) 
        Ny_ex = Ny(Xv, Yv)
        
        q_test_y = np.reshape(Zyv, (1, nxi*(nyi-1)))
        q_test_y_ex = np.reshape(Ny_ex, (1, nxi*(nyi-1)))
        
        q_test = np.concatenate((q_test_x, q_test_y), axis=1)
        q_test_ex = np.concatenate((q_test_x_ex, q_test_y_ex), axis=1)
        
        q_test = q_test[0]
        q_test_ex = q_test_ex[0]
        
        # Top Wall BC
        qBC["uT"] = fu(xu,Ly)
        qBC["vT"] = fv(xv,Ly)
        # Bottom Wall BC
        qBC["uB"] = fu(xu,0)
        qBC["vB"] = fv(xv,0)
        # Left Wall BC
        qBC["uL"] = fu(0,yu)
        qBC["vL"] = fv(0,yv)
        # Right Wall BC
        qBC["uR"] = fu(Lx,yu)
        qBC["vR"] = fv(Lx,yv) # added +0.5*dxi
        
        N = op.adv(q_test, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 
        
        diff = N-q_test_ex
        dxdy.append(dyi)
        
        L2.append( LA.norm(diff) / len(N) ) 
        Linf.append(LA.norm(diff, np.inf))

    err = Linf
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Nonlinear Advection', save=save)
        
    return dxdy, err, acc

