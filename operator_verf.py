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
            "fxy2"  : lambda x, y : np.sin(x)*np.cos(y) - np.cos(x)*np.sin(y),
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
        
        dxdy.append(dyi)
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

            "fx2"   : lambda x, y : x**2 * y**2, 
            "fy2"   : lambda x, y : x**2 * y**2,
            "Lfx2" : lambda x, y : 2. * (x**2 + y**2),
            "Lfy2" : lambda x, y : 2. * (x**2 + y**2)
            }

    fx = functions["fx1"]
    Lfx = functions["Lfx1"] 
    
    fy = functions["fy1"] 
    Lfy = functions["Lfy1"]  

    dxdy = []
    L2 = []
    Linf = []
    acc = 0
    qBC = {}

    grid = zip(dx, dy, nx, ny, q_size)
    for dxi, dyi, nxi, nyi, q_sizei in grid:
        print('dxi, dyi')
        print(dxi, dyi)
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
        print('q_test_y')
        print(q_test_y)
        print('q_test_y_ex')
        print(q_test_y_ex)
        # Top Wall BC
        qBC["uT"] = fx(xu,Ly)
        qBC["vT"] = fy(xv,Ly)
        # Bottom Wall BC
        qBC["uB"] = fx(xu,0)
        qBC["vB"] = fy(xv,0)
        
        print('vB')
        print(xv)
        print(qBC["vB"])
        
        # Left Wall BC
        qBC["uL"] = fx(0,yu)
        qBC["vL"] = fy(0,yv)
        
        print('vL')
        print(yv)
        print(qBC["vL"])
        
        # Right Wall BC
        qBC["uR"] = fx(Lx,yu)
        qBC["vR"] = fy(Lx,yv)
        
        Lq = op.laplace(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 
        LqBC  =  op.bclap(q_test, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 
        q = Lq + LqBC 
        

        diff = q-q_test_ex
        print('Bottom left corner of exact solution: %.3ef' % (q_test_ex[vi[0,0]]))
        plt.plot(list(range(0,len(diff))), np.abs(diff))
        plt.show()
        dxdy.append(dxi)
        L2.append( LA.norm(diff) / len(q) ) 
        Linf.append(LA.norm(diff, np.inf))
    
    err = L2
    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Laplace', save=save)
        
    return dxdy, err, acc
