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
            "divf2" : lambda x, y : x*0. + y*0.
            }

    fx = functions["fx2"]
    fy = functions["fy2"] 
    divf = functions["divf2"] 
    

    dxdy = []
    err = []
    acc = 0
    grid = zip(dx, dy, nx, ny, g_size)
    qBC = {}

    for dxi, dyi, nxi, nyi, g_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=True)

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
        
        g_ex = divf_ex[0][1::]
        
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
        
        gDiv = op.div(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=True) 
        gBC  =  op.bcdiv(qBC, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=True) 
        g = gDiv + gBC 
        
        err.append( LA.norm(g-g_ex) / len(g) ) 
        dxdy.append(dxi*dyi)
        Linf = LA.norm(err, ord=np.inf)

    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Divergence', save=save)
        
    return dxdy, err, acc

def test_laplace(dx, dy, nx, ny, Lx, Ly, q_size, outFile, plots=True, save=False):
    
    # Choose function with known analytic solution for divergence
    functions = {
            "fx1"   : lambda x, y : -y + x*0, 
            "fy1"   : lambda x, y : x*y,
            "divf1" : lambda x, y : x + y*0,
            "fx2"   : lambda x, y : np.sin(x)*np.cos(y), 
            "fy2"   : lambda x, y : -np.cos(x)*np.sin(y),
            "divf2" : lambda x, y : x*0. + y*0.
            }

    fx = functions["fx2"]
    fy = functions["fy2"] 
    divf = functions["divf2"] 
    

    dxdy = []
    err = []
    acc = 0
    grid = zip(dx, dy, nx, ny, g_size)
    qBC = {}

    for dxi, dyi, nxi, nyi, g_sizei in grid:
        
        [ui, vi, pi] = init(nxi, nyi, pinned=True)

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
        
        g_ex = divf_ex[0][1::]
        
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
        
        gDiv = op.div(q_test, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=True) 
        gBC  =  op.bcdiv(qBC, ui, vi, pi, dxi, dyi, nxi, nyi, g_sizei, pinned=True) 
        g = gDiv + gBC 
        
        err.append( LA.norm(g-g_ex) / len(g) ) 
        dxdy.append(dxi*dyi)
        Linf = LA.norm(err, ord=np.inf)

    lin = linregress(np.log10(dxdy), np.log10(err))
    acc = lin.slope
    
    if plots:
        vis.plotL2vsGridSize(lin, dxdy, err, outFile, 'Divergence', save=save)
        
    return dxdy, err, acc
