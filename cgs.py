"""
Created on May 2 2021
@author: Shehan M. Parmar
Conjugate Gradient solver for pressure poisson and 
momentum equations. 
"""
from numba import jit
import numpy as np
from sklearn.datasets import make_spd_matrix
import numpy.linalg as LA
from numpy.random import rand 
from numpy.random import seed 
import matplotlib.pyplot as plt
import sys
import operators as op

#np.set_printoptions(threshold=sys.maxsize)
def Atimes(x, b, eqn, u, v, p, dx, dy, nx, ny, q_size, g_size, alpha, nu, dt, pinned=False, **kwargs):
    
    i = 1
    imax = 5000
    eps = 1e-6
    
    if eqn == 0:
        if "A" not in [*kwargs]:
            raise("Must specify matrix variable 'A'.")
        A = kwargs["A"]
        Ax = np.dot(A, x)
    elif eqn == 1:
        Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True) # Momentum Eq.
    elif eqn == 2:
        GP_np1 = op.grad(x, u, v, p, dx, dy, nx, ny, q_size) 
        RinvGP_np1 = op.Rinv(GP_np1, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True)
        DRinvGP_np1 = op.div(RinvGP_np1, u, v, p, dx, dy, nx, ny, g_size, pinned=True)
        Ax = np.multiply(-1., DRinvGP_np1)
        # Pressure Poisson Eq
    elif eqn == 3: # Diffusion Eq.
        Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
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
            q = np.dot(A, d)
        elif eqn == 1:
            Ad = op.R(d, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
            q = Ad
        elif eqn == 2:
            GP_np1 = op.grad(d, u, v, p, dx, dy, nx, ny, q_size) 
            RinvGP_np1 = op.Rinv(GP_np1, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True)
            DRinvGP_np1 = op.div(RinvGP_np1, u, v, p, dx, dy, nx, ny, g_size, pinned=True)
            Ad = np.multiply(-1., DRinvGP_np1)
            #checkAx(Ad)
            q = Ad
        elif eqn == 3: # Diffusion Eq.
            Ad = op.R(d, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
            q = Ad

        alpha_cg = np.divide( del_new , np.dot(d.T, q) )
        x = np.add(x , np.multiply(alpha_cg,d))
         
        if (i % 50) == 0:
            if eqn == 0:
                r = np.subtract(b, np.dot(A, x))
            elif eqn == 1:
                Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
                r = np.subtract(b, Ax)
            elif eqn == 2:
                GP_np1 = op.grad(x, u, v, p, dx, dy, nx, ny, q_size) 
                RinvGP_np1 = op.Rinv(GP_np1, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True)
                DRinvGP_np1 = op.div(RinvGP_np1, u, v, p, dx, dy, nx, ny, g_size, pinned=True)
                Ax = np.multiply(-1., DRinvGP_np1)
                #checkAx(Ax)
                r = np.subtract(b, Ax)
            elif eqn == 3: # Diffusion Eq.
                Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
                r = np.subtract(b, Ax)
        else:
            r = np.subtract(r , np.multiply(alpha_cg,q))
        del_old = del_new
        del_new = np.dot(r.T, r)
        del_new_vals.append(del_new)
        beta = del_new / del_old
        
        d = np.add(r , beta*d)
        i += 1
     
    if eqn == 0:
        Ax = np.dot(A, x)
    elif eqn == 1: # Momentum Eq.:
        Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False) 
    elif eqn == 2:
        GP_np1 = op.grad(x, u, v, p, dx, dy, nx, ny, q_size) 
        RinvGP_np1 = op.Rinv(GP_np1, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=True)
        DRinvGP_np1 = op.div(RinvGP_np1, u, v, p, dx, dy, nx, ny, g_size, pinned=True)
        #checkAx(Ax)
        Ax = np.multiply(-1., DRinvGP_np1)
    elif eqn == 3: # Diffusion Eq.
        Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
    
    if 'convIter' in kwargs:
        return [i, Ax]
    else:   
        #plt.scatter(list(range(0,len(del_new_vals))), del_new_vals, marker='o')
        #plt.show()
        #print('CGS cnverged in %d iterations.' % (i))
        return [x, Ax]

def testMatrix(ndim, seed = None):
    
    A = make_spd_matrix(ndim, random_state=seed) 
    
    eigVals = LA.eigvals(A)
    posDef = (eigVals > 0).all()
    symmetric = LA.norm(A.T - A, np.inf) < 1e-6
    
    if not posDef or not symmetric:
        raise("Matrix is not Positive Definite.")

    return A
def checkAx(Ax):
    A = np.diag(Ax)
    eigVals = LA.eigvals(A)
    posDef = (eigVals > 0).all()
    
    if not posDef:
        print(eigVals)
        raise("Matrix is not Positive Definite.")

# Test CGS
#ndim_val = [10, 10**2, 10**3, int(10**(3.5))]
#for ndim in ndim_val:
#    A_test = testMatrix(ndim, seed = None)
#    print('%.1e' % (np.size(A_test)))
#    #b = np.rand(ndim, 1)
#    b = np.ones((ndim, 1))
#
#    soln = np.dot(LA.inv(A_test), b)
#    x_guess = np.zeros( (ndim, 1))
#    [i, Ax] = Atimes( x_guess, b, eqn = 0, A = A_test, convIter = None)
#    
#    print('Error Norm:')
#    print(LA.norm(Ax-b, np.inf))
#    print('Converged in %d iterations.' % (i))
#
