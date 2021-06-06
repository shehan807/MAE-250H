"""
Created on May 2 2021
@author: Shehan M. Parmar
Conjugate Gradient solver for pressure poisson and 
momentum equations. 
"""
import numpy as np
from sklearn.datasets import make_spd_matrix
import numpy.linalg as LA
from numpy.random import rand 
from numpy.random import seed 
import matplotlib.pyplot as plt

import operators as op
#def unpackInputs(**kwargs):
#    for key in kwarg:


def Atimes(x, b, eqn, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False, **kwargs):
    
    # Initial Solver Conditions
    i = 1
    imax = 100
    eps = 1e-6
    
    if eqn == 0:
        if "A" not in [*kwargs]:
            raise("Must specify matrix variable 'A'.")
        A = kwargs["A"]
        Ax = np.dot(A, x)
    elif eqn == 1:
        # Ax = ...
        pass # Momentum Eq.
    elif eqn == 2:
        # Ax = div(Rinv(grad(x))
        pass # Pressure Poisson Eq
    elif eqn == 3: # Diffusion Eq.
        [Lq, a, I, Ax] = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
        print('x')
        print(x) 
        print('Lq')
        print(Lq) 
        print('a')
        print(a)
        print('Ax') 
        print(Ax) 
    r = np.subtract(b, Ax)
    d = r
    del_new = np.dot(r.T, r)
    del0 = del_new
    print('************Initial Variables')
    print('b')
    print(b)
    print('Ax')
    print(Ax)
    print('r')
    print(r)
    
    print('del_new = r.T*r = %f' % (del_new))
    print('eps**2*del0 = %.4e' % (eps**2*del0))
    del_new_vals = []
    del_new_vals.append(del_new)
    while (i < imax) and (del_new > eps**2*del0):
        
        print('************Iteration %d ************\n' %(i))
        
        if eqn == 0:
            q = np.dot(A, d)
        elif eqn == 1:
            # Ad = ...
            q = Ad
        elif eqn == 2:
            # Ad = div(rinv(grad(d)))
            q = Ad
        elif eqn == 3: # Diffusion Eq.
            [Lq, a, I, Ad] = op.R(d, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
            
            print('1) Determine Ad:\n')
            print('R = I - a*dt*nu*L')
            print('Lq')
            print(Lq) 
            print('a')
            print(a)
            print('I -a*Lq')
            print(1 - a*Lq)
            print('alpha = %.3f, nu = %.3f, dt = %.3f, alpha*nu*dt = %.3e' % (alpha, nu, dt, a))
            print('d')
            print(d)
            print('q updated with aboce d')
            q = Ad
            print('q = Ad = op.R(d)')
            print(q)

        print('2) Update r:\n')
        alpha_cg = np.divide( del_new , np.dot(d.T, q) )
        print('alpha_cg = del_new / (d.T*q) = %4e' % (alpha_cg))
        x = np.add(x , np.multiply(alpha_cg,d))
        print('x = x + alpha_cg*d = ')
        print(x)
         
        if (i % 50) == 0:
            if eqn == 0:
                r = np.subtract(b, np.dot(A, x))
            elif eqn == 1:
                # Ax = ...
                r = np.subtract(b, Ax)
            elif eqn == 2:
                # Ax = div(rinv(grad(x)))
                r = np.subtract(b, Ax)
            elif eqn == 3: # Diffusion Eq.
                [Lq, a, I, Ax] = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
                r = np.subtract(b, Ax)
        else:
            r = np.subtract(r , np.multiply(alpha_cg,q))
        print('r = r - alpha_cg*q')
        print(r)
       
        print('3) Values for next iteration and conv check:')
        del_old = del_new
        del_new = np.dot(r.T, r)
        del_new_vals.append(del_new)
        beta = del_new / del_old
        print('del_new = np.dot(r.T,r) @ (i = %d) = %.3e' % (i, del_new))
        print('beta = del_new / del_old @ (i = %d) = %.3e' % (i, beta))
        d = np.add(r , beta*d)
        print('d = r + beta*d')
        print(d)
        i += 1
     
    if eqn == 0:
        Ax = np.dot(A, x)
    elif eqn == 1:
        # Ax = ...
        pass # Momentum Eq.
    elif eqn == 2:
        # Ax = div(Rinv(grad(x))
        pass # Pressure Poisson Eq.
    elif eqn == 3: # Diffusion Eq.
        Ax = op.R(x, u, v, p, dx, dy, nx, ny, q_size, alpha, nu, dt, pinned=False)
    
    if 'convIter' in kwargs:
        return [i, Ax]
    else:   
        plt.scatter(list(range(0,len(del_new_vals))), del_new_vals, marker='o')
        plt.show()
        print('CGS cnverged in %d iterations.' % (i))
        return [x, Ax]

def testMatrix(ndim, seed = None):
    
    A = make_spd_matrix(ndim, random_state=seed) 
    
    eigVals, eigVec = LA.eig(A)
    posDef = (eigVals > 0).all()
    symmetric = LA.norm(A.T - A, np.inf) < 1e-6
    
    if not posDef or not symmetric:
        raise("Matrix is not Positive Definite.")

    return A

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
