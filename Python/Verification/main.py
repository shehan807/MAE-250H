"""
Created on May 2 2021
@author: Shehan M. Parmar
Main Navier-Stokes solver
"""

# Main.py local dependencies 
from get_global import * # nx, ny, Lx, Ly, dx, dy, q_size, p_size are 'GLOBAL'
from init import *
import operators as op
import operator_verf as opf
from cgs import *

outFile = 'output'+filename.split('inputs')[-1]
#[u, v, p] = init(nx, ny)

# Test Gradient Operator is Second-Order Accurate
#[dxdy, err, acc] = opf.test_grad(dx, dy, nx, ny, Lx, Ly, q_size, outFile, save=True) 

# Test Divergence Operator is Second-Order Accurate
#[dxdy, err, acc] = opf.test_div(dx, dy, nx, ny, Lx, Ly, p_size, outFile, save=True) 

# Test Laplace Operator is Second-Order Accurate
#[dxdy, err, acc] = opf.test_laplace(dx, dy, nx, ny, Lx, Ly, q_size, outFile, save=True) 

# Test Advective Operator is Second-Order Accurate
[dxdy, err, acc] = opf.test_adv(dx, dy, nx, ny, Lx, Ly, q_size, outFile, save=True) 

# Test CGS Solver 

#A = testMatrix()
#Ax = Atimes(x, b, 0, A)

