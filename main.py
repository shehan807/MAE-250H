"""
Created on May 2 2021
@author: Shehan M. Parmar
Main Navier-Stokes solver
"""
global nx, ny, u, v, p, dx, dy, q_size, p_size  

from init import *
import operators as op
print(q_size)


print(op.grad(u))

