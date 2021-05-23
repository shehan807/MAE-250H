"""
Created on May 2 2021
@author: Shehan M. Parmar
Main Navier-Stokes solver
"""

import operators as op
from init import *

global nx, ny, u, v, p, dx, dy, nq, np

with open('inputs.txt', 'r') as inp: 
    inputs = {}
    for line in inp:
        print(line)
        inputs[line.split('=')[0]] = line.split('=')[1]
    nx = int(inputs["nx"])
    ny = int(inputs["ny"]) 
    Lx = float(inputs["Lx"])
    Ly = float(inputs["Ly"])
    
    dx = Lx/nx
    dy = Ly/ny

    nq = (nx-1)*ny + nx*(ny-1)
    np = nx*ny-1

init()
print(u)

