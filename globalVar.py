"""
Created on May 2 2021
@author: Shehan M. Parmar
Global variables for NS solver. 
"""
with open('inputs.txt', 'r') as inp: 
    
    inputs = {}
    
    for line in inp:
        key = line.split('=')[0].strip()
        attr = line.split('=')[1].strip()
        inputs[key] = attr
    
    nx = int(inputs["nx"])
    ny = int(inputs["ny"]) 
    Lx = float(inputs["Lx"])
    Ly = float(inputs["Ly"])
    
    dx = Lx/nx
    dy = Ly/ny

    q_size = (nx-1)*ny + nx*(ny-1)
    p_size = nx*ny-1
