"""
Created on May 2 2021
@author: Shehan M. Parmar
Global variables for NS solver. 
"""
import numpy as np
filename = 'inputs.txt'
#filename = 'inputsDiffEqTest.txt'
filename = 'inputsLapTest.txt'
#filename = 'inputsAdvTest.txt'
#filename = 'inputsDivTest.txt'
#filename = 'inputsGradTest.txt'
#filename = 'inputsGradTest_min.txt'
inpFilePath = './InputFiles/'
with open(inpFilePath + filename, 'r') as inp: 
    
    inputs = {}
    
    for line in inp:
        key = line.split('=')[0].strip()
        attr = line.split('=')[1].strip()
        if ',' in attr: # applies only for nx, ny, or dt
            attr = attr.split(',')
            if ("nx" == key) or ("ny" == key):
                attr = np.array([int(entry) for entry in attr])
                inputs[key] = attr
            elif "dt" == key:
                attr = np.array([float(entry) for entry in attr])
                inputs[key] = attr
            continue
        
        inputs[key] = float(attr)
  
    if isinstance(inputs["nx"], str): # NOTE: will not occure, remove in future push 
        nx = int(inputs["nx"])
        ny = int(inputs["ny"])
    elif isinstance(inputs["nx"], float): # occurs everytime unless nx is an array 
        nx = int(inputs["nx"])
        ny = int(inputs["ny"])
    elif isinstance(inputs["nx"], np.ndarray):
        nx = inputs["nx"]
        ny = inputs["ny"]
    
    try:
        if len(nx) != len(ny):
            raise("Inputs in nx and ny MUST match.")
    except: 
        pass # inputs in nx, ny are single integer values

    Lx = float(inputs["Lx"])
    Ly = float(inputs["Ly"])
    
    dx = Lx/(nx)
    dy = Ly/(ny)
    
    q_size = (nx-1)*ny + nx*(ny-1)
    #q_size = (nx-2)*(ny-1) + (nx-1)*(ny-2)
    
    p_size = nx*ny-1 # subtract one only for pinned pressure values
    #p_size = (nx-1)*(ny-1)-1 # subtract one only for pinned pressure values

    #dt = inputs["dt"]
