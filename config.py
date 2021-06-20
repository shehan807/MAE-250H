"""
Created on May 2 2021
@author: Shehan M. Parmar
Initialize global variables for NS solver. 
"""
def init(filename):
    import numpy as np
    global u, v, p, nx, ny, dx, dy, Lx, Ly, q_size, p_size, dt, T, Re 
    
    inpFilePath = './InputFiles/'
    pinned = True # Hard-coded and should stay true for NS solver
    # Read in Vriables from Input File
    with open(inpFilePath + filename, 'r') as inp:     
        inputs = {}
        for line in inp:
            key = line.split('=')[0].strip()
            attr = line.split('=')[1].strip()
            if ',' in attr: # applies only for nx, ny, or dt with multiple values
                attr = attr.split(',')
                if ("nx" == key) or ("ny" == key):
                    attr = np.array([int(entry) for entry in attr])
                    inputs[key] = attr
                elif "dt" == key:
                    attr = np.array([float(entry) for entry in attr])
                    inputs[key] = attr
                continue
            
            inputs[key] = float(attr)
      
        nx = int(inputs["nx"])
        Lx = float(inputs["Lx"])
        dx = Lx/(nx)
        
        ny = int(inputs["ny"])
        Ly = float(inputs["Ly"])
        dy = Ly/(ny)
        
        q_size = (nx-1)*ny + nx*(ny-1) 
        p_size = nx*ny-1 # subtract one only for pinned pressure values
    
        dt = inputs["dt"]
        T  = int(inputs["T"])
        Re = inputs["Re"]

        u = np.ndarray((nx-1, ny), dtype=object)
        v = np.ndarray((nx, ny-1), dtype=object)
        p = np.ndarray((nx, ny)  , dtype=object)
        
        # Create pointers for velocity, u, v
        ind = int(0)
        for j in range(0,ny):
            for i in range(0,nx-1):
                u[i,j] = int(ind)
                ind += 1
        for j in range(0,ny-1):
            for i in range(0,nx):
                v[i,j] = int(ind)
                ind += 1
        if ind != ((nx-1)*ny + nx*(ny-1)):
            raise IndexError('wrong velocity size')
    
        # create points for pressure, p
        ind = 0
        for j in range(0,ny):
            for i in range(0,nx):
                if (i==0) and (j==0):
                    if pinned: 
                        p[i,j] = None 
                        pass # skip pinned pressure
                    else:
                        p[i,j] = int(ind)
                        ind += 1
                else:
                    p[i,j] = int(ind)
                    ind += 1
        
        if ind != (nx*ny-1):
            if pinned:
                raise IndexError('wrong pressure index (pinned)')
            elif not pinned and (ind != (nx*ny)):
                raise IndexError('wrong pressure index (not pinned)')
