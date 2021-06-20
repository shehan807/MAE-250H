"""
Created on May 2 2021
@author: Shehan M. Parmar
Test the Crank-Nicholson Method for the 
2D diffusion equation, ut = a (uxx + uyy)
"""

# Main.py local dependencies 
from get_global import * # nx, ny, Lx, Ly, dx, dy, q_size, p_size are 'GLOBAL'
from init import *
import operators as op
import operator_verf as opf
import matplotlib.pyplot as plt
from matplotlib import cm
from cgs import *
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import cg

outFile = 'output'+filename.split('inputs')[-1]


# Choose function with known analytic solution for the diffusion equation
functions = {
        "u_xyt"   : lambda x, y, t: np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y),
        "v_xyt"   : lambda x, y, t: np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y),
        "Lu_xyt"  : lambda x, y, t: -2*np.pi**2*np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y)
        }

u_xyt = functions["u_xyt"]
Lu_xyt = functions["Lu_xyt"]
v_xyt = functions["v_xyt"]

dxdy = []
L2 = []
Linf = []
acc = 0
qBC = {}

dt = .1
T = 100
Nt = int(round(T/float(dt)))
t = np.linspace(0, Nt*dt, Nt+1)
alpha = .5 # Crank-Nicholson 
nu = 1

Nt = inttf = 10
grid = zip(dx, dy, nx, ny, q_size)
for dxi, dyi, nxi, nyi, q_sizei in grid:
    
    # ---------- Initialize Simulation Domain ------------------
    
    print('dxi = %.3e, dyi = %.3e, dx*dy = %.3e' % (dxi, dyi, dxi*dyi))
    print('dt < dxdx/nu -> %.3e < %.3e / %.5f -> %s' % (dt, dxi*dyi, nu, str(dt<(dxi*dyi/nu)) ))
    [ui, vi, pi] = init(nxi, nyi, pinned=False)
    
    # U Positions
    xu = dxi*(1. + np.arange(0, nxi-1))
    yu = dyi*(0.5 + np.arange(0, nyi)) 
    Xu, Yu = np.meshgrid(xu, yu)
    
    # V Positions 
    xv = dxi*(0.5 + np.arange(0, nxi))
    yv = dyi*(1.0 + np.arange(0, nyi-1))
    Xv, Yv = np.meshgrid(xv, yv)
    
    # IC t = 0 
    t0 = 0
    U = np.reshape(u_xyt(Xu, Yu, t0), (1, nyi*(nxi-1)))
    V = np.reshape(v_xyt(Xv, Yv, t0), (1, nxi*(nyi-1))) 
    
    q_n = np.concatenate( (U, V), axis = 1)
    q_n = q_n[0] 

    # ---------- Plot Initial U ------------------
    plotInit = False
    if plotInit:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        q_u = np.reshape(q_n[0:nyi*(nxi-1)], (Xu.shape)) 
        surf = ax.plot_surface(Xu, Yu, q_u, rstride=1, cstride=1,\
                cmap=cm.viridis, linewidth=0, antialiased=True)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('$xu$')
        ax.set_ylabel('$yu$')
        ax.view_init(30, 45)
        plt.show()
    
    # ---------- Set Boundary Conditions -----------------------
    
    # Top Wall BC
    qBC["uT"] = u_xyt(xu,Ly, 0)
    qBC["vT"] = v_xyt(xv,Ly, 0)
    # Bottom Wall BC
    qBC["uB"] = u_xyt(xu,0, 0)
    qBC["vB"] = v_xyt(xv,0, 0) 
    # Left Wall BC
    qBC["uL"] = u_xyt(0,yu, 0)
    qBC["vL"] = v_xyt(0,yv, 0)
    # Right Wall BC
    qBC["uR"] = u_xyt(Lx,yu, 0)
    qBC["vR"] = v_xyt(Lx,yv, 0)
    
    
    q_np1 = np.zeros(q_n.shape) 
    q_np1 = q_np1[0]

    for tn in range(0, Nt):
        
        # ---------- Set RHS of Ax = b for Diffusion Eq.  --------
        
        bcL_n = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)

        # Top Wall BC
        qBC["uT"] = u_xyt(xu,Ly, (1+tn)*dt)
        qBC["vT"] = v_xyt(xv,Ly, (1+tn)*dt)
        # Bottom Wall BC
        qBC["uB"] = u_xyt(xu,0,  (1+tn)*dt)
        qBC["vB"] = v_xyt(xv,0,  (1+tn)*dt) 
        # Left Wall BC
        qBC["uL"] = u_xyt(0,yu,  (1+tn)*dt)
        qBC["vL"] = v_xyt(0,yv,  (1+tn)*dt)
        # Right Wall BC
        qBC["uR"] = u_xyt(Lx,yu, (1+tn)*dt)
        qBC["vR"] = v_xyt(Lx,yv, (1+tn)*dt)
        if tn == 0:
            bcL_np1 = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        else: 
            bcL_np1 = op.bclap(q_np1, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        
 
        # ---------- Plot Lq ------------------
        
        plotInit = False
        if plotInit:
            Lu = np.reshape(Lu_xyt(Xu, Yu, t0), (1, nyi*(nxi-1)))
            Lu = Lu[0] 
            
            Lq_u = Lq[0:nyi*(nxi-1)] + bcl_n[0:nyi*(nxi-1)]
            print(LA.norm(Lu-Lq_u, np.inf))
            
            Lq_u = np.reshape(Lq_u, (Xu.shape))
            
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(Xu, Yu, Lq_u, rstride=1, cstride=1,\
                    cmap=cm.viridis, linewidth=0, antialiased=True)
            #ax.set_zlim(0, 1.5)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()
        
        #S = op.S(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False) 
        Lq = op.laplace(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        S = np.add(1, np.multiply(alpha*nu*dt, Lq))
        bcL_n = np.multiply(alpha*dt*nu, bcL_n)
        bcL_np1 = np.multiply(alpha*dt*nu, bcL_np1)
        
        b = S + bcL_n + bcL_np1 
        # ---------- Compare exact "b" value, op.R(q_ex)  --------
        q_n_exact = np.concatenate(\
                    (np.reshape(u_xyt(Xu, Yu, (tn)*dt), (1, nyi*(nxi-1))),\
                    np.reshape(v_xyt(Xv, Yv, (tn)*dt), (1, nxi*(nyi-1)))),\
                    axis = 1)
        q_np1_exact = np.concatenate(\
                    (np.reshape(u_xyt(Xu, Yu, (1+tn)*dt), (1, nyi*(nxi-1))),\
                    np.reshape(v_xyt(Xv, Yv, (1+tn)*dt), (1, nxi*(nyi-1)))),\
                    axis = 1)
        q_n_exact = q_n_exact[0]
        q_np1_exact = q_np1_exact[0]
        RHS = op.S(q_n_exact, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)\
            + bcL_n + bcL_np1
        LHS = op.R(q_np1_exact, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
        LHS = LHS[-1]
        print('RHS of Ax = b (op.S(q_n))')
        print(RHS)
        print('LHS of Ax = b (op.R(q_np1))')
        print(LHS)
        diff_sides = LHS-RHS
        print(RHS-LHS)
        print(LA.norm(RHS-LHS, np.inf))


        # ---------- Solve for the LHS of Ax = b for Diffusion Eq.  --------
         
        #[x, Ax] = Atimes(np.ones(q_n.shape), b, 3, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False) 
        
        #Lq = op.laplace(b, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        #R = np.add(1, np.multiply(Lq, -alpha*nu*dt))
        

        #R = op.R(np.ones(q_n.shape), ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
        R = np.subtract(1, np.multiply(Lq, alpha*nu*dt))
        A = np.diag(R) 
        eigVals = LA.eigvals(A)
        posDef = (eigVals > 0).all()
        symmetric = LA.norm(A.T - A, np.inf) < 1e-6
        if not posDef or not symmetric:
            print(R)
            raise("Matrix is not Positive Definite.")

        
        #c = cg(A, b, x0 = np.zeros(b.shape))
        #c = c[0]
        #c = LA.solve(A, b)
        print(c)
        [c, Ax] = Atimes(np.ones(q_n.shape), b, 3, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
        #c = np.multiply(-1,c)
        #b2 = np.dot(A, c) 
        print('LA.solve norm')
        print(LA.norm(b-np.multiply(A,c), np.inf))

        # ---------- Plot U^n+1 ------------------
        
        plotInit = True
        if plotInit:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            q_u = np.reshape(c[0:nyi*(nxi-1)], (Xu.shape)) 
            surf = ax.plot_surface(Xu, Yu, q_u, rstride=1, cstride=1,\
                    cmap=cm.viridis, linewidth=0, antialiased=True)
            #ax.set_zlim(0, 1.5)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()
        
        q_np1 = c

        q_n = q_np1
        #q_np1 = q_n
        
        q_n_exact = np.concatenate(\
                    (np.reshape(u_xyt(Xu, Yu, (1+tn)*dt), (1, nyi*(nxi-1))),\
                    np.reshape(v_xyt(Xv, Yv, (1+tn)*dt), (1, nxi*(nyi-1)))),\
                    axis = 1)
        
        q_n_exact = q_n_exact[0]
        print(q_n_exact)
        print('Time = %.3f' % ((tn+1)*dt))
        print('Error = %.3e' % (LA.norm(q_n_exact-q_n, np.inf)))
        
        # ---------- Plot Uex^n+1 ------------------
        plotExact = True
        if plotExact:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #q_u_exact = np.reshape(q_n_exact[0:(nyi*(nxi-1))], (Xu.shape)) 
            b_ex = op.R(q_n_exact, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
            q_u_exact = np.reshape(b_ex[0:nyi*(nxi-1)], Xu.shape)
            #plt.contourf(Xu, Yu, q_u_exact)
            #ax.contour3D(Xu, Yu, q_u_exact, 50)
            surf = ax.plot_surface(Xu, Yu, q_u_exact, rstride=1, cstride=1,\
                    cmap=cm.viridis, linewidth=0, antialiased=True)
            ax.set_zlim(0, 1.5)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()
        
        plotCurrent = False
        if plotCurrent:
            # Current Simulation
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            q_u = np.reshape(q_n[0:(nyi*(nxi-1))], (Xu.shape)) 
            #plt.contourf(Xu, Yu, q_u_exact)
            #ax.contour3D(Xu, Yu, q_u_exact, 50)
            surf = ax.plot_surface(Xu, Yu, q_u, rstride=1, cstride=1,\
                    cmap=cm.viridis, linewidth=0, antialiased=True)
            ax.set_zlim(0, 1.5)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()
