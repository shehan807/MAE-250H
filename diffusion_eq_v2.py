from get_global import * 
from init import *
import operators as op
import operator_verf as opf
import matplotlib.pyplot as plt
from matplotlib import cm
from cgs import *
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import cg

# Choose function with known analytic solution for the diffusion equation
functions = {
        "u_xyt"   : lambda x, y, t: np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y),
        "v_xyt"   : lambda x, y, t: np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y),
        "Lu_xyt"  : lambda x, y, t: -2*np.pi**2*np.exp(-2*np.pi**2*t)*np.sin(np.pi*x)*np.sin(np.pi*y)

        }


u_xyt = functions["u_xyt"]
v_xyt = functions["v_xyt"]
Lu_xyt = functions["Lu_xyt"]

dxdy = []
L2 = []
Linf = []
acc = 0
qBC = {}

dt = .01
T = 1
Nt = 10 #int(T/dt)
t = np.linspace(0, Nt*dt, Nt+1)
alpha = .5 # Crank-Nicholson 
nu = 1

grid = zip(dx, dy, nx, ny, q_size)
for dxi, dyi, nxi, nyi, q_sizei in grid:
    
    # ---------- Initialize Simulation Domain ------------------
    
    [ui, vi, pi] = init(nxi, nyi, pinned=False)
    
    # U Positions
    xu = dxi*(1. + np.arange(0, nxi-1))
    yu = dyi*(0.5 + np.arange(0, nyi)) 
    Xu, Yu = np.meshgrid(xu, yu)
    
    # V Positions 
    xv = dxi*(0.5 + np.arange(0, nxi))
    yv = dyi*(1.0 + np.arange(0, nyi-1))
    Xv, Yv = np.meshgrid(xv, yv)
    
    # IC U, V @(x,y,t=0) 
    t0 = 0
    U = np.reshape(u_xyt(Xu, Yu, t0), (1, nyi*(nxi-1)))
    V = np.reshape(v_xyt(Xv, Yv, t0), (1, nxi*(nyi-1))) 
    
    q_n = np.concatenate( (U, V), axis = 1)
    q_n = q_n[0] 
    
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
    
    bcL_n = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False) 

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
    
    # ---------- Plot Laplacian (it works) ------------------
    plotLap = False
    if plotLap:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        Lq_n = op.laplace(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        Lq_n = Lq_n[0:nyi*(nxi-1)]
        LqBC = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        LqBC = LqBC[0:nyi*(nxi-1)]
        Lq = Lq_n + LqBC
        Lq_n_ex = np.reshape(Lu_xyt(Xu, Yu, t0), (1, nyi*(nxi-1)))
        Lq_n_ex = Lq_n_ex[0]
        
        error = LA.norm(Lq - Lq_n_ex, np.inf)
        for j in range(0,nyi):
            print('***** Row %d *****' % (j) )
            Lq_n_row = Lq[(nxi-1)*j:(nxi-1)*(j+1)]
            Lq_n_ex_row = Lq_n_ex[(nxi-1)*j:(nxi-1)*(j+1)]
            error = LA.norm(Lq_n_row - Lq_n_ex_row, np.inf)
            print('Error norm for j = %d is %.3e' % (j, error))     

        Lq = np.reshape(Lq[0:nyi*(nxi-1)], (Xu.shape)) 
        Lq_n_ex = np.reshape(Lq_n_ex[0:nyi*(nxi-1)], (Xu.shape)) 

        surf = ax.plot_surface(Xu, Yu, Lq, rstride=1, cstride=1,\
                cmap=cm.viridis, linewidth=0, antialiased=True)
        ax.set_title('Error Norm of Laplacian: ' + str(error))
        ax.set_xlabel('$xu$')
        ax.set_ylabel('$yu$')
        ax.view_init(30, 45)
        plt.show()
    

    # ---------- Begin Time-Stepping ---
    tn = 0
    print(Nt)
    for tn in range(0, Nt):
    
        # ---------- Set Boundary Conditions for n+1 ------------
    
        # Top Wall BC
        qBC["uT"] = u_xyt(xu,Ly, dt*(1+tn))
        qBC["vT"] = v_xyt(xv,Ly, dt*(1+tn))
        # Bottom Wall BC
        qBC["uB"] = u_xyt(xu,0, dt*(1+tn))
        qBC["vB"] = v_xyt(xv,0, dt*(1+tn)) 
        # Left Wall BC
        qBC["uL"] = u_xyt(0,yu, dt*(1+tn))
        qBC["vL"] = v_xyt(0,yv, dt*(1+tn))
        # Right Wall BC
        qBC["uR"] = u_xyt(Lx,yu, dt*(1+tn))
        qBC["vR"] = v_xyt(Lx,yv, dt*(1+tn))

        bcL_np1 = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)

        # ---------- Set RHS of Ax = b for Diffusion Eq.  --------
        bc = np.multiply(0.5*dt*nu, np.add(bcL_n, bcL_np1))
        Sq_n = op.S(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
        b = Sq_n + bc
        
        # Solve without CGS
        cgs = True
        if not cgs:
            R = op.R(np.ones(q_n.shape), ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False) 
            A = np.diag(R[-1])
            q_np1 = LA.solve(A, b)
        else: 
            [q_np1, Rq_np1] = Atimes(np.zeros(q_n.shape), b, 3, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)


        qu_np1 = q_np1[0:nyi*(nxi-1)]
        q_np1_ex = np.concatenate(\
                    (np.reshape(u_xyt(Xu, Yu, (1+tn)*dt), (1, nyi*(nxi-1))),\
                    np.reshape(v_xyt(Xv, Yv, (1+tn)*dt), (1, nxi*(nyi-1)))),\
                    axis = 1)
        qu_np1_ex = q_np1_ex[0][0:nyi*(nxi-1)]

        error = LA.norm(qu_np1 - qu_np1_ex, np.inf)
        print('Time = %f' % ((tn+1)*dt))
        print('Error b/w qu_np1 and qu_np1_ex: ' + str(error))

        # ---------- Plot U^n+1 ------------------
        
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
        
        q_n = q_np1
        bcL_n = bcL_np1

        # ---------- Plot Uex^n+1 ------------------
        plotExact = False
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
