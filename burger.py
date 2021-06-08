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
        "u_xyt"   : lambda x, y, t, nu, a: (2*np.pi*nu)*((np.sin(np.pi*x)*np.exp(-np.pi**2*nu*t))/\
                (a + np.cos(np.pi*x)*np.exp(-np.pi**2*nu*t ))) + y*0,
        "v_xyt"   : lambda x, y, t: 0*x + 0*y + 0*t

        }


u_xyt = functions["u_xyt"]
v_xyt = functions["v_xyt"]

dxdy = []
L2 = []
Linf = []
acc = 0
qBC_nm1 = {}
qBC = {}

dt = .01
T = 1
Nt = 100 #int(T/dt)
t = np.linspace(0, Nt*dt, Nt)
alpha = .5 # Crank-Nicholson 
nu = 0.05
a = 2

grid = zip(dx, dy, nx, ny, q_size)
for dxi, dyi, nxi, nyi, q_sizei in grid:
    time = []
    Xu_data = [] 
    Tu_data = []
    U_data  = []
    
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
    U = np.reshape(u_xyt(Xu, Yu, t0, nu, a), (1, nyi*(nxi-1)))
    V = np.reshape(v_xyt(Xv, Yv, t0), (1, nxi*(nyi-1))) 
    
    q_nm1 = np.concatenate( (U, V), axis = 1)
    q_nm1 = q_nm1[0] 
    
    # ---------- Set Boundary Conditions -----------------------
    
    # Top Wall BC
    qBC_nm1["uT"] = u_xyt(xu,Ly, dt*t0, nu, a)
    qBC_nm1["vT"] = v_xyt(xv,Ly, dt*t0)
    # Bottom Wall BC
    qBC_nm1["uB"] = u_xyt(xu,0, dt*t0, nu, a)
    qBC_nm1["vB"] = v_xyt(xv,0, dt*t0) 
    # Left Wall BC
    qBC_nm1["uL"] = u_xyt(0,yu, dt*t0, nu, a)
    qBC_nm1["vL"] = v_xyt(0,yv, dt*t0)
    # Right Wall BC
    qBC_nm1["uR"] = u_xyt(Lx,yu, dt*t0, nu, a)
    qBC_nm1["vR"] = v_xyt(Lx,yv, dt*t0)
    
     
    # ---------- SOLVE FOR u(x,y,tn) WHERE n = 1 ------------
    # ---------- Set Boundary Conditions for n+1 ------------
    
    U = np.reshape(u_xyt(Xu, Yu, dt*(t0+1), nu, a), (1, nyi*(nxi-1)))
    V = np.reshape(v_xyt(Xv, Yv, dt*(t0+1)), (1, nxi*(nyi-1))) 
    
    q_n = np.concatenate( (U, V), axis = 1)
    q_n = q_n[0] 
    
    # Top Wall BC
    qBC["uT"] = u_xyt(xu,Ly, dt*(t0+1), nu, a)
    qBC["vT"] = v_xyt(xv,Ly, dt*(t0+1))
    # Bottom Wall BC
    qBC["uB"] = u_xyt(xu,0, dt*(t0+1), nu, a)
    qBC["vB"] = v_xyt(xv,0, dt*(t0+1)) 
    # Left Wall BC
    qBC["uL"] = u_xyt(0,yu, dt*(t0+1), nu, a)
    qBC["vL"] = v_xyt(0,yv, dt*(t0+1))
    # Right Wall BC
    qBC["uR"] = u_xyt(Lx,yu, dt*(t0+1), nu, a)
    qBC["vR"] = v_xyt(Lx,yv, dt*(t0+1))
    
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
    for tn in range(1, Nt):
    
        # ---------- Set Boundary Conditions for n+1 ------------
    
        # Top Wall BC
        qBC["uT"] = u_xyt(xu,Ly, dt*(tn+1), nu, a)
        qBC["vT"] = v_xyt(xv,Ly, dt*(1+tn))
        # Bottom Wall BC
        qBC["uB"] = u_xyt(xu,0, dt*(tn+1), nu, a)
        qBC["vB"] = v_xyt(xv,0, dt*(1+tn)) 
        # Left Wall BC
        qBC["uL"] = u_xyt(0,yu, dt*(tn+1), nu, a)
        qBC["vL"] = v_xyt(0,yv, dt*(1+tn))
        # Right Wall BC
        qBC["uR"] = u_xyt(Lx,yu, dt*(tn+1), nu, a)
        qBC["vR"] = v_xyt(Lx,yv, dt*(1+tn))
    
        bcL_np1 = op.bclap(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)

        # ---------- Set RHS of Ax = b for Diffusion Eq.  --------
        bc = np.multiply(0.5*dt*nu, np.add(bcL_n, bcL_np1))
        Sq_n = op.S(q_n, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, alpha, nu, dt, pinned=False)
        
        Aq_nm1 = op.adv(q_nm1, qBC_nm1, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        Aq_n = op.adv(q_n, qBC, ui, vi, pi, dxi, dyi, nxi, nyi, q_sizei, pinned=False)
        adv = np.multiply(0.5*dt, np.subtract(np.multiply(3, Aq_n), Aq_nm1))
        
        b = Sq_n + bc + adv
        
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
                    (np.reshape(u_xyt(Xu, Yu, (1+tn)*dt, nu, a), (1, nyi*(nxi-1))),\
                    np.reshape(v_xyt(Xv, Yv, (1+tn)*dt), (1, nxi*(nyi-1)))),\
                    axis = 1)
        qu_np1_ex = q_np1_ex[0][0:nyi*(nxi-1)]

        error = LA.norm(qu_np1 - qu_np1_ex, np.inf)
        if (tn % 20) == 0:
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
        
        # ---------- Save X-Data at y = 0.5 ------------------
        plotXTime = True
        if plotXTime:
            q_u = np.reshape(q_n[0:nyi*(nxi-1)], (Xu.shape)) 
            U_data.append(q_u[5])
            time.append(tn*dt)
            #plt.plot(xu, q_u[5])

        q_nm1 = q_n
        qBC_nm1 = qBC
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
            ax.set_zlim(0, 0.5)
            ax.set_xlabel('$xu$')
            ax.set_ylabel('$yu$')
            ax.view_init(30, 45)
            plt.show()
    
    #print(U_data)
    Xu_data, Tu_data = np.meshgrid(xu, time)
    U_data = np.array(U_data)
    print(Xu_data)
    print(Tu_data)
    print(time)
    print(Xu_data.shape)
    print(U_data.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #plt.contourf(Xu, Yu, q_u_exact)
    #ax.contour3D(Xu, Yu, q_u_exact, 50)
    surf = ax.plot_surface(Xu_data, Tu_data, U_data, rstride=1, cstride=1,\
            cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_zlim(0, 0.2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$time$')
    #ax.view_init(30, 45)
    plt.show()
