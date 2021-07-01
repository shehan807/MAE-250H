import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

import visualization as vis 
import config as cfg 

cfg.init('inputsMAIN.txt')
skip = (slice(None, None, 5), slice(None, None, 5))

dataFormat = '_Data_dt_{:.3e}'.format(cfg.dt).replace('.','p') + '.npy'
simVars = {'X':[], 'Y':[], 'U':[], 'V':[]}

fig, ax  = plt.subplots()

for var in simVars:
    with open(cfg.outputPath + var + dataFormat, 'rb') as data:
        simVars[var] = np.load(data)

X, Y, U, V = simVars['X'], simVars['Y'], simVars['U'], simVars['V']
U = np.delete(U, 1, 0)
V = np.delete(V, 1, 0)
t_n = 0.0
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel(r'$X$', fontsize=16)
ax.set_ylabel(r'$Y$', fontsize=16)

levels = np.linspace(0,1,21)
cntrf = ax.contourf(X, Y, np.sqrt(U[0]**2 + V[0]**2), levels=levels, cmap=cm.viridis)
cbar = plt.colorbar(cntrf, format='%.2f')
cbar.set_label('Velocity Magnitude', fontsize=14)
cbar.ax.tick_params(labelsize=14)
ax.set_title(r"Velocity Distribution at t = {:.3f}".format(t_n), fontsize=20) 

#quiv = ax.quiver(X[skip], Y[skip], U[0][skip], V[0][skip], color='white', width=0.003)
#strm = ax.streamplot(X, Y, U[0], V[0], color='white', linewidth=.5)

def animate(i):
    
    global X, Y, t_n, quiv
    
    Ui = U[i]
    Vi = V[i]
    
    ax.clear()
    ax.contourf(X, Y, np.sqrt(Ui**2 + Vi**2), levels=levels, cmap=cm.viridis)
    #quiv.set_UVC(Ui, Vi)
    #ax.quiver(X[skip], Y[skip], Ui[skip], Vi[skip], color='white', width=0.003)
    #ax.streamplot(X, Y, Ui, Vi, color='white', linewidth=.5)
    ax.set_title(r"Velocity Distribution at t = {:.3f}".format(t_n), fontsize=20) 
    
    t_n += cfg.dt


anim = animation.FuncAnimation(fig, animate, np.arange(1,int(U.shape[0])), interval=10, blit=False, repeat=False)
anim.save(cfg.outputPath + 'animation.mp4')
#plt.show()
