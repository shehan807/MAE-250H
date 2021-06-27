""" 
Create June 6th, 2021
@author: Shehan Parmar
Python routine for modal analysis of 
lid-driven cavity. 
"""
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm
def modal_analysis(data, x, y):
    """
    data -- numpy stack of arrays (i.e. U(x,y,t1), U(x,y,t2), ... U(x,y,tn)
    """
    POD, sing_val, temp = LA.svd(data)
    LidDrivenRecon = np.matrix(POD[:, :20])*np.diag(sing_val[:20])*np.matrix(temp[:20, :])
    plt.imshow(LidDrivenRecon, cmap=cm.viridis)
    plt.show()
    #plt.contourf(Xu, Yu, POD)
    print(POD[:,0].shape)
    print(x.shape)
    print(y.shape)
