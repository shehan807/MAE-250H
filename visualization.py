"""
Created on May 27 2021
@author S. M. Parmar
Various visualization routines for 
verification and flow visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

def plotL2vsGridSize(linReg, dxdy, error, outFile, oprtr, save=False):
    """
    INPUTS:
    ------
    linReg - linear regression data from linregress function
    dxdy   - array of spatial grid sizes (x-axis)
    error  - array of error values (y-axis)
    oprtr  - string value name of the operator being tested (for title)
    outFile- name of output file for figure
    """
    figFilePath = "./Figures/"
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.rc('grid', c='0.5', ls='-', alpha=0.5, lw=0.5)
    
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r'$\Delta$ $x$, $\Delta$ $y$', fontsize=16)
    ax.set_ylabel(r'$L^{\infty}$ Norm, $||x||_{\infty}$', fontsize=16)
    ax.set_title(r"Spatial Convergence of " + oprtr + " Operator", fontsize=20) 
    ax.annotate(r"Log-Log Slope = $%.2f$" % (linReg.slope), 
            xy=(0.75, 0.05), 
            xycoords="axes fraction",
            size=16,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round4", fc="aqua", ec="k", alpha=0.7))

    plt.loglog(dxdy, error, 'bo', mfc="none", markersize=8, label=oprtr + ' Operator Tests')
    plt.loglog(dxdy, 10**(linReg.slope*np.log10(dxdy)+linReg.intercept), '-r', label='Fitted Line',linewidth=2)
    plt.legend(prop={'size':14})
    plt.grid(True, which="both")
    if save:
        plt.savefig(figFilePath + outFile.split('.')[0])
    plt.show()

    return
