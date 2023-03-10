# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:56:12 2023

@author: nicho
"""

import numpy as np
import scipy
from scipy.special import genlaguerre
import sys
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append("C:/Users/nicho/Documents/gitProjects/customPyDrivers")  # nicho
# sys.path.append('C:/Users/srv_plank/Documents/customPyDrivers') #planck

dirCurr = "C:/Users/nicho/Documents/gitProjects/lab-analysis-software/lundeen-lab/LGModes/"  # nicho
# dirCurr = 'L:/spaceplate/threeLensSpaceplate' #planck

from plot_custom import plot_custom

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, xcolor, siunitx}"

#%%

def LGBeam(l, m, r, phi, A = 1, w = 30):
    I = (A * (r*np.sqrt(2)/w)**np.abs(l)/w * genlaguerre(m,l)(2*r**2/w**2)*np.exp(-r**2/w**2) 
         * np.exp(-1j * l * phi) * np.exp(1j*(np.abs(l)+2*m + 1)))
    return I

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#%% create NxN array of normalized LGlm mode. 

def LGModeArray(l,m,N=100, M=100, w=20):
    x = np.arange(-np.floor(N/2), np.ceil(N/2))
    y = np.arange(-np.floor(M/2), np.ceil(M/2))
    xx,yy = np.meshgrid(x, y, indexing='ij')
    rr, pphi = cart2pol(xx, yy)
    
    field = LGBeam(l=l, m=m, r=rr, phi=pphi, w = w)
    return field/np.max(np.abs(field))

#%% Example: Plot the first l x m LG modes.

# L,M = 5,7

# N = 150 # even natural number
# x = np.arange(-N/2, N/2)
# xx,yy = np.meshgrid(x, x, indexing='ij')
# rr, pphi = cart2pol(xx, yy)

# space = 0.05*L/M
# fontsize = 25*L/M

# fig, ax = plt.subplots(nrows=L, ncols=M,figsize=(9.8, 10*L/M))
# plt.setp(ax, xticks=[], xticklabels = [], yticks = [], yticklabels = [], aspect = 'equal')
# fig.subplots_adjust(wspace=0.1, hspace=0.1)
# # gs1 = gridspec.GridSpec(4, 4)
# # gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

# for l in range(L):
#     for m in range(M):
#         field = LGBeam(l=l, m=m, r=rr, phi=pphi, w = N/5)
#         ax[l,m].imshow(np.abs(field)*0, cmap = 'hot')
#         ax[l,m].imshow(np.angle(field),  alpha = np.abs(field)**2/np.max(np.abs(field))**2, cmap = 'hsv', vmin = -np.pi, vmax = np.pi, interpolation = 'none')
        
# for i in range(L):
#     ax[i,0].set_ylabel(r'$l=$ ' + str(i), fontsize = fontsize)
    
# for i in range(M):
#     ax[0,i].set_title(r'$m=$ ' + str(i), fontsize = fontsize)
    
# dateTimeObj = datetime.now()
# preamble = (str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_')
# # plt.savefig(
# #     dirCurr + "figures/" + preamble + "LGModeslm" + str(l) + str(m) + ".png", dpi=300, bbox_inches="tight"
# # )


