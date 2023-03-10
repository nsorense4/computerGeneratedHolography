# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:44:11 2023

@author: nicho
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from PIL import Image

sys.path.append("C:/Users/nicho/Documents/gitProjects/customPyDrivers")  # nicho
# sys.path.append('C:/Users/srv_plank/Documents/customPyDrivers') #planck

dirCurr = "C:/Users/nicho/Documents/gitProjects/lab-analysis-software/lundeen-lab/hologram/CGH/"  # nicho
# dirCurr = 'L:/spaceplate/threeLensSpaceplate' #planck

from plot_custom import plot_custom
from gaussianModes import LGModeArray

#%%
def fft2d(inp):
    ft = np.fft.ifftshift(inp)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def ifft2d(inp):
    ft = np.fft.ifftshift(inp)
    ft = np.fft.ifft2(ft)
    return np.fft.fftshift(ft)

def loadImage(farFieldFilename, dirCurr = dirCurr):
    # load the image into an array (1 is white, 0 is black). Load a flat phase. 
    farFieldAmp = 1 - np.mean(plt.imread(dirCurr+farFieldFilename, 0)[:,:,0:3], axis=2)/255 # far field amplitude
    farFieldField = farFieldAmp*np.exp(1j*0)
    return farFieldField
    
def CGHGaussianInput(farFieldFld, tol = 0.12, maxIter = 100, goal = 'amp', 
                     phaGuess = 'flat', N = 100, adh = 0.95, r = 1/2, 
                     nearGaussWidth = 0.2):
    '''
    purpose:    To calculate the phase mask required to produce some arbitrary
                far field field amplitude given a monochromatic Gaussian input.
                
    inputs:
        farFieldFilename - directory to the image file defining the far field 
            field amplitude (string)
        tol - the required tolerance for the calculated phase mask between the
            final and penultimate iterations (float)
        maxIter - number of iterations before returning the optimized phasemask
        goal - 'amp' or 'amp+pha'; choice of whether to set the amplitude and let
            the phase change, or the amplitude and phase within a certain boundary
            and let the amplitude and phase outside of the boundary change
        phaGuess - selects the starting guess of the near field phase: 'flat',
            a flat field phase;'circ', a circular perdiodic field phase; or 'fft', 
            which takes the fourier transform of the far field target and selects
            its phase. 
            
    outputs:
        nearFieldFld - initial input field and phase guess
        farFieldFld - target field at the far field        
        nearFieldFldFinal - near field phase mask calculated to produce the far field
            field amplitude (array/image)
        farFieldFldFinal - far field result (array/image)
        err - avg difference between intended field and resulting field.
    '''

    farFieldAmp = np.abs(farFieldFld)
    farFieldPha = np.angle(farFieldFld)
    (ln, ht) = np.shape(farFieldAmp)    # length and height of the image
    nearFieldAmp = np.abs(LGModeArray(0,0, N=ln, M=ht, w=np.min([ln,ht])*nearGaussWidth))    # starting near field amplitude array
    
    if phaGuess == 'flat':
        nearFieldPha = nearFieldAmp*0       # start with a flat phase distribution guess
    elif phaGuess == 'circ':
        nearFieldPha = np.ones([ln, ht]) 
        for i in range(ln):
            for j in range(ht):
                I,J= float(i), float(j)
                LN,HT= float(ln), float(ht)
                nearFieldPha[i,j] = np.arctan2((J-ht/2),(I-ln/2))
    elif phaGuess == 'fft':
        nearFieldPha = np.angle(fft2d(farFieldFld))
                
    nearFieldFld = nearFieldAmp*np.exp(1j*nearFieldPha) # this is a starting guess
                
    goalZoneIter = np.ones([ln, ht]) 
    goalZoneTarg = np.ones([ln, ht]) 
    
    r = r*min([ln, ht])
    for i in range(ln):
        for j in range(ht):
            # calc goal zone array
            if (i-ln/2)**2 + (j-ht/2)**2 > r**2:
                goalZoneIter[i,j] = 1
                goalZoneTarg[i,j] = 0
            else:
                goalZoneIter[i,j] = 1-adh
                goalZoneTarg[i,j] = 1


        
    # Enforced parameter: amplitude of near field beam (gaussian)
    # Goal parameter: amplitude of the far field
    # Changing parameter: pahse of the near field beam 
    
    A = fft2d(nearFieldFld) # far field given the assumed starting phase
    counter = 0 # set a maximum number of iterations
    err = 1
    if goal == 'amp':
        while err > tol:        
            B = farFieldAmp*np.exp(1j*np.angle(A))
            C = ifft2d(B)
            D = nearFieldAmp*np.exp(1j*np.angle(C)) 
            A = fft2d(D) # far field given the iterated phase
            err = np.mean(np.abs(np.abs(A)/np.max(np.abs(A))-farFieldAmp))
            
            if counter > maxIter:
                print("Number of iterations exceeded max. Err is "+ str(err)+'.')
                break 
            counter += 1
            
    elif goal == 'amp+pha':
        while err > tol:        
            B = calcGoalField(goalZoneIter, goalZoneTarg, farFieldAmp, farFieldPha, normalize(A), adh = adh)
            C = normalize(ifft2d(B))
            D = normalize(nearFieldAmp*np.exp(1j*np.angle(C)))
            
            if counter >= maxIter:
                # return nearFieldFld, farFieldFld, normalize(A), B, C, D
                A = normalize(fft2d(D)) # far field given the iterated phase
                print("Number of iterations (" + str(maxIter) + ") exceeded max.")
                break 
            counter += 1
            
            A = normalize(fft2d(D)) # far field given the iterated phase
            err = np.mean(goalZoneTarg/adh*np.abs(np.abs(A)-farFieldAmp))
            
    print("Err is "+ str(err)+'.')
    
    nearFieldFldFinal = D/np.max(np.abs(D))
    farFieldFldFinal = A/np.max(np.abs(A))
        
    return nearFieldFld, farFieldFld, nearFieldFldFinal, farFieldFldFinal, err

def findMaxAmp(field):
    return np.max(np.abs(field))

def normalize(field):
    return field/findMaxAmp(field)

def calcGoalField(goalZoneIter, goalZoneTarg, setAmp, setPha, iterField, adh):
    BAmp = setAmp*goalZoneTarg + np.abs(iterField)*goalZoneIter
    BPha = setPha*goalZoneTarg + np.angle(iterField)*goalZoneIter
    return BAmp*np.exp(1j*BPha)

def plotIntAndPhase(field, ax):
    ax.imshow(np.angle(field),  alpha = np.abs(field)**2/np.max(np.abs(field))**2, cmap = 'hsv', vmin = -np.pi, vmax = np.pi, interpolation = 'none')
    
def boolFileExists(path):
    from pathlib import Path
    path = Path(path)
    return path.is_file()
    
def exportPlotPhase(field, dirCurr, fileName):
    
    arr = ((np.angle(field)+ np.pi)*255/2/np.pi).astype('uint8')
    im=Image.fromarray(arr, mode = 'L')
    
    dateTimeObj = datetime.now()
    preamble = (str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_')
    dirFig = dirCurr + preamble + fileName

    if boolFileExists(dirFig) == False:
        im.save(dirFig)
    else:
        print('Warning: file not saved as there is a preexisting file with the specified name.')


# %% Example: Calculate the amplitude phase mask for a field defined by a picture
# farFieldField = loadImage('figures/smiley.bmp', dirCurr)
# # farFieldField = LGModeArray(0,2, N=100, M=100, w=15)
# nearFieldFld, farFieldFld, nearFieldFldFinal, farFieldFldFinal, err = (
#     CGHGaussianInput(farFieldField, tol = 0.001, maxIter = 100))
# nrows = 2
# ncols = 2

# fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(9.8, 10))
# plt.setp(ax, xticks=[], xticklabels = [], yticks = [], yticklabels = [], aspect = 'equal')
# fig.subplots_adjust(wspace=0.1, hspace=0.1)

# # plot black backgrounds
# for i in range(nrows):
#     for j in range(ncols):
#         ax[i,j].imshow(np.abs(farFieldField)*0, cmap = 'hot')

# # plot the intial guesses and targets, as well as the results
# plotIntAndPhase(nearFieldFld, ax[0,0])
# plotIntAndPhase(farFieldFld, ax[0,1])
# plotIntAndPhase(nearFieldFldFinal, ax[1,0])
# plotIntAndPhase(farFieldFldFinal, ax[1,1])
# fontsize = 25
# ax[0,0].set_title('Near-Field Guess', fontsize = fontsize)
# ax[0,1].set_title('Far-Field Target', fontsize = fontsize)
# ax[1,0].set_title('Near-Field Result', fontsize = fontsize)
# ax[1,1].set_title('Far-Field Result', fontsize = fontsize)

# # dateTimeObj = datetime.now()
# # preamble = (str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_')
# # dirFig = dirCurr + "figures/" + preamble + "smiley" + ".png"

# # if boolFileExists(dirFig) == False:
# #     plt.savefig(
# #         dirFig, dpi=300, bbox_inches="tight"
# #     )
# # else:
# #     print('Warning: file not saved as there is a preexisting file with the specified name.')


# # %% Example: Calculate the goaled phase mask for a field
# # farFieldField = loadImage('figures/smiley.bmp', dirCurr)
# farFieldField = LGModeArray(1,1, N=200, M=200, w=20)
# nearFieldFld, farFieldFld, nearFieldFldFinal, farFieldFldFinal, err = (
#     CGHGaussianInput(farFieldField, tol = 0.001, maxIter = 500, 
#                      goal = 'amp+pha', r = 0.25, adh = 0.9, nearGaussWidth= 0.05,
#                      phaGuess = 'flat'))
# nrows = 2
# ncols = 2

# fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(9.8, 10*nrows/ncols))
# plt.setp(ax, xticks=[], xticklabels = [], yticks = [], yticklabels = [], aspect = 'equal')
# fig.subplots_adjust(wspace=0.1, hspace=0.1)

# # plot black backgrounds
# for i in range(nrows):
#     for j in range(ncols):
#         ax[i,j].imshow(np.abs(farFieldField)*0, cmap = 'hot')

# # plot 
# plotIntAndPhase(nearFieldFld, ax[0,0])
# plotIntAndPhase(farFieldFld, ax[0,1])
# ax[1,0].imshow(np.angle(nearFieldFldFinal), cmap = 'hsv')
# plotIntAndPhase(farFieldFldFinal, ax[1,1])

# exportPlotPhase(nearFieldFldFinal, dirCurr + 'figures/', 'lm11Phase.bmp')




