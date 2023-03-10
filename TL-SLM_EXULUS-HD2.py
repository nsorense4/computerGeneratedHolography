# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:49:57 2023

@author: nicho
@purpose: driver for various functionality of the ThorLabs EXULUS-HD2 SLM

Specifications:
    wl: 400 - 850 nm
    res: 1920 x 1200 
    active area: 15.42 mm x 9.66 mm
    pitch: 8 um
    reflectance: 80 % (avg)
    phase range: 2 pi @ 633 nm
    frame rate: 60 hz
"""

import numpy as np
import scipy
import os
import matplotlib.pyplot as plt

from gaussianModes import LGModeArray
from gerchbergSaxtonAlgorithm import  *

dirPath = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", '/')

class EXULUS_HD2:
    
    xRange, yRange = 1920, 1200 # pixel
    pitch = 0.008 # mm
    
    def exportPhasePlot(l, m, figName, beamWidth = 1.5, modeWidth = 1, tol = 0.01, 
                        maxIter = 200, r = 1/2, adh = 0.95):
        # modeWidth and beamWidth are in mm
        
        
        farFieldField = LGModeArray(l=l,m=m, N=EXULUS_HD2.yRange, M=EXULUS_HD2.xRange, 
                                    w=modeWidth/EXULUS_HD2.pitch)
        nearFieldFld, farFieldFld, nearFieldFldFinal, farFieldFldFinal, err = (
            CGHGaussianInput(farFieldField, tol = tol, maxIter = maxIter, 
                              goal = 'amp+pha', r = r, adh = 0.9,
                              nearGaussWidth= beamWidth/EXULUS_HD2.pitch/EXULUS_HD2.yRange,
                              phaGuess = 'flat'))[:]
        
        nrows = 2
        ncols = 2

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(9.8, 10*nrows/ncols))
        plt.setp(ax, xticks=[], xticklabels = [], yticks = [], yticklabels = [], aspect = 'equal')
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        # plot black backgrounds
        for i in range(nrows):
            for j in range(ncols):
                ax[i,j].imshow(np.abs(farFieldField)*0, cmap = 'hot')

        # plot 
        plotIntAndPhase(nearFieldFld, ax[0,0])
        plotIntAndPhase(farFieldFld, ax[0,1])
        ax[1,0].imshow(np.angle(nearFieldFldFinal), cmap = 'hsv')
        plotIntAndPhase(farFieldFldFinal, ax[1,1])
        
        exportPlotPhase(nearFieldFldFinal, dirPath + '/figures/', figName + '.bmp')
        
        

