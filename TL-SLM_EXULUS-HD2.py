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

from gaussianModes import LGModeArray, HGModeArray
from gerchbergSaxtonAlgorithm import  *

dirPath = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", '/')

class EXULUS_HD2:
    
    xRange, yRange = 1920, 1200 # pixel
    pitch = 0.008 # mm
    
    def exportPhasePlot(l, m, figName, beamWidth = 1.5, modeWidth = 1, tol = 0.01, 
                        maxIter = 200, r = 1/2, adh = 0.95, gaussMode = 'lg'):
        # modeWidth and beamWidth are in mm
        
        if gaussMode == 'lg':
            farFieldField = LGModeArray(l=l,m=m, N=EXULUS_HD2.yRange, M=EXULUS_HD2.xRange, 
                                    w=modeWidth/EXULUS_HD2.pitch)
        elif gaussMode == 'hg':
            farFieldField = HGModeArray(l=l,m=m, N=EXULUS_HD2.yRange, M=EXULUS_HD2.xRange, 
                                    w=modeWidth/EXULUS_HD2.pitch)
        nearFieldFldFinal = (CGHGaussianInput(farFieldField, tol = tol, maxIter = maxIter, 
                              goal = 'amp+pha', r = r, adh = 0.9,
                              nearGaussWidth= beamWidth/EXULUS_HD2.pitch/EXULUS_HD2.yRange,
                              phaGuess = 'flat'))[3]

        
        exportPlotPhase(nearFieldFldFinal, dirPath + '/figures/', figName + '.bmp')
        
    def exportPhasePlotFromImageAmp(image, figName, beamWidth = 1.5):
        
        farFieldField = loadImage(image, dirPath)
        
        if np.shape(farFieldField) != (EXULUS_HD2.yRange, EXULUS_HD2.xRange):
            farFieldField = EXULUS_HD2.padArray(farFieldField)        
        
        nearFieldFldFinal = CGHGaussianInput(farFieldField, tol = 0.001, maxIter = 100,
                                        goal = 'amp',
                                        nearGaussWidth= beamWidth/EXULUS_HD2.pitch/EXULUS_HD2.yRange,
                                        phaGuess = 'flat')[3]
        
        
        exportPlotPhase(nearFieldFldFinal, dirPath + '/figures/', figName + '.bmp')
        
    def padArray(A):
        (w,h) = np.shape(A)
        
        A = np.pad(A, ((np.floor((EXULUS_HD2.yRange-h)/2).astype('int'),np.ceil((EXULUS_HD2.yRange-h)/2).astype('int')), 
                       (np.floor((EXULUS_HD2.xRange-w)/2).astype('int'),np.floor((EXULUS_HD2.xRange-w)/2).astype('int'))), 'constant')
        
        return A

