# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:46:34 2023

@author: nicho
"""

import numpy as np
import scipy
from scipy.optimize import root_scalar

def sincInverse(y):
    # calculate the inverse of sinc = sin(x)/x over [-pi,0]
    sinc = lambda x: np.sin(x)/x
    sincZero = lambda x: np.sin(x) - x*y
    return root_scalar(sincZero, bracket=[-np.pi, -1e-9])
    

class SLMKarimiMethod:
    
    def calcSLMPhase(fieldInt, fieldPha):
        nCount,mCount = np.shape(fieldInt)
        n = np.arange(0, nCount) + 1
        m = np.arange(0, mCount) + 1
        M = 1/np.pi* sincInverse(fieldInt) + 1
        F = fieldPha - m*M 
        
        psi = 
        
        
        