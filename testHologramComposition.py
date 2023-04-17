# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:26:26 2023

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt

def fft2d(inp):
    ft = np.fft.ifftshift(inp)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def fft1d(inp):
    ft = np.fft.ifftshift(inp)
    ft = np.fft.fft(ft)
    return np.fft.fftshift(ft)

M = 1000
d = 100 # um
f = 0.1/1000
x = np.linspace(-d/2, d/2-1, M)*1e-6
y = np.array([0])#np.linspace(-d/2, d/2-1, M)*1e-6
# y.shape = (M, 1)
lensPhase = lambda x, y, f, lamb: -np.pi / lamb * (x**2+y**2)/f

flatPhase = lensPhase(x,y,0.01, 633e-9).flatten()
phase = (flatPhase + np.pi) % (2 * np.pi) - np.pi
# phase = np.reshape((lensPhase(x,y,f, 633e-9).flatten() + np.pi) % (2 * np.pi) - np.pi, [M,-1])

phase = phase*np.cos(x*2*np.pi/2e-6)
ftPhase = fft1d(phase)

intftPhase = np.abs(ftPhase)**2

fig, ax = plt.subplots()
ax.plot(intftPhase)
ax.set_yscale('symlog')
# cbar = plt.colorbar(f)

# f = plt.imshow(phase)
# cbar = plt.colorbar(f)



