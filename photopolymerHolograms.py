# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:05:32 2023

@author: nicho
"""

# %%
import csv
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import pandas as pd
from scipy.optimize import curve_fit

sys.path.append("C:/Users/nicho/Documents/gitProjects/customPyDrivers")  # nicho
sys.path.append("C:/Users/nicho/Documents/gitProjects/lab-analysis-software/lundeen-lab/hologram/")
# sys.path.append('C:/Users/srv_plank/Documents/customPyDrivers') #planck

dirCurr = "C:/Users/nicho/Documents/gitProjects/lab-analysis-software/lundeen-lab/hologram/computerGeneratedHolography/"  # nicho
# dirCurr = 'L:/spaceplate/threeLensSpaceplate' #planck

from plot_custom import plot_custom

from hologram_mode_solver import modeSolver

# %%

def readCSV(filename):
    """
    Input: directory to a four column row CSV file
    Output: array of all CSV values

    """
    from numpy import genfromtxt
    
    my_data = genfromtxt(filename, delimiter=',')
    return my_data

def plotHologramEff(filename):
    data = readCSV(filename)
    

    fig, ax = plot_custom(
        6, 6, r"angle ($^\circ$)", 'efficiency, $\eta$ (au)', nrow=1, axefont = 27, labelPad = 15, spineColor = 'black', tickColor='white',
        axew=2.5
    )
    
    # ax.set_ylim([0, 1])
    
    ax.plot(data[0,:], data[1,:])
    
    return data


# nu = lambda nvar, lamb, theta0, d: np.pi*nvar*d/lamb/np.cos(theta0)
ksi = lambda scalar, angleRad, theta0Rad: scalar * (angleRad - theta0Rad)

def fresnel(theta, n1, n2):
    # assumes s polarized, returns reflectivity
    num = n1* np.cos(theta) - n2*np.sqrt(1-(n1*np.sin(theta)/n2)**2)
    den = n1* np.cos(theta) + n2*np.sqrt(1-(n1*np.sin(theta)/n2)**2)
    return (num/den)**2

def fEfficiency(angleRad, scalar,theta0Rad, nu):
    return np.sin((ksi(scalar, angleRad, theta0Rad)**2 + nu**2)**0.5)**2/(1 + ksi(scalar, angleRad, theta0Rad)**2/nu**2)
  

def fitDE(xData, yData, braggAngleDeg):
  
    popt, pcov = curve_fit(fEfficiency, np.deg2rad(xData), yData, p0=[22,  np.deg2rad(braggAngleDeg), np.pi/2],
                        bounds=([0,  np.deg2rad(braggAngleDeg-90), 0.1*np.pi/2], 
                                [1000,  np.deg2rad(braggAngleDeg+90), 5*np.pi/2]),
                       maxfev=10000)
    return popt
    
def plotHologramEffs(filenames, config = 'eff', offset = 0, fitBool = False):
    
    if config == 'eff':
        fig, ax = plot_custom(
            10, 6, r"input mode angle, $\theta_0$ ($^\circ$)", 'efficiency, $\eta$ (au)', nrow=1, axefont = 27, spineColor = 'black',
            axew=2.5, 
        )
        ax.set_ylim([0, 1])
        
        labels = [r'$\{3,-1 \}_{\text{EXP}}$', r'$\{3,1 \}_{\text{EXP}}$', r'$\{3,-1 \}_{\text{EXP}}$', r'$\{3,-2 \}_{\text{EXP}}$']
        colors = ['darkorange', 'red', 'darkred', 'gold']
        i = -1
        for name in filenames:
            i += 1
            data = readCSV(name)

            ax.plot(data[0,:]+offset[i], data[1,:]*(1- fresnel(np.deg2rad(data[0,:]+offset[i]), n1, n2)), zorder = 10,
                    color = colors[i], label = labels[i], linewidth = 2 )
                        
            if fitBool == True:
                popt = fitDE(data[0,:]+offset[i], data[1,:]*(1- fresnel(np.deg2rad(data[0,:]+offset[i]), n1, n2)), offset[i])
                print(popt)
                ax.plot(data[0,:]+offset[i], fEfficiency(np.deg2rad(data[0,:]+offset[i]), popt[0], popt[1], popt[2]),
                        linewidth = 2, color = 'k', label = r'fit')
            
    elif config == 'power':
        fig, ax = plot_custom(
            10, 6, r"input mode angle, $\theta_0$ ($^\circ$)", 'power (mW)', nrow=1, axefont = 27, spineColor = 'black',
            axew=2.5, 
        )
        
        i = -1
        for name in filenames:
            i += 1
            data = readCSV(name)
            
            maxData = np.max(data[2,:]+ data[3,:])
        
            ax.plot(data[0,:]+offset[i], data[2,:])
            ax.plot(data[0,:]+offset[i], data[3,:])

    # theta = np.linspace(-90, 90, 1000)
    # ax.plot(theta, fEfficiency(np.deg2rad(theta), 53, np.deg2rad(44.7), 1.13), label = 'forced')

    ax.set_xlim([-90, 90])
    ax.grid()
    
    return fig, ax
    

def plotHologramTheoryAndResponse(ax, theta0 = 45.2, phi = 90, d= 0.93, epsilonRatio = 0.023, lamb = 0.04,
                                  N = 1000,M = 5,n1 = 1,n2 = 1.5,
                                  n3 = 1.5, theoryBool = True, saveBool = False, lambAir = 632.8e-9,
                                  plots2Show = [1, 0, -1], xmin = -90, xmax = 90,
                                  legendTitle = r'$d = \SI{14.7}{\micro m}$\\ $\frac{\Delta \epsilon}{\epsilon_2}=0.0230$',
                                  fileName = 'test', legendBool = True, legPos = 'upper right'):
    if theoryBool == True:


        DE1j = np.zeros([N, M])
        DE3j = np.zeros([N, M])
        theta0 = np.deg2rad(theta0)
        theta1 = np.deg2rad(np.linspace(-90, 90, N))
        
        phi = np.deg2rad(phi) # peaks are 37 degrees apart
        

        dRatio = lambda epsilonRatio, theta20, phi: 2 / epsilonRatio * np.cos(theta20) * np.cos(phi - theta20)
    
        dRatioNum = dRatio(epsilonRatio, theta0, phi )
    
        print(dRatioNum)
    
        for i in range(N):
            DE1j[i, :], DE3j[i, :], rj, tj, __, __, __, __, __, __, dExp = modeSolver(
                phi= phi,
                theta0=theta0,
                epsilonRatio=epsilonRatio,
                jStart=int((M-1)/2),
                jEnd=int((1-M)/2),
                dRatio=dRatioNum,
                theta1=theta1[i],
                d=d,
                e1=n1**2,
                e2=n2**2,
                e3=n3**2,
                lamb=lamb,
            )[0:11]
        print(dExp)
        print(dExp*lambAir/lamb)
        print(epsilonRatio/2*n2)
            
        # theta1 = np.arcsin(np.sin(theta1)*n2)
        plotLabels3 = [r"$\{ 3,2 \}$", r"$\{ 3,1 \}$", r"$\{ 3,0 \}$", r"$\{ 3,-1 \}$", r"$\{ 3,-2 \}$"]
        plotStyle = [':', ':', '-.', '--', '--']
        for plotNum in plots2Show:
            ax.plot(
                np.degrees(theta1),
                DE3j[:,plotNum+ int(np.floor(M/2))],
                linestyle=plotStyle[plotNum+ int(np.floor(M/2))],
                color="black",
                linewidth=2,
                label=plotLabels3[plotNum+ 2],
            )

        ax.plot(
            np.degrees(theta1),
            DE1j[:,int(np.floor(M/2))],
            linestyle="-",
            color="black",
            linewidth=2,
            label=r"$\{ 1,1 \}$",
        )
    if legendBool is True:
        leg = ax.legend(fontsize=18, loc=legPos, title = legendTitle)
        leg.get_title().set_fontsize('18')
    ax.set_xlim([xmin,xmax])
    # plotHologramResponse(dirCurr + 'data/2023417-15_39_31angleSweep_holo4_mode1_center.csv')
    
    if saveBool == True:
        dateTimeObj = datetime.now()
        preamble = (str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_' + 
                    str(dateTimeObj.hour)+ '_' + str(dateTimeObj.minute) + '_')
        plt.savefig(
            dirCurr + "/figures/" + preamble + "AngularDEHologram45DegExpZoom.png", dpi=300, bbox_inches="tight"
        )

    
    
if __name__ == "__main__":
    # 20230415
    filenames0 = [dirCurr + "data/2023417-15_39_31angleSweep_holo4_mode1_center.csv",
                  dirCurr + "data/2023417-15_28_41angleSweep_holo4_mode0_center.csv"]
    offset0  = [-445, 45]
    # 20230418-11
    filenames1 = [dirCurr + "data/2023418-11_17_28angleSweep_holo4_mode0_center.csv",
                  dirCurr + "data/2023418-11_31_26angleSweep_holo4_mode0_center.csv"]
    offset1  = [44.5, -45]
    # 20230418-12
    filenames2 = [dirCurr + "data/2023418-12_41_24angleSweep_holo4_mode1_center.csv",
                  dirCurr + "data/2023418-12_55_17angleSweep_holo4_mode0_center.csv",
                  dirCurr + "data/2023418-13_39_1angleSweep_holo4_mode2_center.csv"]
    offset2 = [-22.5, 22.5, 22.5]
    # 20230419
    filenames3 = [dirCurr + "data/2023420-11_41_15angleSweep_holo4_mode1_center.csv",
                  dirCurr + "data/2023420-11_59_16angleSweep_holo4_mode2_center.csv",
                  dirCurr + "data/2023420-12_16_18angleSweep_holo4_mode-1_center.csv"]
    offset3 = np.array([30,30,30])*-1 + 22.5
    
    # 
    filenames4 = [dirCurr + "data/2023426-12_59_4angleSweep_holo45_mode-1.csv",
                  dirCurr + "data/2023426-11_4_48angleSweep_holo45_mode1.csv"]
    offset4 = np.array([30,30,30])*0
    
    filenames5 = [dirCurr + "data/2023426-14_4_36angleSweep_holo18_mode1.csv",
                  dirCurr + "data/2023426-14_18_16angleSweep_holo18_mode2.csv",
                  dirCurr + "data/2023426-14_30_48angleSweep_holo18_mode-1.csv",
                  dirCurr + "data/2023426-14_42_6angleSweep_holo18_mode-2.csv"]
    offset5 = np.array([30,30,30,30])*0
    
    # 
    filenames6 = [dirCurr + "data/2023426-15_28_20angleSweep_holo18Offset_mode-1.csv",
                  dirCurr + "data/2023426-15_12_37angleSweep_holo18Offset_mode1.csv"]
    offset6 = np.array([30,30,30])*0 
    
    # plot the 45 degree hologram

    # fig, ax = plotHologramEffs(filenames4, config = 'eff', offset = offset4, fitBool=(False))
    
    
    # plotHologramTheoryAndResponse(  ax, theta0 = 45.2, phi = 90, d= 0.93, epsilonRatio = 0.023, lamb = 0.04,
    #                                 N = 1000,M = 5,n1 = 1,n2 = 1.5,
    #                                 n3 = 1.5, theoryBool = True, saveBool = False,
    #                                 plots2Show = [1, 0, -1], xmin = 30, xmax = 60,
    #                                 legendTitle = r'$d = \SI{14.7}{\micro m}$\\ $\frac{\Delta \epsilon}{\epsilon_2}=0.0230$' ,
    #                                 fileName = 'AngularDEHologram45DegExpZoom'
    #                               )
    
    # fig, ax = plotHologramEffs(filenames5, config = 'eff', offset = offset5, fitBool=(False))
    
    
    # plotHologramTheoryAndResponse(  ax, theta0 = 18.8, phi = 90, d= 1, epsilonRatio = 0.019, lamb = 0.04,
    #                                 N = 1000,M = 9, theoryBool = True, saveBool = True,
    #                                 plots2Show = [2, 1, 0, -1,-2], xmin = 0, xmax = 60,
    #                                 legendTitle = r'$d = \SI{15.8}{\micro m}$\\ $\frac{\Delta \epsilon}{\epsilon_2}=0.0190$', 
    #                               fileName = 'AngularDEHologram18DegExpZoom')
    
    fig, ax = plotHologramEffs(filenames6, config = 'eff', offset = offset6, fitBool=(False))
    
    
    plotHologramTheoryAndResponse(  ax, theta0 = 50.2, phi = 90+19.45, d= 1.0, epsilonRatio = 0.021, lamb = 0.04,
                                    N = 1000,M = 5, theoryBool = True, saveBool = False,
                                    plots2Show = [1, 0, -1], xmin = -90, xmax = 90,
                                    legendTitle = r'$d = \SI{15.8}{\micro m}$\\ $\frac{\Delta \epsilon}{\epsilon_2}=0.0210$', 
                                  fileName = 'AngularDEHologram18DegExpZoom', legendBool = True, legPos = 'upper left')



