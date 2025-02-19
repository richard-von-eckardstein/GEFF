import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import inspect
import src.GEFConstructor

h = 0.67
def PlotSensitivityCurves(ax : plt.Axes, names : list=[], cols : list=[], alpha : float=0.25):
    """
    Plot the sensitivity curves for current and planned gravitational wave experiments on a OmegaGW vs. frequency plot. Experiments with existing data
    are shown with filled-in sensivity curves.

    Parameters
    ----------
    ax : Axes-type
        the plot over which to overlay the sensitivity curves 
    names : list
        the names of GW experiments for which to plot the sensitivity curves. 
        Accepted names are 'LISA', 'EPTA', 'IPTA', 'HLV', 'BBO', 'HLVK', 'HLVO2', 'DECIGO', 'HL', 'NANOGrav', 'SKA', 'CE', 'ET', 'NANOGrav', 'PPTA'
        Empty lists, or lists containing invalid names are parsed to show all sensitivity curves
    cols : list
        a list of colours for which to plot the sensitivity curves. The ordering of the colors is the same as the ordering of experiments in names.
        If no colors are given (col=[]), the colors are generated at random.
    alpha : float
        the opacity with which to fill in the sensivity curves for experiments with existing data

    Returns
    -------
    ax : Axes-type
        the updated plot.
    """
    #the path to the sensitivity curve data
    path = os.path.abspath(inspect.getfile(src.GEFConstructor)).replace("GEFConstructor.py", "Tools/power-law-integrated_sensitivities/")
    arr = os.listdir(path)
    
    #Obtain List of experiments and running experiments
    exp = [a.replace("plis_","").replace(".dat", "") for a in arr ]
    RunningExp = ["IPTA", "NANOGrav", "PPTA", "EPTA", "HLVK", "HLV", "HLV02"]

    #Parse Input Names
    if names!=[]:
        err=False
        for name in names:
            if name not in exp:
                print(name,"is not a recognised experiment")
                err = True
        if err:
            print("Recognised experiments are given by", exp)
            print("Defaulting to showing all known sensitivity curves")
            names=exp
        else:
            arr = []
            for name in names:
                if name=="NANOGrav":
                    arr.append("NANOGrav/sensitivity_curves_NG15yr_fullPTA.txt")
                else:
                    arr.append("plis_" + name + ".dat")

    elif names==[]:
        names=exp

    #Parse Input Colors
    if cols==[]:
        colavail = list(mcolors.CSS4_COLORS.keys())
        rndint = random.sample(range(0, len(colavail)-1), len(names))
        cols=rndcols = [colavail[n] for n in rndint]

    #Create Dictionary linking experiment names to unique files and colors
    dic = {}
    for i, name in enumerate(names):
        dic[name] = {"file":arr[i],"col":cols[i],"running":name in RunningExp}


    for key in dic.keys():
        
        if key=="NANOGrav":
            tab = pd.read_table(path+dic[key]["file"], comment="#", sep=",").values.T
            f = tab[0,:]
            SCurve = h**2*tab[3,:]
        else:
            tab = pd.read_table(path+dic[key]["file"], comment="#").values.T
            f = 10**tab[0,:]
            SCurve = 10**tab[1,:]
        f = np.array([f[0]] + list(f) + [f[-1]])
        SCurve = np.array([1.] + list(SCurve) + [1.])
        
        if dic[key]["running"]:
            ax.fill_between(f, max(SCurve)*np.ones(f.shape), SCurve, color=dic[key]["col"], alpha=alpha)
        ax.plot(f, SCurve, color=dic[key]["col"])
    return ax