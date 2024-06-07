import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def PlotComp(ax1, ax2, N, Y, Nref, Yref, col="k:", label=None, interp="lin"):
    ax1.plot(N, Y, col, label=label)
    ax1.set_xlabel(r"$N_e$")
    if interp=="cubic":
        spl = CubicSpline(N, Y)
        ax2.plot(Nref, abs(spl(Nref)/Yref-1), col)
    else:
        spl = np.interp(Nref, N, Y)
        ax2.plot(Nref, abs(spl/Yref-1), col)
    ax2.set_yscale("log")
    ax2.set_ylabel("rel. err.")
    ax2.set_xlabel(r"$N_e$")
    
    return
        


def SaveMode(t, k, Ap, dAp, Am, dAm, af, name=None):
    logk = np.log10(k)
    N = list(np.log(af(t)))
    N = np.array([np.nan]+N)
    t = np.array([np.nan]+list(t))
    dic = {"t":t}
    dic = dict(dic, **{"N":N})
    for k in range(0, len(Ap[:,0])):
        dictmp = {"Ap_" + str(k) :np.array([logk[k]] + list(Ap[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"Am_" + str(k) :np.array([logk[k]] + list(dAp[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"dAp_" + str(k):np.array([logk[k]] + list(Am[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"dAm_" + str(k):np.array([logk[k]] + list(dAm[k,:]))}
        dic = dict(dic, **dictmp)
        
    if(name==None):
        filename = "Modes+Beta" + str(beta) + "+M6_16" +  + ".dat"
    else:
        filename = name

    DirName = os.getcwd()

    path = os.path.join(DirName, filename)

    output_df = pd.DataFrame(dic)  
    output_df.to_csv(path)
    
    return

def ReadModeFile(file):
    input_df = pd.read_table(file, sep=",")
    dataAp = input_df.values

    x = np.arange(3,dataAp.shape[1], 4)
    
    t = np.array(dataAp[1:,1])
    N = np.array(dataAp[1:,2])
    logk = np.array([(complex(dataAp[0,y])).real for y in x])
    Ap = np.array([[complex(dataAp[i+1,y]) for i in range(len(N))] for y in x])
    dAp = np.array([[complex(dataAp[i+1,y+1]) for i in range(len(N))] for y in x])
    Am = np.array([[complex(dataAp[i+1,y+2]) for i in range(len(N))] for y in x])
    dAm = np.array([[complex(dataAp[i+1,y+3]) for i in range(len(N))] for y in x])
    
    return t, N, logk, Ap, dAp, Am, dAm