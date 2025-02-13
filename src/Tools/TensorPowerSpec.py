import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapezoid

from src.GEFClassic.ModeByModeClassic import ReadMode

from ptarcade.models_utils import g_rho, g_rho_0, g_s, g_s_0, T_0, M_pl, gev_to_hz, omega_r, h


fcmb = 7.73e-17 #in Hz

class PowSpecT: 
    def __init__(x, G):
        if G.units: G.Unitless()
        
        a = G.vals["a"]
        H = G.vals["H"]
        
        Nend = G.EndOfInflation()[0]
        N = G.vals["N"]

        x.__omega = G.omega
        
        if max(N) < Nend:
            print("This GEF run has not run reached the end of inflation. The code will assume Nend = max(N). Proceed with caution!")
        maxN = min(max(N), Nend)
            
        x.maxk = CubicSpline(N, a*H)(maxN)
        x.mink = 1e4
        
        #Define Hidden variables

        #Define Useful quantities
        x.__beta = G.beta

        x.__t = G.vals["t"]
        x.__N = N
        x.__H = H
        x.__xi= G.vals["xi"]

        
        x.__af = CubicSpline(x.__t, a)
        x.__Hf = CubicSpline(x.__t, H)
        x.__HN = CubicSpline(x.__N, x.__H)
        x.__khN = CubicSpline(N, G.vals["kh"])

        x.__Nend = maxN
        print(Nend)
   
        return
    
    def _InitialKTN_(x, init, mode="t", pwr=5/2):
        t = x.__t
        #print("earlier?")
        logkH = lambda t: np.log(x.__af(t)*x.__Hf(t))
        if mode=="t":
            tstart = init
            logks = logkH(tstart)
            k = 10**(pwr)*np.exp(logks)
        elif mode=="k":
            k = init
            tstart = []
            for l in k:
                x0 = np.log(l) - pwr*np.log(10)
                f = lambda x: np.log(l) - logkH(x) - pwr*np.log(10)
                ttmp = fsolve(f, x0)[0]
                tstart.append(ttmp)
            tstart = np.array(tstart)
        elif mode=="N":
            tstart = CubicSpline(x.__N, t)(init)
            k = 10**pwr*np.exp(logkH(tstart))
        else:
            print("not a valid choice")
            return

        return k, tstart
        
    def _GetHomSol_(x, k, tstart, teval=[]):
        def EoM(Z, H, q, a):
            #The equation of motion we want to solve
            dZdt = np.zeros(Z.shape)

            #real
            dZdt[0] = (H*Z[0] + k/a*Z[1])
            dZdt[1] = ( -H*Z[1] - q/(k*a)*Z[0] )

            #imaginary
            dZdt[2] = (H*Z[2] + k/a*Z[3])
            dZdt[3] = ( -H*Z[3] - q/(k*a)*Z[2] )

            return dZdt

        if len(teval)==0:
            teval = x.__t

        tend = max(x.__t)
        lstart = 0
        while teval[lstart]<tstart:
            lstart+=1
        
        qf = lambda t: k**2
        
        ode = lambda t, y: EoM( y, x.__Hf(t), qf(t), x.__af(t) )
        
        Zini = np.array([1, -10**(-5/2), 0, -1])

        sol = solve_ivp(ode, [tstart, tend], Zini, t_eval=teval[lstart:], method="RK45", atol=1e-8, rtol=1e-8)
        if not(sol.success):
            print("Z: something went wrong")
            
        deta = lambda t, y: 1/x.__af(t)
        
        soleta = solve_ivp(deta, [min(x.__t), tend], np.array([-1]), t_eval=teval)

        if not(soleta.success):
            print("eta: something went wrong")
        
        eta = (soleta.y[0,:] - soleta.y[0,lstart]).astype(complex)

        #ensure that Zs is the same length for every mode k
        phik = np.array( list( np.exp(-1j*k*eta[:lstart]) )
                      + list( sol.y[0,:] + 1j*sol.y[2,:] ) )/x.__af(teval)

        dphik = np.array( list( (-1j)*np.exp(-1j*k*eta[:lstart]) )
                      + list( sol.y[1,:] + 1j*sol.y[3,:] ) )/x.__af(teval)

        return phik, dphik

    def _GreenFunc_(x, k, phik, ind, tstart, teval=[]):
        def ode(A, H, q, a):
            dAdt = np.zeros(A.shape)
            dAdt[0] =(2*H*A[0] + k/a*A[1])
            dAdt[1] = -q/(k*a)*A[0]
            return dAdt

        qf = lambda t: k**2

        if len(teval)==0:
            teval = x.__t
            
        lstart = 0
        while teval[lstart]<tstart:
            lstart+=1

        Aini = np.array([0, 1])

        Aode = lambda t, y: -ode(y, x.__Hf(-t), qf(-t), x.__af(-t))
        solA = solve_ivp(Aode, [-teval[ind], -tstart], Aini, t_eval=-teval[lstart:ind+1][::-1],
                         method="RK45", atol=1e-8, rtol=1e-8)

        GreenN = np.zeros(teval.shape)
        GreenN[lstart:ind+1] = solA.y[0,:][::-1]
        GreenN[:lstart] = ( (phik[ind].conjugate()*phik).imag*x.__af(teval)**2 )[:lstart]

        return GreenN

    def _VacuumPowSpec_(x, k, phik):
        return 2*(k*x.__omega)**2/( np.pi**2 ) * abs(phik)**2

    def _InducedTensorPowerSpecLog_(x, k, lgrav, ind, Ngrid, HN, GreenN, kgrid, l1, A1, dA1, l2, A2, dA2):
        
        cutUV = x.__khN(Ngrid[ind])/k
        cutIR = min(kgrid)/k

        logAs = np.linspace(np.log(max(0.5, cutIR)), np.log(cutUV), 100)

        Afuncx = CubicSpline(np.log(kgrid), A1)
        dAfuncx = CubicSpline(np.log(kgrid), dA1)
        
        Afuncy = CubicSpline(np.log(kgrid), A2)
        dAfuncy = CubicSpline(np.log(kgrid), dA2)

        IntOuter = []
        for logA in logAs:
            A = np.exp(logA)
            Blow = (cutIR - A)
            Bhigh = (A - cutIR)
            if Bhigh>0.5:
                Blow = (A - cutUV)
                Bhigh = (cutUV - A)
                if Bhigh>0.5:
                    Blow = -0.5
                    Bhigh = 0.5
            Bs = np.linspace(Blow, Bhigh, 100)[1:-1]

            IntInner = np.zeros(Bs.shape)

            for j, B in enumerate(Bs):
                Ax = Afuncx(np.log(k*(A+B)))
                dAx = dAfuncx(np.log(k*(A+B)))
                Ay = Afuncy(np.log(k*(A-B)))
                dAy = dAfuncy(np.log(k*(A-B)))

                mom =  abs( l1*l2 + 2*lgrav*( (l1+l2)*A + (l1-l2)*B ) + 4*(A**2 - B**2) + 8*lgrav*A*B*( (l1-l2)*A - (l1+l2)*B ) - 16*l1*l2*A**2*B**2 )
                z = max(A+B,A-B)
                mask = np.where(z<x.__khN(Ngrid)/k, 1, 0)

                val = (dAx*dAy + l1*l2*Ax*Ay)*mask*k/(np.exp(3*Ngrid)*HN)
                timeintegrand = GreenN*val*mom
            
                timeintre = trapezoid(timeintegrand[:ind].real, Ngrid[:ind])
                
                timeintim = trapezoid(timeintegrand[:ind].imag, Ngrid[:ind])
                
                IntInner[j] = (timeintre**2 + timeintim**2)*A

  
            IntOuter.append(trapezoid(IntInner, Bs))
        IntOuter = np.array(IntOuter)

        int = trapezoid(IntOuter, logAs)

        return int / (16**2*np.pi**4)*(k*x.__omega)**4

    
    def ComputePowSpec(x, vals, Nfin=None, mode="k", ModePath=None, FastGW=True):

        ks, tstarts = x._InitialKTN_(vals, mode=mode)

        if ModePath==None:
            ModePath = f"../Modes/Modes+Beta{x.__beta}+M6_16.dat"
        
        tgrid, Ngrid, kgrid, Ap, dAp, Am, dAm = ReadMode(ModePath)
        Ngrid = np.array(list(Ngrid))

        GaugeModes = {"+":(Ap, dAp), "-":(Am, dAm)}

        if Nfin==None:
            Nfin = x.__Nend
        
        inds = np.where(Ngrid < Nfin)[0]
        indend = inds[-1]

        PT = {"tot":[], "vac":[], "ind+,++":[], "ind+,+-":[], "ind+,--":[], "ind-,++":[], "ind-,+-":[], "ind-,--":[]}

        GWpols = [("+", 1), ("-",-1)]

        gaugepols=[(("+",1),("+",1)),
                    (("+",1),("-",-1)),
                        (("-",-1),("-",-1))]
        
        HN = x.__HN(Ngrid)
        sign = np.sign(x.__xi[0])

        for i, k in enumerate(ks):
            tstart = tstarts[i]

            if k > 5*(x.__af(tgrid[indend])*x.__Hf(tgrid[indend])):
                for key in PT.keys():
                    PT[key].append(0)
            else:
                f, _ = x._GetHomSol_(k, tstart, tgrid)
                Green = x._GreenFunc_(k, f, indend, tstart, tgrid)
                
                PT["vac"].append(x._VacuumPowSpec_(k, f[indend]))

                for lgrav in GWpols:
                    for mu in gaugepols:
                        GWpol = lgrav[1]

                        Ax = GaugeModes[mu[0][0]][0]
                        dAx = GaugeModes[mu[0][0]][1]
                        lx = mu[0][1]
                        Ay = GaugeModes[mu[1][0]][0]
                        dAy = GaugeModes[mu[1][0]][1]
                        ly = mu[1][1]

                        if FastGW and (lx != sign or ly != sign):
                            PT[f"ind{lgrav[0]},{mu[0][0]}{mu[1][0]}"].append(0.)
                        else:
                            PT[f"ind{lgrav[0]},{mu[0][0]}{mu[1][0]}"].append(
                                x._InducedTensorPowerSpecLog_(k, GWpol, indend, Ngrid, HN, Green, kgrid, lx, Ax, dAx, ly, Ay, dAy) )

        PT["tot"] = np.zeros(ks.shape)
        for key in PT.keys():
            PT[key] = np.array(PT[key])
            if ("+" in key) or ("-" in key):
                PT["tot"] += 0.5*PT[key]
            elif key=="vac":
                PT["tot"] += PT[key]

        return PT
    
    def ktofreq(x, k):
        Hend = x.__HN(x.__Nend)

        Trh = np.sqrt(3*Hend*x.__omega/np.pi)*(10/106.75)**(1/4)*M_pl
        #Trh = Trh*(106.75/g_rho(Trh))**(1/4)

        return k*x.__omega*M_pl*gev_to_hz/(2*np.pi*np.exp(x.__Nend)) * T_0/Trh * (g_s(Trh)/g_s_0)**(-1/3)

    def PTtoOmega(x, PT, k):
        f = x.ktofreq(k)
        OmegaGW = h**2*omega_r/24  * PT * (g_rho(f, True)/g_rho_0) * (g_s_0/g_s(f, True))**(4/3)
        return OmegaGW
    
    def PTAnalytical(x):
        H = x.__omega*x.__H
        xi = abs(x.__xi)
        pre = (H/np.pi)**2 # * (x.__H)**(x.__nT)
        if np.sign(x.__xi[0]) > 0:                
            indP = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
        else:
            indP = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
        
        #Factors of two to match my convention
        PTanalytic = {"tot":(2*pre + indP + indM), "vac":2*pre, "ind+":2*indP, "ind-":2*indM}
        return PTanalytic

