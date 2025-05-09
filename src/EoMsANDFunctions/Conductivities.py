import numpy as np

def ComputeSigmaCollinear(vals, pic):
        E0 = vals.E
        B0 = vals.B
        G0 = vals.G
        mu = (E0+B0)
        if mu<=0:
            return 0., 0., 1e-2*vals.kh
        else:
            mu = (mu/2)**(1/4)
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*vals.H0))))
            
            H = vals.H
            a = vals.a

            sigma = (41.*gmu**3/(72.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B0/E0))))
            sigmaE =  np.sqrt(B0) * (min(1., (1.- pic))*E0 + max(-pic, 0.)*B0) * sigma / (E0+B0)         
            sigmaB = -np.sign(G0) * np.sqrt(E0)*(min(1., (1.+ pic))*B0 + max(pic,0.)*E0)* sigma/(E0+B0)
            
            ks = gmu**(1/2)*E0**(1/4)*a
            
            return sigmaE, sigmaB, ks

def ComputeImprovedSigma(vals):
    E0 = vals.E
    B0 = vals.B
    G0 = vals.G
    Sigma = np.sqrt((E0 - B0)**2 + 4*G0**2)
    if Sigma<=0:
        return 0., 0., 1e-2*vals.kh
    else:
        mz = 91.2/(2.43536e18)
        mu = ((Sigma)/2)**(1/4)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*vals.H0))))

        Eprime = np.sqrt(E0 - B0 + Sigma)
        Bprime = np.sqrt(B0- E0 + Sigma)
        Sum = E0 + B0 + Sigma
        
        H = vals.H
        a = vals.a


        sigma = ( 41.*gmu**3/(72.*np.pi**2) / (np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime)))
        
        ks = gmu**(1/2)*Eprime**(1/2)*a

        return abs(G0)*Eprime*sigma, -G0*Bprime*sigma, ks