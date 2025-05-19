import numpy as np

def ComputeSigmaCollinear(a, H, E, B, G, pic, H0):
        mu = (E+B)
        if mu<=0:
            return 0., 0., 0.
        else:
            mu = (mu/2)**(1/4)
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*H0))))
            
            sigma = (41.*gmu**3/(72.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B/E))))
            sigmaE =  np.sqrt(B) * (min(1., (1.- pic))*E + max(-pic, 0.)*B) * sigma / (E+B)         
            sigmaB = -np.sign(G) * np.sqrt(E)*(min(1., (1.+ pic))*B + max(pic,0.)*E)* sigma/(E+B)
            
            ks = gmu**(1/2)*E**(1/4)*a
            
            return sigmaE, sigmaB, ks

def ComputeImprovedSigma(a, H, E, B, G, H0):
    Sigma = np.sqrt((E - B)**2 + 4*G**2)
    if Sigma<=0:
        return 0., 0., 0.
    else:
        mz = 91.2/(2.43536e18)
        mu = ((Sigma)/2)**(1/4)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*H0))))

        Eprime = np.sqrt(E - B + Sigma)
        Bprime = np.sqrt(B- E + Sigma)
        Sum = E + B + Sigma
        
        sigma = ( 41.*gmu**3/(72.*np.pi**2) / (np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime)))
        
        ks = gmu**(1/2)*Eprime**(1/2)*a

        return abs(G)*Eprime*sigma, -G*Bprime*sigma, ks