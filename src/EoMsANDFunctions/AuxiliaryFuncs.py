import numpy as np

def Heaviside(x, eps):
    return 1/(1+np.exp(-x/eps))