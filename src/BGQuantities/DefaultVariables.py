from src.BGQuantities.BGTypes import BGVal

#Space--time variables:
t=BGVal("t", -1, 0) #physical time
N=BGVal("N", 0, 0) #e-folds
a=BGVal("a", 0, 0) #scale factor
H=BGVal("H", 1, 0) #Hubble rate
spacetime = {t, N, a, H}