#Import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os


#def gNRW(s, rs, gamma, rc):
#    r = np.sqrt(s*s + rc*rc - 2*s*rc)
#    a = (r / rs)**(-gamma)
#    b = (1 + r / rs)**(-3+gamma)

#    return a * b 

#l = 20
#b = 20
#rs = 20
#gamma = 1.2
#rc = 8.5

#I = integrate.quad(gNRW, 1.0e-8 , 1.0e8, args=(rs, gamma, rc))
b_min = np.deg2rad(2)
b_max = np.deg2rad(20)
l_min = 0
l_max = b_max

 #I = integrate.dblquad(lambda l, b : 4*np.cos(l)*np.cos(b), 0, l_max)
def int(x, b):
    return np.cos(x)*b

i1 = integrate.quad(int, l_min , l_max, args=(1))[0]
i2 = integrate.quad(int, b_min , b_max, args=(1))[0]
print(4*i1*i2)