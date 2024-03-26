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
stupid =4*i1*i2 

GeVtoerg = 0.00160218


def broken_pl(x, norm, x_b, n1, n2):             #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index               
    bpl = []

    
    if x < x_b:
        frac = (x/x_b)**(2-n1)
    else:
        frac = (x/x_b)**(2-n2)
        bpl.append(norm *frac)
    
    return norm * frac

I = integrate.quad(broken_pl, 0.1, 10, args=(1.0e-6, 2.06, 1.42, 2.63))
print(I[0] * GeVtoerg )