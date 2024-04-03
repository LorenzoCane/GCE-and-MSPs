#Import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os

def cmtokpc(x):                  
    #convertion of cm to kpc 
    return x*3.2407792896664e-22 


def gNRW2(s, l , b , rs, gamma, rc):    
    #general Navarro-Frenk-White squared
    #s: Earth point distance; l, b:  long and lat.; rs:scale radius; gamma: slope; rc:Earth-GC dist.
    r = np.sqrt(s*s + rc*rc - 2*s*rc*np.cos(l)*np.cos(b))
    a = (r / rs)**(-2*gamma)
    b = (1 + r / rs)**(2*(-3+gamma))

    return a * b 

def denom(s, l , b , rs, gamma, rc):
    # return s^2 * gNFW function 
    #ALERT: THE RESULT HAS [s]^2 AS UNIT
    return (s**2)*gNRW2(s, l , b , rs, gamma, rc)

b_min = np.deg2rad(2)
b_max = np.deg2rad(20)
l_min = 0
l_max = b_max
rs = 20 #kpc
gamma = 1.2 #kpc
rc = 8.5 #kpc

I = integrate.nquad(gNRW2, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, gamma, rc))
I2 = integrate.nquad(denom, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, gamma, rc))

#print(I[0] )


#def int(x, b):
#    return np.cos(x)*b

i1 = integrate.quad(np.cos, l_min , l_max)[0]
i2 = integrate.quad(np.cos, b_min , b_max)[0]
stupid =4*i1*i2 
#print(stupid)
#GeVtoerg = 0.00160218


#def broken_pl(x, norm, x_b, n1, n2):             #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index               
#   bpl = []

    
#    if x < x_b:
#       frac = (x/x_b)**(2-n1)
#    else:
#        frac = (x/x_b)**(2-n2)
#        bpl.append(norm *frac)
#    
#    return norm * frac

#I = integrate.quad(broken_pl, 0.1, 10, args=(1.0e-6, 2.06, 1.42, 2.63))
#print(I[0] * GeVtoerg )

def integr(t, s):
    return t**(s-1) * np.exp(-t)

def mygamma(s,x):
    I = integrate.quad(integr, x , 1.0e50, args=(s))
    return I

norm = mygamma(-0.94, 1.0e-6)[0]
print(norm * (1.0e35)**(1-1.94))