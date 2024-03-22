#Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
import scipy.integrate as integrate
import os


#def gNFW(r, gamma, r_s):
 #   return (r/r_s)**(-gamma) * (1 + r/r_s)**(gamma-3)


#Flux = integrate.quad()
    
def integrand(t, a):
    return t**(a-1) * np.exp(-t)

def gamma_inc(s, x):
    return integrate.quad(integrand, x,100000000, args=(s))[0]

print(gamma_inc(-0.94,10**(-6)))
