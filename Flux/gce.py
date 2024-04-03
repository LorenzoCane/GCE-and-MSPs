
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import gammaincc,gamma, exp1
import os


#LUMINOSITY FUNCTIONS DEFINITIONS

# array power law func(high flux exp c-o, low flux step c-o)
def power_law (l, alpha , l_min , l_max, a):   #alpha: slope, l_min = step c-o, l_max: exp c-o               
    if (a):
        den =5.84e-28
    else:
        den = 6.31e-15
    num = l**(-alpha) * np.exp(-l/l_max)

    return num / den 
#-----------------------------------------------------------------------------

# array log normal func
def log_norm (l, l_0 , sigma):                          
    num = np.log10(np.e)
    den = sigma * l * (2*np.pi)**0.5 
    exp = np.exp(-1.0 * (np.log10(l/l_0))**2 / (2 * sigma * sigma))

    return num * exp / den 
#----------------------------------------------------------------------------
def broken_pl_arr(x, norm, x_b, n1, n2):               
    #array broken power law function  (be carefull about exponential renorm)  
    #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index 
    bpl = []

    for a in range(0, len(x)):
        frac = x[a] / x_b
        if frac < 1:
            frac = frac**(2-n1)
        else:
            frac = frac**(2-n2)
        bpl.append(norm *frac)
    
    return np.array(bpl)
#-------------------------------------------------------

def broken_pl(x, norm, x_b, n1, n2):                  
    #broken power law function  (be carefull about exponential renorm)  
    #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index
    frac = x / x_b
    if frac < 1:
        frac = frac**(2-n1)
    else:
        frac = frac**(2-n2)
        
    return (norm *frac)
#-------------------------------------------------------

def GeVtoerg(x):
    #convertion from GeV to erg
    return x * 0.00160218
#-------------------------------------------------------

def cmtokpc(x):                  
    #convertion from cm to kpc 
    return x*3.2407792896664e-22 
#-------------------------------------------------------

def kpctocm(x):
    #convertion from kpc to cm
    return x * 3.0857e21
#-------------------------------------------------------

def gNRW2(s, l, b, rs, gamma, rc):    
    #general Navarro-Frenk-White squared
    #s: Earth point distance; l, b:  long and lat.; rs:scale radius; gamma: slope; rc:Earth-GC dist.
    r = np.sqrt(s*s + rc*rc - 2*s*rc*np.cos(l)*np.cos(b))
    a = (r / rs)**(-2*gamma)
    b = (1 + r / rs)**(2*(-3+gamma))

    return a * b 
#-------------------------------------------------------

def sgNRW(s, l , b , rs, gamma, rc):
    # return s^2 * gNFW function 
    #ALERT: THE RESULT HAS [s]^2 AS UNIT
    return (s**2)*gNRW2(s, l , b , rs, gamma, rc)
#****************************************************************************