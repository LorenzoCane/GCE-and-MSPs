#######################################################################################
# In this file are reported the definition of mot common luminosity functions
# Import this file to use them multiple times
# Created by Lorenzo Cane (Unito, LAPTh) on 14/04/2024
# This code is part of my thesis project
#######################################################################################


import numpy as np
#import scipy.integrate as integrate


#LUMINOSITY FUNCTIONS DEFINITIONS

# array power law func(high flux exp c-o, low flux step c-o)
def power_law (l, alpha ,l_max, norm):  #alpha: slope, l_min = step c-o, l_max: exp c-o               
    num = l**(-alpha) * np.exp(-l/l_max)

    return norm * num  
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
            frac = frac**(-n1)
        else:
            frac = frac**(-n2)
        bpl.append(norm *frac)
    
    return np.array(bpl)
#-------------------------------------------------------

def broken_pl(x, norm, x_b, n1, n2):                  
    #broken power law function  (be carefull about exponential renorm)  
    #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index
    frac = x / x_b
    if frac < 1:
        frac = frac**(-n1)
    else:
        frac = frac**(-n2)
        
    return (norm *frac)
#-------------------------------------------------------

def gNRW2(s, l , b , rs, gamma, rc):    
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
#FLUX FUNCTIONS DEF (all sources are considered in the Galactic Center)

def l_log(l, l_0 , sigma):
    num = np.log10(np.e)
    den = sigma * l * (2*np.pi)**0.5 
    exp = np.exp(-1.0 * (np.log10(l/l_0))**2 / (2 * sigma * sigma))

    return l*  num * exp / den 
#-------------------------------------------------------

def l_bpl(x, norm, x_b, n1, n2):  
    return broken_pl(x, norm, x_b, n1, n2) * x