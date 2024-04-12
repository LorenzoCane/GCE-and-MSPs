#Import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os

test_numb = 3

#------------------------------------------
#some usefull constants/bounds

b_min = np.deg2rad(2)
b_max = np.deg2rad(20)
l_min = 0
l_max = b_max
rs = 20 #kpc
gamma = 1.2 #kpc
rc = 8.5 #kpc

#integration conditions
abs_err = 0.0
rel_err = 1.0e-8
div_numb = 100

#--------------------------------------------
#Functions

def cmtokpc(x):                  
    #convertion of cm to kpc 
    return x*3.2407792896664e-22 

def GeVtoerg(x):
    #convertion from GeV to erg
    return x * 0.00160218

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

def broken_pl(x, norm, x_b, n1, n2):             #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index               
    bpl = []    
    if x < x_b:
        frac = (x/x_b)**(2-n1)
    else:
        frac = (x/x_b)**(2-n2)
    bpl.append(norm *frac)
    
    return norm * frac

def log_range_conv(x1, x2, inf_approx):
    #transform the limits of an integral into the log-scale int. limits
    # x1 : lower limit ; x2 : upper limit
    # it returns the new limits in an array
    #!!! FOR +INF LIMIT AN APPROX. inf_approx MUST BE SELECTED
    if x1 != np.infty and x1 != 0:  y1 = np.log10(x1)
    elif x1 == 0 : y1 = - np.infty
    elif x1 == np.infty : y1 = np.log10(inf_approx)
    
    if x2 != np.infty and x2 != 0:  y2 = np.log10(x2)
    elif x2 == 0 : y2 = - np.infty
    elif x2 == np.infty : y2 = np.log10(inf_approx)

    return [y1, y2]

def func_log(y, func):
    return np.log(10) * 10**y * func(10**y)

def log_scale_int(func, x_min, x_max, inf_approx, abs_err, rel_err, div_numb):
    y_lim = log_range_conv(x_min, x_max, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]

    res = integrate.quad(func_log, y1, y2,  args=(func), epsabs = abs_err, epsrel = rel_err, limit=div_numb)

    return res






#********************************************************************************************
if test_numb == 1:
    #SOLID ANGLE INTEGRATION

    I = integrate.nquad(gNRW2, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, gamma, rc))
    I2 = integrate.nquad(denom, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, gamma, rc))

    print(I[0] )


    def int(x, b):
        return np.cos(x)*b

    i1 = integrate.quad(np.cos, l_min , l_max)[0]
    i2 = integrate.quad(np.cos, b_min , b_max)[0]
    stupid =4*i1*i2 
    print(stupid)

#********************************************************************************************
elif test_numb == 2 :
    #GAMMA INCOMPLETE IMPLEMENTATION
    def integr(t, s):
        return t**(s-1) * np.exp(-t)

    def mygamma(s,x):
        I = integrate.quad(integr, x , 1.0e50, args=(s))
        return I

    norm = mygamma(-0.94, 1.0e-6)[0]
    print(norm * (1.0e35)**(1-1.94))

#********************************************************************************************
elif test_numb == 3 :
    #LOG-SCALE INTEGRATION
    inf_approx = 1.0e50
    x1 = 0
    x2 = 10
    y_lim = log_range_conv(x1, x2, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]


    def func_x(x):
        return x
    
    integr =log_scale_int(func_x, x1, x2, inf_approx, abs_err, rel_err, div_numb)

    print(y1, "  ", y2)
    print(integr)