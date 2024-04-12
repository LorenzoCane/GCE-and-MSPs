#COLLECTION OF USEFULL FUNCTION DEVELOPED WORKING ON MY MASTER THESIS

import numpy as np
import scipy.integrate as integrate
#from scipy.special import gammaincc,gamma, exp1
#import os


def GeVtoerg(x):
    #convertion from GeV to erg
    return x * 0.00160218
#-------------------------------------------------------

def cmtokpc(x):                  
    #convertion of cm to kpc 
    return x*3.2407792896664e-22 
#-------------------------------------------------------

def kpctocm(x):
    #convertion from kpc to cm
    return x * 3.0857e21
#-------------------------------------------------------

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
#-------------------------------------------------------

def integrand_log(y, func, arg):
    #tranf a integrand into its log scaled form
    #y: variable of the func; func: func of which tha log scaling is desired; arg: arguments of func
    return np.log(10) * 10**y * func(10**y, *arg)
#-------------------------------------------------------

def log_scale_int(func, x_min, x_max, inf_approx, arg, abs_err, rel_err, div_numb):
    #perform the log-scaled integral of a funcion func 
    #x1, x2: original integrational limits; inf_approx: infinit approx; arg: argument of func; other parameters used as in quad
    y_lim = log_range_conv(x_min, x_max, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]

    res = integrate.quad(integrand_log, y1, y2,  args=(func, arg), epsabs = abs_err, epsrel = rel_err, limit=div_numb)

    return res
#-------------------------------------------------------

def mygamma_inc(s,x, inf_approx, abs_err, rel_err, div_numb):
    #gamma incomplete calculation of s with lower limit x
    #inf approx: infinity approx desired; other parameters used as in quad
    def integr(t, s):
        return t**(s-1) * np.exp(-t)
    arg = (s,)
    I = log_scale_int(integr, x, np.infty, inf_approx, arg, abs_err, rel_err, div_numb)
    #I = integrate.quad(integr, x , 1.0e70, args=(s), epsabs=abs_err, epsrel=rel_err, limit=div_numb)
    return I[0]