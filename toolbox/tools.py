#COLLECTION OF USEFULL FUNCTION DEVELOPED WORKING ON MY MASTER THESIS

import numpy as np
import scipy.integrate as integrate
from scipy.special import erf
#from scipy.special import gammaincc,gamma, exp1
#import os

#*************************************************************
# Root finders

def bisection(func, arg, a, b, tol= 1.0e8, nmax=1.0e4, print_stat = False):
    #bisection method root finder.
    #Search a root (in the interval [a, b])  of a function func with arguments arg. 
    #Tolerance and number of maximum interations are pre-selected.
    #print_stat: if true print the # of iter, root value computed and function evaluated in that point
    dx = abs(b-a)
    res = a + 0.5 * dx
    k = 0
    fa = func(a, *arg)
    fb = func(b, *arg)
    fres = func(res, *arg)

    if fa == 0.0: res = a
    if fb == 0.0: res = b


    while (abs(fres) >= tol):
        if k>nmax: 
            print(res)
            raise ValueError("Too many iteration in bisection root finder")
            
        res = a + 0.5 * dx
        fres = func(res, *arg)

        if (fa * fres) < 0.0:
            b = res
            fb = fres
        else :
            a = res
            fa = fres
    
        k += 1
        dx = abs(b-a)
        
    if print_stat : 
        print("# of iteractions: ", k, "\nRoot found: ", res, "\nFunction evaluated at root value: ", fres)

    return res
#-------------------------------------------------------
def newton_root_finder(func, func_prime, arg, arg_prime, a, c, tol=1.0e8, n_max=1.0e4, print_stat = False ):
    #Newtonmethod root finder.
    #Search a root (in the interval [a, b])  of a function func with arguments arg.
    #!PRIME DERIVATIVE (func_prime) AND ITS ARGUMENTS (arg_prime) MUST BE PROVIDDED
    #Tolerance and number of maximum interations are pre-selected.
    #print_stat: if true print the # of iter, root value computed and function evaluated in that point
   
    b = a + abs(c-a)
    fa = func(a, *arg)
    fb = func(b, *arg)
    fb_prime = func_prime(b, *arg_prime)
    dx = b - a
    k = 0
        
    if fa == 0.0 : res = a
    elif fb == 0.0 : res = b

    else:
        while (abs(fb) >= tol):
            k += 1

            if k > n_max:
                #print(res)
                raise ValueError("Too many iteration in bisection root finder")
            
            dx = fb / fb_prime
            a = b
            b -= dx

            fb = func(b, *arg)
            fb_prime = func_prime(b,  *arg_prime)
        
        res = b
        fres = func(res, *arg)

        if print_stat : 
            print("# of iteractions: ", k, "\nRoot found: ", res, "\nFunction evaluated at root value: ", fres)

        return res
    
#*************************************************************
#Function modifications

def func_norm(x, func, x_min, x_max, normal=1.0, arg=(), abs_err =0.0, rel_err=1.0e8):
    #takes a function "func" with arguments "arg" and normalize it to "normal" on the range (x_min; x_max)
    # options for integration procedure are pre-seleceed but can be modified
    i =  integrate.quad(func,x_min, x_max, arg, epsabs=abs_err, epsrel=rel_err)[0]
    return func(x, *arg) / i * normal
#-------------------------------------------------------
   
def accum_func(a, func, min, arg = ()):
    #evaluate the accumulation function of a function func from min to a, Use arg for add argument of the function
    acc = integrate.quad(func, min, a, arg, epsabs=0.0, epsrel=1.0e-12)[0]
    return acc
#-------------------------------------------------------
  
def func_shifter(x, func, value, args= ()):
    #Take a function func (with arguments args) and shifted it by value
    shifted = func(x, *args) - value
    return shifted

#*************************************************************
#Conversion between units

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

#*************************************************************
#Log-scale conversion and integration

def log_range_conv(x1, x2, inf_approx = 1.0e50):
    #transform the limits of an integral into the log-scale int. limits
    # x1 : lower limit ; x2 : upper limit
    # it returns the new limits in an array
    #infinity approx is selected but can be changed
    if x1 != np.infty and x1 != 0:  y1 = np.log10(x1)
    elif x1 == 0 : y1 = - np.infty
    elif x1 == np.infty : y1 = np.log10(inf_approx)
    
    if x2 != np.infty and x2 != 0:  y2 = np.log10(x2)
    elif x2 == 0 : y2 = - np.infty
    elif x2 == np.infty : y2 = np.log10(inf_approx)

    return [y1, y2]
#-------------------------------------------------------

def integrand_log(y, func, arg=()):
    #tranf a integrand into its log scaled form
    #y: variable of the func; func: func of which tha log scaling is desired; arg: arguments of func
    return np.log(10) * 10**y * func(10**y, *arg)
#-------------------------------------------------------

def log_scale_int(func, x_min, x_max,arg, inf_approx=1.0e50, abs_err = 0.0, rel_err = 1.0e-8, div_numb = 50):
    #perform the log-scaled integral of a funcion func 
    #x1, x2: original integrational limits; inf_approx: infinit approx; arg: argument of func; other parameters used as in quad
    # options for integration procedure are pre-selected but can be modified
    y_lim = log_range_conv(x_min, x_max, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]

    res = integrate.quad(integrand_log, y1, y2,  args=(func, arg), epsabs = abs_err, epsrel = rel_err, limit=div_numb)

    return res

#*************************************************************
#Mathematical functions

def mygamma_inc(s,x, inf_approx=1.0e50, abs_err=0.0, rel_err=1.0e-8, div_numb=50):
    #gamma incomplete calculation of s with lower limit x
    # options for integration procedure are pre-selected but can be modified
    def integr(t, s):
        return t**(s-1) * np.exp(-t)
    arg = (s,)
    I = log_scale_int(integr, x, np.infty, arg, inf_approx, abs_err, rel_err, div_numb)
    #I = integrate.quad(integr, x , 1.0e70, args=(s), epsabs=abs_err, epsrel=rel_err, limit=div_numb)
    return I[0]
#-------------------------------------------------------

def gaussian(x , x0, sigma, x_min, x_max):
    #Gauss distribution function with  peak at x0 and width sigma. Normalized on the range (x_min; x_max)
    norm1 = 1.0 / sigma / (2.0 * np.pi)**0.5
    norm2 = 1.0 / (erf((x_max-x0)/sigma/2.0**0.5) - erf((x_min-x0)/sigma/2.0**0.5))
    return np.exp(-(x - x0)*(x - x0)/sigma/sigma/2.0) * norm1 * norm2
#*************************************************************

#Random distribution algorithms

def normal_distr(mu=0.0, sigma=1.0):
    #generate a number according to the normal distribution density
    # if not specifid mu and sigma are pre-selected as 0.0 and 1.0
    rng = np.random.default_rng()
    u1 = rng.random()
    u2 = rng.random()

    y = sigma * (np.sin(2.0*np.pi*u1) * (-2.0 * np.log(u2))**0.5) + mu

    return y
#-------------------------------------------------------

def bounded_norm_distr(mu, sigma, x_min, x_max):
    rng = np.random.default_rng()
    y = x_min - 1.0

    while y < x_min or y > x_max :
        u1 = rng.random()
        u2 = rng.random()
        y = sigma * (np.sin(2.0*np.pi*u1) * (-2.0 * np.log(u2))**0.5) + mu

    return y


