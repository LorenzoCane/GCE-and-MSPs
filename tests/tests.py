#Import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os

test_numb = 2
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
inf_approx = 1.0e70

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

def func_log(y, func, arg):
    return np.log(10) * 10**y * func(10**y, *arg)

def log_scale_int(func, x_min, x_max, inf_approx, arg, abs_err, rel_err, div_numb):
    y_lim = log_range_conv(x_min, x_max, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]

    res = integrate.quad(func_log, y1, y2,  args=(func, arg), epsabs = abs_err, epsrel = rel_err, limit=div_numb)

    return res


def accum_func(a, func, arg, min):
    acc = integrate.quad(func, min, a, arg, epsabs=0.0, epsrel=1.0e-6)[0]
    return acc



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


    def mygamma_inc(s,x,inf_approx, abs_err, rel_err, div_numb):
        def integr(t, s):
            return t**(s-1) * np.exp(-t)
        arg = (s,)
        I = log_scale_int(integr, x, np.infty, inf_approx, arg, abs_err, rel_err, div_numb)
        #I = integrate.quad(integr, x , 1.0e70, args=(s), epsabs=abs_err, epsrel=rel_err, limit=div_numb)
        return I

    norm = mygamma_inc(-0.94, 1.0e-6, inf_approx, abs_err, rel_err, div_numb)[0]
    print(norm* 1.0e35**(1-1.94))

#********************************************************************************************
elif test_numb == 3 :
    #LOG-SCALE INTEGRATION
  
    x1 = 0
    x2 = 10
    y_lim = log_range_conv(x1, x2, inf_approx)
    y1 = y_lim[0]
    y2 = y_lim[1]


    def func_x(x):
        return x
    arg = ()
    integr =log_scale_int(func_x, x1, x2, inf_approx, arg, abs_err, rel_err, div_numb)

    print(y1, "  ", y2)
    print(integr)

#********************************************************************************************
elif test_numb == 4 :

    def normal_distr(mu, sigma):
        rng = np.random.default_rng()
        u1 = rng.random()
        u2 = rng.random()

        y = sigma * (np.sin(2.0*np.pi*u1) * (-2.0 * np.log(u2))**0.5) + mu

        return y
    
    mu = 0.0
    sigma = 1.0
    n = 10000
    n_bins = round(n**0.5)
    counter = 0
    temp = []
    while counter < n :
        temp.append(normal_distr(mu, sigma))
        counter +=  1

    x = np.array(temp)   

    fig, ax = plt.subplots()

    count, bins, ignored = ax.hist(x, n_bins, density=True)
    plt.savefig(os.path.join("test.png"))
#********************************************************************************************
elif test_numb == 5 :
    x_min = 0.0
    x_max = 10.0


    def bounded_norm_distr(mu, sigma, x_min, x_max):
        rng = np.random.default_rng()


        y = x_min - 1.0
        while y < x_min or y > x_max :
            u1 = rng.random()
            u2 = rng.random()
            y = sigma * (np.sin(2.0*np.pi*u1) * (-2.0 * np.log(u2))**0.5) + mu
            print(y)
    
        return y
    
    y = bounded_norm_distr(1.0, 0.5, x_min, x_max)

#********************************************************************************************
elif test_numb == 6 : 

    def Newton_root_finder(func, func_prime, arg, a, c, tol, n_max, print_stat = False ):
        b = a + abs(c-a)
        fa = func(a, *arg)
        fb = func(b, *arg)
        fb_prime = func_prime(b, *arg)
        dx = b - a
        k = 0
        
        if fa == 0.0 : res = a
        elif fb == 0.0 : res = b

        else:
            while (abs(dx) >= tol):
                k += 1

                if k > n_max:
                    #print(res)
                    raise ValueError("Too many iteration in bisection root finder")
            
                dx = fb / fb_prime
                a = b
                b -= dx

                fb = func(b, *arg)
                fb_prime = func_prime(b,  *arg)
        
            res = b
            fres = func(res, *arg)
        if print_stat : 
            print("# of iteractions: ", k, "\nRoot found: ", res, "\nFunction evaluated at root value: ", fres)

        return res
            
    
    x_min = 0.1
    x_max = 3
    test_func = 3
    args = ()
    value = 0
    tol = 1.0e-6
    n_max = 50

    if test_func == 0:
        def myfunc(x):
            return x +1
        def myfuncprime(x):
            return 1.0
    elif test_func == 1:
        def myfunc(x):
            return x*x -1.0
        def myfuncprime(x):
            return 2.0 *x 
    elif test_func ==  2:
        def myfunc(x):
            return np.exp(x-1.0) - 2.0
        def myfuncprime(x):
            return np.exp(x-1.0)
    elif test_func == 3:
        def myfunc(x):
            return np.log10(x**1.5) + 2.0 * x
        def myfuncprime(x):
            return 0.651442 / x + 2.0
        

    res = Newton_root_finder(myfunc, myfuncprime, args, x_min, x_max, tol, n_max, True)

    print(res)

#********************************************************************************************
elif test_numb == 7:

    def bisection(func, arg, a, b, tol, nmax, print_stat = False):
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
                #print(res)
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
    

    x_min = -4.0
    x_max = 5.0
    test_func = 3
    args = ()
    value = 0
    tol = 1.0e-6
    n_max = 50

    if test_func == 0:
        def myfunc(x):
            return x +1.0
    elif test_func == 1:
        def myfunc(x):
            return x*x -1.0
    elif test_func ==  2:
        def myfunc(x):
            return np.exp(x-1.0) - 2.0
    elif test_func == 3:
        def myfunc(x):
            return np.log10(x**1.5 - x*x) + 2.0 * x
        

    res = bisection(myfunc, args, x_min, x_max, tol, n_max, True)

    print(res)


           #def chisq (x, y , func, arg=()):
            #chi = 0.0
            #for i in x:
            #    expect = func(x[i], *arg)
            #    chi_i = ((y[i] - expect)**2) / expect
            #    chi += chi_i
            #return chi 