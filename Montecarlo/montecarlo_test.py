#Exercise from PDG reviews

import time 
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import os
import sys
sys.path.insert(0, '/home/lorenzo/GCE-and-MSPs/toolbox')
from tools import bounded_norm_distr, bisection, newton_root_finder, log_scale_int, accum_func, func_shifter, func_norm, gaussian
start_time = time.monotonic()


exercise = 9      #to execute only one exercise at time

#**************************************************************
#plotting config
marker_st =  'o'        
marker_color = 'orange'
func_color = 'black'

#-----------------------------------------------
#integral config
abs_err = 0.0
rel_err = 1.0e-6
inf_approx = 1.0e40


#**************************************************************
#42.4.1 Exponential decay
if exercise == 1:
    npoints = 1000000

    x = np.linspace(0,10,10000)

    tau = 5 
    a = 0
    b = 10
    alpha = np.exp(-a/tau)
    beta = np.exp(-b/tau)

    rng = np.random.default_rng()

    u = rng.random(npoints)

    def decay(x, tau):
        return np.exp(-x/tau)/tau

    #count, bins, ignored = plt.hist(u, 50, density=True)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    #plt.savefig(os.path.join('random_gen.png'))

    t = -tau * np.log(beta + u * (alpha - beta))

    fig, ax = plt.subplots()

    ax.plot(x , decay(x,tau), color = func_color)
    ax.scatter(t, decay(t,tau), color = marker_color, marker = marker_st)

    plt.savefig(os.path.join('exp_dist.png'))

#**************************************************************
#42.4.2 Isotropic direction in 3D
elif exercise == 2:
    rng = np.random.default_rng()

    count = 0
    Sin, Cos = [], []
    npoints = 1000000
    while count < npoints:
        u1 = rng.random()
        u2 = rng.random()

        v1= 2 *u1 -1
        v2 = u2
        rsq = v1*v1 + v2*v2
        if rsq <= 1:
         count +=1
         Sin.append(2*v1*v2/rsq)
         Cos.append((v1*v1 - v2*v2)/rsq)

    fig, ax = plt.subplots(figsize =(15,15))
    ax.scatter(Cos, Sin, color = marker_color, marker = marker_st)

    plt.savefig(os.path.join('sin_and_cos.png'))

#**************************************************************
#REJECTION-ACCEPTANCE METHOD SAMPLING
elif exercise == 3 :
 #RAM parameters


    x_min = -1.0                            #interval of interest
    x_max = 1.0
    draw = np.linspace(x_min, x_max, 10000)

    sample_dim = 1.0e2                      #number of points in the sample
    n_bins = round(sample_dim**0.5)         #numbers of bins for scatter/hist plot
    M = 20.0                                # Parameter of selection (must be as small as possible) 
    rng = np.random.default_rng()
 #-----------------------------------------------
 #Gaussian distr parameters   
    sigma = 1.0 
    mu = 0.0
 #-----------------------------------------------
 #f(x) parameters and definition
    k = 1.0       
    alpha = -2.0
    beta = 1.0
    
    test_func = 1                   #use this to move between test functions f(x)         

    def myfunc(x):
        #f(x) functions used to test algorithm
        if test_func == 0: 
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return k * (x+2.0)**a 
        elif test_func == 1:
            #abs func (cusp)
            return -abs(x) + 2.0
        elif test_func == 2:
            return -x+2.0
        
    #print(integrate.quad(func_norm,x_min, x_max, (myfunc, arg, x_min, x_max, sample_dim), epsabs=abs_err, epsrel=rel_err)[0])               
 #-----------------------------------------------

    temp = []

    counter = 0                     #counter of successfully selected points 
    iter = 0                        #iterations need to collect the desired number of accepted points
    while counter < sample_dim:     #algorithm implementation

        y = bounded_norm_distr(mu, sigma, x_min, x_max)         
        #y = normal_distr(mu, sigma)
        u = rng.random()

        discr = func_norm(y, myfunc, x_min, x_max, 1.0) / M / gaussian(y, mu, sigma, x_min, x_max)
        if u < discr : 
            temp.append(y)
            counter += 1
        iter += 1 

    x = np.array(temp)
    print("Iterations needed: ", iter)

 #---------------------------------------------------------------------------------------------
 #plot
    fig, ax = plt.subplots()

    y_n, bin_edges = np.histogram(x, n_bins, density = False)
    bin_means = np.zeros(len(bin_edges)-1)
    for n in range(0 , len(bin_edges)-1):
        bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5       #mid point of bins
    
    bin_dim = np.diff(bin_edges)[0]                                #bins width


    #print(np.sum(y_n/sample_dim))
    y_err = (y_n)**0.5                                              #error on counts
    #ax.plot (draw, gaussian(draw, mu, sigma))
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    #ax.scatter(bin_means, y_n, marker = marker_st, color = marker_color)
    ax.errorbar(bin_means, y_n , yerr = y_err, color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Rej-Acc method)')
    ax.plot(draw, func_norm(draw, myfunc, x_min, x_max, sample_dim)* bin_dim, color = "black", label = "function f(x)")
    #ax.plot(draw, myfunc(draw) /i , color = "black", label = "function f(x)")



    plt.legend()
    plt.savefig(os.path.join("RAM_function_sample.png"))

#**************************************************************
#REJECTION-ACCEPTANCE METHOD MULTIPLE-SAMPLING (not working now)
elif exercise == 4 : 
    
    x_min = 0.0
    x_max = 10
    draw = np.linspace(x_min, x_max, 10000)
 
    sigma = 1.0
    mu = 0.0

    n_samples = 3
    sample_dim = 1.0e3
    n_bins = round(sample_dim**0.5)

    k = 1.0
    alpha = -2.0
    beta = 1.0
    arg = () 

    fig, ax = plt.subplots()
    
    def func_norm(x, func, arg, x_min, x_max, sample_dim):
        i =  integrate.quad(func,x_min, x_max, arg, epsabs=abs_err, epsrel=rel_err)[0]
        return func(x, *arg) / i * sample_dim
    
    def gaussian_norm(x , x0, sigma, x_min, x_max):
        norm1 = 1.0 / sigma / (2.0 * np.pi)**0.5
        norm2 = 1.0 / (erf((x_max-x0)/sigma/2.0**0.5) - erf((x_min-x0)/sigma/2.0**0.5))
        return np.exp(-(x - x0)*(x - x0)/sigma/sigma/2.0) * norm1 * norm2
    
    test_func = 2
    def myfunc(x):
        if test_func == 0: 
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return k * (x+2.0)**a 
        elif test_func == 1:
            #abs func (cusp)
            return -abs(x) + 2.0
        elif test_func == 2:
            return 2*(-x+2.0)
        
    
    M = 9.0

    for i in range(0, n_samples):
        rng = np.random.default_rng()
        temp = []

        counter = 0 
        iter = 0
        while counter < sample_dim:

            y = bounded_norm_distr(mu, sigma, x_min, x_max)
            u = rng.random()

            discr = func_norm(y, myfunc, arg, x_min, x_max, 1.0) / M / gaussian_norm(y, mu, sigma, x_min, x_max)
            if u < discr : 
                temp.append(y)
                counter += 1
            iter += 1 

        x = np.array(temp)

        print("Iterations needed for simulation" , i+1 , ": ", iter)

        y_n, bin_edges = np.histogram(x, n_bins, density = False)
        bin_means = np.zeros(len(bin_edges)-1)
        for n in range(0 , len(bin_edges)-1):
            bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5 
    
        bin_dim = np.diff(bin_edges)[0]
        y_err = (y_n)**0.5
        lab = "MCS #" + str(i+1)
        ax.errorbar(bin_means, y_n , yerr = y_err, capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = lab)

    ax.plot(draw, func_norm(draw, myfunc, arg, x_min, x_max, sample_dim)* bin_dim, color = "black", label = "function f(x)")

    plt.legend()
    plt.savefig(os.path.join("RAM_multi_sample.png"))

#**************************************************************
#INVERSE FUNCTION (ACCUMULATION) METHOD SAMPLING (using fsolve)
elif exercise == 5 :
    x_min = -1.0                            #interval of interest
    x_max = 1.0
    mid = (x_max + x_min)/2.0               #middle point can be use as first extimator
    draw = np.linspace(x_min, x_max, 10000)
  #-----------------------------------------------
 #f(x) parameters
    k = 1.0
    alpha = -2.0
    beta = 1.0
 #-----------------------------------------------

    #n_samples = 3                          
    sample_dim = 1.0e3                      #sample dimension
    n_bins = round(sample_dim**0.5)         #number of bins involved
    counter = 0                             # numb of success
    tol = 1.0e-10                           #desired tollerance (must be compatible with rel err of integrals)

    rng = np.random.default_rng()           #random seed

 #-----------------------------------------------
    #Functions definition
    test_func = 2
    def myfunc(x):
        if test_func == 0: 
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return (k * (x+2.0)**a) / 0.591333 
        elif test_func == 1:
            #abs func (cusp)
            return (-abs(x) + 2.0) / 3.0
        elif test_func == 2:
            return (-x+2.0)*0.25

 #-----------------------------------------------
    #plot and function arguments
    fig, ax = plt.subplots()
    arg1 = ()
    arg = (myfunc, x_min)
 #-----------------------------------------------
 
    temp = []
    while counter < sample_dim:     #algorithm implementation
        u = rng.random()
        
        zeros = fsolve(func_shifter, x_min, (accum_func,u, arg), xtol= tol) #use of fsolve !TAKE CARE OF EXTIMATOR!  
        y = zeros
        temp.append(y)
        counter +=1

    x = np.array(temp)

 #-----------------------------------------------
    #scatter/hist plot
    y_n, bin_edges = np.histogram(x, n_bins, density = False)
    bin_means = np.zeros(len(bin_edges)-1)
    for n in range(0 , len(bin_edges)-1):
        bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5     #mid point of my bins  
    
    bin_dim = np.diff(bin_edges)[0]                              #bins width

    y_err = (y_n)**0.5                                          #error on counts
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    ax.errorbar(bin_means, y_n , yerr = y_err, color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Inv func + fsolve)')
    
    ax.plot(draw, myfunc(draw)*bin_dim*sample_dim, color = "black", label = "function f(x)")

    plt.legend()
    plt.savefig(os.path.join("inv_func_sample(fsolve).png"))

#**************************************************************
#INVERSE FUNCTION (ACCUMULATION) METHOD SAMPLING (using custom root finders) (not working)
elif exercise == 6 :
    x_min = -1.0
    x_max = 1.0
    mid = (x_max + x_min)/2.0
    draw = np.linspace(x_min, x_max, 10000)
    bisec = True
    newton = True
 
    k = 1.0
    alpha = -2.0
    beta = 1.0

    #n_samples = 3
    sample_dim = 1.0e2
    n_bins = round(sample_dim**0.5)
    counter = 0
    tol = 1.0e-8
    n_max = 1.0e5

    rng = np.random.default_rng()

    test_func = 0
    def myfunc(x):
        if test_func == 0: 
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return (k * (x+2.0)**a) / 0.591333 
        elif test_func == 1:
            #abs func (cusp)
            return (-abs(x) + 2.0) / 3.0
        elif test_func == 2:
            return (-x+2.0)*0.25
        
    fig, ax = plt.subplots()
    arg1 = ()
    arg = (myfunc, arg1, x_min)
    if bisec:
        temp = []
        counter = 0
        while counter < sample_dim:
            u = rng.random()
        
            y = bisection(func_shifter, (accum_func, u), x_min, x_max, tol, n_max)
        
            temp.append(y)
            counter +=1

        x = np.array(temp)

    
        y_n, bin_edges = np.histogram(x, n_bins, density = False)
        bin_means = np.zeros(len(bin_edges)-1)
        for n in range(0 , len(bin_edges)-1):
            bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5 
    
        bin_dim = np.diff(bin_edges)[0]

        y_err = (y_n)**0.5
        #count, bins, ignored = ax.hist(x, n_bins, density=True)
        ax.errorbar(bin_means, y_n , yerr = y_err, color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Inv func + bisection)')
    

    if newton:
        temp = []
        counter = 0
        while counter < sample_dim:
            u = rng.random()
        
            y = newton_root_finder(func_shifter, myfunc, (accum_func, u), arg1, x_min, x_max, tol, n_max)
        
            temp.append(y)
            counter +=1

        x2 = np.array(temp)

    
        y_n, bin_edges = np.histogram(x2, n_bins, density = False)
        bin_means = np.zeros(len(bin_edges)-1)
        for n in range(0 , len(bin_edges)-1):
            bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5 
    
        bin_dim = np.diff(bin_edges)[0]

        y_err = (y_n)**0.5
        #count, bins, ignored = ax.hist(x, n_bins, density=True)
        ax.errorbar(bin_means, y_n , yerr = y_err, color='blue', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Inv func + newton)')
    
    ax.plot(draw, myfunc(draw)*bin_dim*sample_dim, color = "black", label = "function f(x)")

    plt.legend()
    plt.savefig(os.path.join("inv_func_sample(custom).png"))

#**************************************************************
#INVERSE FUNCTION (ACCUMULATION) METHOD SAMPLING (using cumulative func method)
elif exercise == 7 :
    x_min = -1.0                    #interval of interest
    x_max = 1.0
 #drawing and plot
    draw = np.linspace(x_min, x_max, 10000)
    fig, ax = plt.subplots()

 #-----------------------------------------------
    sample_dim = 1.0e3                      #sample dimension
    n_bins = round(sample_dim**0.5)         #number of bins
    bin_dim = (x_max - x_min) / n_bins      #bins width

 #-----------------------------------------------
 #f(x) parameters and definition
    k = 1.0
    alpha = -2.0
    beta = 1.0
    arg = ()

    test_func = 0
    if test_func == 0: 
        def myfunc(x):
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return (k * (x+2.0)**a) 
    elif test_func == 1:
        def myfunc(x):
            #abs func (cusp)
            return (-abs(x) + 2.0) 
    elif test_func == 2:
        def myfunc(x):

            return (-x+2.0)
 #-----------------------------------------------

    cum_points = np.linspace(x_min, x_max, n_bins)    #points where to evaluate cum funct
    norm = integrate.quad(myfunc, x_min, x_max, arg, epsabs= abs_err, epsrel=rel_err)[0] #normalization to unit

    cum_values = np.zeros(len(cum_points))      
    for i in range(len(cum_points)):                   #evaluation of cumulative function in the selected points
        integral = integrate.quad(myfunc, x_min, cum_points[i], arg, epsabs= abs_err, epsrel=rel_err)[0]
        cum_values[i] = integral / norm #normalized

    #rng = np.random.default_rng()
    #u = rng.random(sample_dim)
    u = np.random.random_sample(int(sample_dim))

    x = np.zeros(int(sample_dim))
    y_n = np.zeros(n_bins)              #count how many cases are in a bin
    for i in range(0 , int(sample_dim)):
        step = 0
        while (cum_values[step] < u[i]):
            step += 1
            
        x[i] = cum_values[step]
        y_n[step] +=1

 #-----------------------------------------------
    #plot
    vector_func = np.vectorize(myfunc)
    y_draw = vector_func(draw, *arg)
    ax.plot(draw, y_draw/norm*sample_dim*bin_dim, color = "black", label = "function f(x)")

    y_err = y_n ** 0.5
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    ax.errorbar(cum_points[1:], y_n[1:] , yerr = y_err[1:], color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Inv func + cum)')

    plt.legend()
    plt.savefig(os.path.join("inv_func_sample(cumulative).png"))

#**************************************************************
#INVERSE FUNCTION (ACCUMULATION) METHOD SAMPLING (cum func method on lum func)
elif exercise == 8 :
    x_min = 1.0e30             #interval of interest
    x_max = 1.0e37
 #-----------------------------------------------
    #drawing instructions
    draw = np.geomspace(x_min, x_max, 10000)
    draw_min = 1.0e29
    draw_max = 1.0e38
    fig, ax = plt.subplots()
    ax.set_xlim(draw_min, x_max)
    plt.ylim(1.0e-45 , 1.2e-27)

 #-----------------------------------------------
    sample_dim = 1.0e5            
    n_bins = round(sample_dim**0.5)

 #-----------------------------------------------
    test_func = 1
    if test_func == 0: 
        arg = (1.0, 1.0e33, 0.97, 2.6)      #disk data for ex
        def myfunc(x, norm, x_b, n1, n2):   #broken power law function  (be carefull about exponential renorm)  
            frac = x / x_b
            if  frac < 1:
                frac = frac**(-n1)
            else:
                frac = frac**(-n2)
        
            return (norm *frac)
    elif test_func == 1:
        arg = (1.0, 2.5e34, -0.66, 18.2)      #NPTF data for ex
        def myfunc(x, norm, x_b, n1, n2):   #broken power law function  (be carefull about exponential renorm)  
            frac = x / x_b
            if  frac < 1:
                frac = frac**(-n1)
            else:
                frac = frac**(-n2)
        
            return (norm *frac)

 #--------------------------

    cum_points = np.geomspace(x_min, x_max, n_bins, endpoint = False)    #points where to evaluate cum funct (to be fixed)
    norm = log_scale_int(myfunc, x_min, x_max, arg)[0]

    bin_dim = np.ones(n_bins)                           #bins width
    for i in range(1, len(cum_points)):
        bin_dim[i] = cum_points[i] - cum_points[i-1]

    cum_values = np.zeros(len(cum_points))
    for i in range(len(cum_points)):                    #evaluation of cumulative function in the selected points
        integral = log_scale_int(myfunc, x_min, cum_points[i],arg)
        cum_values[i] = integral[0] / norm              #normalized

    #rng = np.random.default_rng()
    #u = rng.random(sample_dim)
    u = np.random.random_sample(int(sample_dim))

    x = np.zeros(int(sample_dim))
    y_n = np.zeros(n_bins)
    for i in range(0 , int(sample_dim)):
        step = 0
        while (cum_values[step] < u[i]):
            step += 1
            
        x[i] = cum_values[step]
        y_n[step] +=1

 #-----------------------------------------------
  #print controls
    #print(cum_points)
    #print(bin_dim)
    #print(cum_values)
    #print(y_n)

 #-----------------------------------------------
    #plot
    vector_func = np.vectorize(myfunc)
    y_draw = vector_func(draw, *arg)
   
    ax.loglog(draw, y_draw/norm, color = "black", label = "function f(x)")

    y_err = y_n ** 0.5
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    ax.errorbar(cum_points[1:], y_n[1:]*(1.0/bin_dim[1:])/sample_dim, yerr = y_err[1:]*(1.0/bin_dim[1:])/sample_dim, color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'MCS (Inv func + cum)')

    plt.legend()
    plt.savefig(os.path.join("inv_func_sample_lum_func.png"))

#**************************************************************
#COMPARING (simple func)
elif exercise == 9:
    x_min = -1.0                    #interval of interest
    x_max = 1.0
 #drawing and plot
    draw = np.linspace(x_min, x_max, 10000)
    fig, ax = plt.subplots()

 #-----------------------------------------------
    sample_dim = 1.0e2                      #sample dimension
    n_bins = round(sample_dim**0.5)         #number of bins
    bin_dim = (x_max - x_min) / n_bins      #bins width

 #-----------------------------------------------
 #f(x) parameters and definition
    k = 1.0
    alpha = -2.0
    beta = 1.0
    args = ()

    test_func = 1
    if test_func == 0: 
        def myfunc(x):
            #log parabola
            a = alpha - beta * np.log10(x+2.0)
            return (k * (x+2.0)**a) 
    elif test_func == 1:
        def myfunc(x):
            #abs func (cusp)
            return (-abs(x) + 2.0) 
    elif test_func == 2:
        def myfunc(x):

            return (-x+2.0)
 #-----------------------------------------------
 #cumulative
    start_time = time.monotonic()

    cum_points = np.linspace(x_min, x_max, n_bins)    #points where to evaluate cum funct
    norm = integrate.quad(myfunc, x_min, x_max, args, epsabs= abs_err, epsrel=rel_err)[0] #normalization to unit

    cum_values = np.zeros(len(cum_points))      
    for i in range(len(cum_points)):                   #evaluation of cumulative function in the selected points
        integral = integrate.quad(myfunc, x_min, cum_points[i], args, epsabs= abs_err, epsrel=rel_err)[0]
        cum_values[i] = integral / norm #normalized

    u = np.random.random_sample(int(sample_dim))

    x = np.zeros(int(sample_dim))
    y_n_cum = np.zeros(n_bins)              #count how many cases are in a bin
    for i in range(0 , int(sample_dim)):
        step = 0
        while (cum_values[step] < u[i]):
            step += 1
            
        x[i] = cum_values[step]
        y_n_cum[step] +=1

    end_time = time.monotonic()
    print("Cumulative tech. execution time: " , timedelta(seconds= end_time - start_time))

 #-----------------------------------------------
 #fsolve
    start_time = time.monotonic()

    counter = 0                             # numb of success
    tol = 1.0e-10                           #desired tollerance (must be compatible with rel err of integrals)
    mid = (x_max + x_min)/2.0               #middle point can be use as first extimator
    

    arg = (func_norm, x_min, (myfunc, x_min, x_max))

    rng = np.random.default_rng()           #random seed

    temp = []
    while counter < sample_dim:     #algorithm implementation
        u = rng.random()
        
        zeros = fsolve(func_shifter, mid, (accum_func, u, arg), xtol= tol) #use of fsolve !TAKE CARE OF EXTIMATOR!  
        y = zeros
        temp.append(y)
        counter +=1

    x_fs = np.array(temp)

    end_time = time.monotonic()
    print("fsolve execution time: " , timedelta(seconds= end_time - start_time))
 #-----------------------------------------------
 #rejection
    start_time = time.monotonic()
    M = 20.0                                # Parameter of selection (must be as small as possible) 
    sigma = 1.0 
    mu = 0.0
    rng = np.random.default_rng()

    temp = []

    counter = 0                     #counter of successfully selected points 
    iter = 0                        #iterations need to collect the desired number of accepted points
    while counter < sample_dim:     #algorithm implementation

        y = bounded_norm_distr(mu, sigma, x_min, x_max)         
        #y = normal_distr(mu, sigma)
        u = rng.random()

        discr = func_norm(y, myfunc, x_min, x_max, 1.0) / M / gaussian(y, mu, sigma, x_min, x_max)
        if u < discr : 
            temp.append(y)
            counter += 1
        iter += 1 

    x_rej = np.array(temp)



    end_time = time.monotonic()
    print("Rejection execution time: " , timedelta(seconds= end_time - start_time))
    print("Iterations needed for rejection method: ", iter)
 #-----------------------------------------------

    #plots

    vector_func = np.vectorize(myfunc)
    y_draw = vector_func(draw, *args)
    ax.plot(draw, y_draw/norm*sample_dim*bin_dim, color = "black", label = "function f(x)")

    #cumulative
    y_err_cum = y_n_cum ** 0.5
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    ax.errorbar(cum_points[1:], y_n_cum[1:] , yerr = y_err_cum[1:], color='r', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'Inv func + cum')

    #fsolve
    y_n_fs, bin_edges_fs = np.histogram(x_fs, n_bins, density = False)
    bin_means_fs = np.zeros(len(bin_edges_fs)-1)
    for n in range(0 , len(bin_edges_fs)-1):
        bin_means_fs[n] = (bin_edges_fs[n+1] + bin_edges_fs[n]) * 0.5     #mid point of my bins  
    
    bin_dim_fs = np.diff(bin_edges_fs)[0]                              #bins width

    y_err_fs = (y_n_fs)**0.5                                          #error on counts
    #count, bins, ignored = ax.hist(x, n_bins, density=True)
    ax.errorbar(bin_means_fs, y_n_fs , yerr = y_err_fs, color='blue', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'Inv func + fsolve')
    
    #rejection
    y_n_rej, bin_edges_rej = np.histogram(x_rej, n_bins, density = False)
    bin_means_rej = np.zeros(len(bin_edges_rej)-1)
    for n in range(0 , len(bin_edges_rej)-1):
        bin_means_rej[n] = (bin_edges_rej[n+1] + bin_edges_rej[n]) * 0.5       #mid point of bins
    
    bin_dim_rej = np.diff(bin_edges_rej)[0]                                #bins width


    #print(np.sum(y_n/sample_dim))
    y_err_rej = (y_n_rej)**0.5                                              #error on counts
    ax.errorbar(bin_means_rej, y_n_rej , yerr = y_err_rej, color='g', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3, label = 'Rej-Acc method')
    
    plt.legend()
    plt.savefig(os.path.join("comparing.png"))

end_time = time.monotonic()
print("Execution time: " , timedelta(seconds= end_time - start_time))