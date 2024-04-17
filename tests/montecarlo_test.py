#Exercise from PDG reviews

import time 
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import erf
import os
import sys
sys.path.insert(0, '/home/lorenzo/GCE-and-MSPs/toolbox')
from tools import bounded_norm_distr, log_scale_int
start_time = time.monotonic()

exercise = 3
marker_st =  '+'
marker_color = 'orange'
func_color = 'black'

npoints = 1000000
rng = np.random.default_rng()

#42.4.1 Exponential decay
if exercise == 1:
    
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

#42.4.2 Isotropic direction in 3D
elif exercise == 2:
    count = 0
    Sin, Cos = [], []
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
#FUNCTION SAMPLING
elif exercise == 3 :

    inf_approx = 1.0e20

    x_max = 2.5
    x_min = 0.5
    draw = np.geomspace(x_min, x_max, 10000)

    sigma = 0.6
    mu = 1.0


    sample_dim = 10000
    n_bins = round(sample_dim**0.5)

    k = 1.0
    alpha = -2.0
    beta = 1.0
    arg = () 
    
    def func_norm(x, func, arg, x_min, x_max):
        i =  log_scale_int(func, x_min, x_max, inf_approx, arg, 0.0, 1.0e-4, 50)[0]
        return func(x, *arg) / abs(i) 
    
    def gaussian_norm(x , x0, sigma, x_min, x_max):
        norm1 = 1.0 / sigma / (2.0 * np.pi)**0.5
        norm2 = 1.0 / (erf((x_max-x0)/sigma/2.0**0.5) - erf((x_min-x0)/sigma/2.0**0.5))
        return np.exp(-(x - x0)*(x - x0)/sigma/sigma/2.0) * norm1 * norm2
    
    
    def myfunc(x):
        a = np.exp(5.0/x) - 1.0
        return 1.0 / a / x**5.0
    

   
    M = 15

    temp = []

    counter = 0 
    iter = 0
    while counter < sample_dim:

        y = bounded_norm_distr(mu, sigma, x_min, x_max)
        u = rng.random()

        discr = func_norm(y, myfunc, arg, x_min, x_max) / M / gaussian_norm(y, mu, sigma, -np.infty, np.infty)
        if u < discr : 
            temp.append(y)
            counter += 1
        iter += 1

    x = np.array(temp)
    print("Iterations needed: ", iter)


 #plot

    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    ax.plot(draw, func_norm(draw, myfunc, arg, x_min, x_max), color = "black", label = "function f(x)")
    #ax.plot (draw, gaussian(draw, mu, sigma))
    count, bins, ignored = ax.hist(x, n_bins, density=True)
    y_n, bin_edges = np.histogram(x, n_bins, density = True)
    bin_means = np.zeros(len(bin_edges)-1)
    for n in range(0 , len(bin_edges)-1):
        bin_means[n] = (bin_edges[n+1] + bin_edges[n]) * 0.5 

    ax.scatter(bin_means, y_n, color = 'red')
    #y_err = 1.0/ y_n**0.5
    #ax2 = plt.errorbar(bin_means, y_n, yerr = y_err, color='black', capsize=1, capthick=1,ls='', elinewidth=0.5,marker='o',markersize=3)



    #plt.legend()
    plt.savefig(os.path.join("function_sample.png"))


end_time = time.monotonic()
print("Execution time: " , timedelta(seconds= end_time - start_time))