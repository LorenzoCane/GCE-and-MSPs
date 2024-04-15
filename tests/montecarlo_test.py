#Exercise from PDG reviews

import time 
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import erf
import os
start_time = time.monotonic()

exercise = 3
marker_st =  '+'
marker_color = 'orange'
func_color = 'black'

npoints = 1000
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

    x_max = 1.0
    x_min = -x_max
    draw = np.linspace(x_min, x_max, 10000)
    n_bins = 100
    sigma = 0.5
    mu = 0.0

    sample_dim = 100000

    arg = () 
    
    def func_norm(x, func, arg, x_min, x_max):
       i = integrate.quad(func, x_min, x_max, *arg, epsabs=0.0, epsrel=1.0e-6)[0]
       return func(x, *arg) / i 
    
    def lim_gaussian(x , x0, sigma, x_min, x_max):
       norm1 = 1.0 / sigma * (2.0 * np.pi)**0.5

       return np.exp(-(x - x0)*(x - x0)/sigma/sigma/2) * norm1
    
    
    def myfunc(x):
      return -(x*x)+2
    

   
    M = 1

    temp = []

    counter = 0 

    while counter < sample_dim:
       
        u1 = rng.random()
        u2 = rng.random()

        y = sigma * (np.sin(2*np.pi*u1) * (-2 * np.log(u2))**0.5) + mu
        u = rng.random()

        discr = func_norm(y, myfunc, arg, x_min, x_max) / M / gaussian(y, mu, sigma)
        if u < discr : 
            temp.append(y)
            counter += 1

    x = np.array(temp)
    print(x)


 #plot

fig, ax = plt.subplots()

ax.plot(draw, func_norm(draw, myfunc, arg, x_min, x_max), color = "black", label = "function f(x)")
#ax.plot (draw, gaussian(draw, mu, sigma))
count, bins, ignored = ax.hist(x, n_bins, density=True)


plt.legend()
plt.savefig(os.path.join("function_sample.png"))


end_time = time.monotonic()
print(timedelta(seconds= end_time - start_time))