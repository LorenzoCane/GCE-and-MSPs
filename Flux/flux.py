
#import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate
import os

#****************************************************************
#****************************************************************

#useful functions

#array broken power law function  (be carefull about exponential renorm)  
def broken_pl_arr(x, norm, x_b, n1, n2): #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index               
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
#broken power law function  (be carefull about exponential renorm)  

def broken_pl(x, norm, x_b, n1, n2):   #norm: normalization, x_b: broken point, n1: 1st part index, n2: 2nd part index               
    frac = x / x_b
    if frac < 1:
        frac = frac**(2-n1)
    else:
        frac = frac**(2-n2)
        
    return (norm *frac)
#-------------------------------------------------------

def GeVtoerg(x):
    return x * 0.00160218

#****************************************************************
#Import data of Calore et al. 2015

class Data(object):
    def __init__(self):
        data = np.loadtxt('covariance.dat')
        self.ebins = data[:,0:2]  # Energy bins [GeV]
        self.emeans = self.ebins.prod(axis=1)**0.5  # Geometric mean energy [GeV]
        self.de = self.ebins[:,1] - self.ebins[:,0]  # Energy bin width [GeV]
        self.flux = data[:,2]  # (Average flux)*E^2 in energy bin[GeV/cm2/s/sr]
        self.flux_err = data[:,3]  # Flux error [GeV/cm2/s/sr]
        self.empirical_variance = data[:,4:28]  # Only empirical component [(GeV/cm2/s/sr)^2]
        self.full_variance = data[:,28:]  # Variance as it enters the spectral fit [(GeV/cm2/s/sr)^2]
        self.empirical_sigma = np.sqrt(np.diagonal(self.empirical_variance)) # Diagonal elements
        self.full_sigma = np.sqrt(np.diagonal(self.full_variance))  # Diagonal elements
        self.invSigma = np.linalg.inv(self.full_variance)  # Inverse matrix


d = Data() 

#****************************************************************
#Fit with a broken power law
l = LeastSquares(d.emeans, d.flux, d.flux_err, broken_pl_arr) 
m = Minuit(l, 1.0e-6, 2, 1.42, 2.6, name=("F_0", "E_b", "n1", "n2" )) #following Dinsmore2022 notation

m.migrad()
m.hesse()
#print(repr(m.params))

en = np.geomspace(d.emeans[0], d.emeans[-1])
flux_fit = broken_pl_arr(en, *m.values)
#y, ycov = propagate(lambda norm, xb, n1, n2: broken_pl(d.emeans, norm, xb, n1, n2)[1], m.values, m.covariance)
#****************************************************************
#Calculation of the total flux

I = integrate.quad(broken_pl, 0.1, 10, args=(1.0e-6, 2.06, 1.42, 2.63))
print("Total Flux =" , I[0], " [GeV/cm^2/s/sr] = ", GeVtoerg(I[0]) , "[erg/cm^2/s/sr]")


#****************************************************************
#Plot

fig, ax = plt.subplots()

ax.loglog()
ax.set_xlim(1.0e-1, 1.0e2)
ax.set_ylim(1.0e-9, 1.0e-5)
ax.set_xlabel(r'Energy $E_\gamma$ [GeV]')
ax.set_ylabel(r'Flux $F_\gamma$ [GeV / $\mathregular{cm^2}$ / s / sr]')

ax = plt.errorbar(d.emeans,np.multiply(d.flux,1), yerr = d.flux_err,
color='black', capsize=1, capthick=1,ls='',
elinewidth=0.5,marker='o',markersize=3, label='Flux data from Calore et al. 2015')

plt.plot(en, flux_fit, color = "red", ls='-.', label = 'Broken power law fit')

plt.legend()

plt.savefig(os.path.join('flux.png'))