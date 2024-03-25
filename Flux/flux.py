
#import

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gammaincc,gamma, exp1
import os

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

fig, ax = plt.subplots()
ax.loglog()
ax.set_xlim(1.0e-1, 1.0e2)
ax.set_ylim(1.0e-9, 1.0e-6)
ax.set_xlabel(r'Energy $E_\gamma$ [GeV]')
ax.set_ylabel(r'Flux $F_\gamma$ [GeV / $\mathregular{cm^2}$ / s / sr]')
ax = plt.errorbar(d.emeans,np.multiply(d.flux,1), yerr = d.flux_err,
color='#00A693', capsize=1, capthick=1,ls='',
elinewidth=0.5,marker='o',markersize=3, label='Template fitting')#,0label='Flux with stat. errors')

plt.savefig(os.path.join('flux.png'))