
#import

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from iminuit import Minuit
from iminuit.cost import LeastSquares
from gce import gNRW2, broken_pl_arr, broken_pl, GeVtoerg
from jacobi import propagate
import os

#****************************************************************
#****************************************************************
#USEFUL VALUES 
f = open('flux.txt', 'w')
lines =['flux.py results', '\n', '\n']
f.writelines(lines)

b_min = np.deg2rad(2)              #ROI latitude min value
b_max = np.deg2rad(20)             #ROI latitude max value 
l_min = 0                          #ROI long. min value
l_max = b_max                      #ROI long. max value
rs = 20            #kpc            #scale radius (gNFW) 
g = 1.2            #kpc            #gamma-sloper (gNFW) 
rc = 8.5           #kpc            #Earth-GC distance

#integration over ROI
num = integrate.nquad(gNRW2, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, g, rc))[0]
i1 = integrate.quad(np.cos, l_min , l_max)[0]
i2 = integrate.quad(np.cos, b_min , b_max)[0]
ang_norm =4*i1*i2 
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
m = Minuit(l, 1.0e-6, 2, -0.6, 0.6, name=("F_0", "E_b", "n1", "n2" )) #following Dinsmore2022 notation

m.migrad()
m.hesse()

f.write('Fit parameters:'), f.write('\n')
for key, value, error in zip(m.parameters, m.values, m.errors):
    line = [str(key), ' = ', str(value), ' +- ', str(error), '\n']
    for i in line: f.write(i)

#y, ycov = propagate(lambda norm, xb, n1, n2: broken_pl(d.emeans, norm, xb, n1, n2)[1], m.values, m.covariance)
#****************************************************************
#Calculation of the total flux
I = integrate.quad(broken_pl, 0.1, 10, args=tuple(m.values))

f.write('\n')
fluxres1 =["Flux (in sr units): \nF_Omega =" , str(I[0]), " [GeV/cm^2/s/sr] = ", str(GeVtoerg(I[0])) , "[erg/cm^2/s/sr]", '\n']
fluxres2 =["Flux: \nF =" , str(I[0]*ang_norm) , " [GeV/cm^2/s] = ", str(GeVtoerg(I[0])*ang_norm) , "[erg/cm^2/s]"]
for i in fluxres1: f.write(i)
for i in fluxres2: f.write(i)

#****************************************************************
#PLOTS

#Flux in sr units
x_min = 1.0e-1
x_max = 1.0e2
y_min = 1.0e-9
y_max =1.0e-5
en = np.geomspace(x_min, x_max)
flux_fit = broken_pl_arr(en, *m.values)
fig, ax = plt.subplots()

ax.loglog()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel(r'Energy $E_\gamma$ [GeV]')
ax.set_ylabel(r'Flux $F_\gamma$ [GeV / $\mathregular{cm^2}$ / s / sr]')

ax = plt.errorbar(d.emeans,np.multiply(d.flux,1), yerr = d.flux_err,
color='black', capsize=1, capthick=1,ls='',
elinewidth=0.5,marker='o',markersize=3, label='Flux data from Calore et al. 2015')

plt.plot(en, flux_fit, color = "red", ls='-.', label = 'Broken power law fit')

plt.legend()

plt.savefig(os.path.join('flux_sr.png'))

#-------------------------------------------------------------------------
#Flux

fig2, ax2 = plt.subplots()

ax2.loglog()
ax2.set_xlim(1.0e-1, 1.0e2)
ax2.set_ylim(1.0e-9, 1.0e-6)
ax2.set_xlabel(r'Energy $E_\gamma$ [GeV]')
ax2.set_ylabel(r'Flux $F_\gamma$ [GeV / $\mathregular{cm^2}$ / s]')

ax2 = plt.errorbar(d.emeans,np.multiply(d.flux,ang_norm), yerr = np.multiply(d.flux_err, ang_norm),
color='black', capsize=1, capthick=1,ls='',
elinewidth=0.5,marker='o',markersize=3, label='Flux data from Calore et al. 2015')

plt.plot(en, np.multiply(flux_fit, ang_norm), color = "red", ls='-.', label = 'Broken power law fit')

plt.legend()

plt.savefig(os.path.join('flux.png'))