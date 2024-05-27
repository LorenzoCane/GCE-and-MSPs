#######################################################################################
# In this file I study redo the analyses on Calore data(arXiv:1411.4647) trying to replecate Flux energy fit and total flux
# Created by Lorenzo Cane (Unito, LAPTh) on 14/04/2024
# This code is part of my thesis project
#######################################################################################

#import
import time 
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from iminuit import Minuit
from iminuit.cost import LeastSquares
import sys
sys.path.insert(0, '/home/lorenzo/GCE-and-MSPs/toolbox')
from tools import cmtokpc, log_scale_int, GeVtoerg
from gce import  *
#from jacobi import propagate
import os

#***************************************************************
#****************************************************************
#USEFUL VALUES 
start_time = time.monotonic()

f = open('flux.txt', 'w')
lines =['flux.py results', '\n************************************************* \n \n']
f.writelines(lines)
#integration conditions
abs_err = 0.0
rel_err = 1.0e-8
div_numb = 100
inf_approx = 1.0e50
#ROI bounds
b_min = np.deg2rad(2)              #ROI latitude min value
b_max = np.deg2rad(20)             #ROI latitude max value 
l_min = 0                          #ROI long. min value
l_max = b_max                      #ROI long. max value
#astronomical constants
rs = 20            #kpc            #scale radius (gNFW) 
g = 1.2            #kpc            #gamma-sloper (gNFW) 
rc = 8.5           #kpc            #Earth-GC distance
#solid angle integration
options = {'epsabs': abs_err, 'epsrel' : rel_err, 'limit' : div_numb}
#num = integrate.nquad(gNRW2, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, g, rc), opts=options)
ang_norm = 2*(b_max-b_min)*(np.sin(l_max)-np.sin(l_min))
print(ang_norm)
#energy integration bound
e_min = 0.1  #GeV
e_max = 10   #GeV
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
spec = np.divide(d.flux, d.emeans)
#****************************************************************
#Fit with a broken power law
l = LeastSquares(d.emeans, d.flux, d.full_sigma , broken_pl_arr) 
m = Minuit(l, 1.0e-6, 2.06, -0.58, 0.63, name=("F_0", "E_b", "n1", "n2" )) #following Dinsmore2022 notation

m.fixed["E_b", "n1", "n2"] = False

m.migrad()
m.hesse()

m.values[2], m.values[3] = m.values[2]+2 , m.values[3]+2  #to be better compared with Dinsmore values
f.write('Broken Power Law results\n')
f.write('Flux fit parameters (BPL):'), f.write('\n')
for key, value, error in zip(m.parameters, m.values, m.errors):
    line = [str(key), ' = ', str(value), ' +- ', str(error), '\n']
    for i in line: f.write(i)
m.values[2], m.values[3] = m.values[2]-2 , m.values[3]-2  #go back to fit values
#y, ycov = propagate(lambda norm, xb, n1, n2: broken_pl(d.emeans, norm, xb, n1, n2)[1], m.values, m.covariance)
#****************************************************************
#Calculation of the total flux
I = log_scale_int(broken_pl, e_min, e_max, tuple(m.values),inf_approx, abs_err, rel_err, div_numb)
#I = integrate.quad(broken_pl, e_min, e_max, args=tuple(m.values), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(I[0], "  ", I[1])
f.write('--------------------------------------------------------\n\n')
fluxres1 =["Flux (in sr units): \nF_Omega =" , str(I[0]), " [GeV/cm^2/s/sr] = ", str(GeVtoerg(I[0])) , "[erg/cm^2/s/sr]", '\n']
fluxres2 =["Flux: \nF =" , str(I[0]*ang_norm) , " [GeV/cm^2/s] = ", str(GeVtoerg(I[0])*ang_norm) , "[erg/cm^2/s]"]
for i in fluxres1: f.write(i)
for i in fluxres2: f.write(i)

#****************************************************************
#Fit with a power law (with exp cut)
l = LeastSquares(d.emeans, d.flux, d.full_sigma , power_law) 
m2 = Minuit(l, 1.0, 2.0 ,3.0e-7,  name=("alpha", "E_c", "norm" )) #following Dinsmore2022 notation

#m.fixed["E_b", "n1", "n2"] = False

m2.migrad()
m2.hesse()
f.write('\n************************************************* \n \n')
f.write('Power Law (exp cutoff) results\n')
f.write('Flux fit parameters (PL - exp. cut):'), f.write('\n')
for key, value, error in zip(m2.parameters, m2.values, m2.errors):
    line = [str(key), ' = ', str(value), ' +- ', str(error), '\n']
    for i in line: f.write(i)
#y, ycov = propagate(lambda norm, xb, n1, n2: broken_pl(d.emeans, norm, xb, n1, n2)[1], m.values, m.covariance)
#****************************************************************
#Calculation of the total flux
I = log_scale_int(power_law, e_min, e_max, tuple(m2.values),inf_approx, abs_err, rel_err, div_numb)
#I = integrate.quad(broken_pl, e_min, e_max, args=tuple(m.values), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(I[0], "  ", I[1])
f.write('--------------------------------------------------------\n\n')
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

#----------------------------------------------------------------------------


#****************************************************************
spec = np.divide(d.flux, np.multiply(d.emeans, d.emeans))
spec_err = np.divide(d.full_sigma, np.multiply(d.emeans*ang_norm, d.emeans))

#Fit with a broken power law
l = LeastSquares(d.emeans, spec, spec_err , broken_pl_arr) 
m2 = Minuit(l, 1.9e-7, 2.5, 1.42, 2.63, name=("K", "E_b", "n1", "n2" )) #following Dinsmore2022 notation

m2.fixed["K", "n1", "n2"] = False

m2.migrad()
m2.hesse()

#m2.values[2], m2.values[3] = m2.values[2]+2 , m2.values[3]+2  #to be better compared with Dinsmore values
f.write('\n\n----------------------------------\n')
f.write('Spectrum fit parameters:'), f.write('\n')
for key, value, error in zip(m2.parameters, m2.values, m2.errors):
    line = [str(key), ' = ', str(value), ' +- ', str(error), '\n']
    for i in line: f.write(i)
#m2.values[2], m2.values[3] = m2.values[2]-2 , m2.values[3]-2  #go back to fit values


#plot
x_min = 1.0e-1
x_max = 1.0e2
y_min = 1.0e-9
y_max =1.0e-5
en = np.geomspace(x_min, x_max)
spec_fit = broken_pl_arr(en, *m2.values)
fig, ax = plt.subplots()

ax.loglog()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel(r'Energy $E_\gamma$ [GeV]')
ax.set_ylabel(r'Spec dN/dE [$\mathregular{cm^2}$ / s / GeV]')

ax = plt.errorbar(d.emeans,np.multiply(spec,1.0), yerr = spec_err,
color='black', capsize=1, capthick=1,ls='',
elinewidth=0.5,marker='o',markersize=3, label='Spectrum')

plt.plot(en, spec_fit, color = "red", ls='-.', label = 'Broken power law fit')

plt.legend()

plt.savefig(os.path.join('spec_bpl.png')) 


f.close()
end_time = time.monotonic()
print(timedelta(seconds= end_time - start_time))

print("Results written in flux.txt file")