

#import

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os

#****************************************************************************
#LUMINOSITY FUNCTIONS DEFINITIONS

#power law lum func(high flux exp c-o, low flux step c-o)
def power_law (l, alpha , l_min , l_max):   #alpha: slope, l_min = step c-o, l_max: exp c-o               
    den = gamma((l_min/l_max)) * l_max**(1-alpha)
    num = l**(-alpha) * np.exp(-l/l_max)

    return num / den 
#-----------------------------------------------------------------------------

#log normal lum func
def log_norm (l, l_0 , sigma):                          
    num = np.log10(np.e)
    den = sigma * l * (2*np.pi)**0.5 
    exp = np.exp(-1.0 * (np.log10(l/l_0))**2 / (2 * sigma * sigma))

    return num * exp / den 
#-----------------------------------------------------------------------------

#broken power law lum func
def broken_pl(l, l_b, n1, n2):             #l_b: broken point, n1: 1st part index, n2: 2nd part index               
    norm = (1 - n1) * (1 -n2) / l_b / (n1 - n2)
    frac = 1.0
    bpl = []

    for a in range(0, len(l)):
        if l[a] < l_b:
            frac = (l[a]/l_b)**(-n1)
        else:
            frac = (l[a]/l_b)**(-n2)
        bpl.append(norm *frac)
    return bpl
#****************************************************************************
#BENCHMARKS EVALUATION

#useful values
min = 1.0e29       #erg s^(-1)     #min for graph
max = 1.0e38       #erg s^(-1)     #max for graph
r = 1.11e-46       #cm^(-2)        #flux/lum ratio
#linestyle
ls1 = '--'
ls2 = 'dotted'

#luminosity and flux 
l= np.geomspace(min, max,10000)    #luminosity
f_pl = l * r                       #flux

#-----------------------------------------------------------------------------
#wavelet 1 - power law 
l_m1 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M1 = 1.0e35  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha1 = 1.94  #slope

p_pl1 = power_law(l , alpha1 , l_m1 , l_M1) #ming power law lum func

#-----------------------------------------------------------------------------
#wavelet 2 - power law 
l_m2 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M2 = 7.0e34  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha2 = 1.5  #slope

p_pl2 = power_law(l, alpha2, l_m2, l_M2) #bartles power law lum funcs

#-----------------------------------------------------------------------------
#GLC - log normal
l_0_glc = 8.8e33
sigma_glc = 0.62 

p_glc = log_norm(l, l_0_glc, sigma_glc)  #log norm from global cluster obs

#-----------------------------------------------------------------------------
#GCE - log normal
l_0_gce = 1.3e32
sigma_gce = 0.70

p_gce = log_norm(l, l_0_gce, sigma_gce)  #log norm from bulge obs

#-----------------------------------------------------------------------------
#AIC - log normal
l_0_aic = 4.3e30
sigma_aic = 0.94

p_aic = log_norm(l, l_0_aic, sigma_aic)  #log norm from bulge obs

#-----------------------------------------------------------------------------
#Disk - broken power law
n1_disk = 0.97
n2_disk = 2.60
l_b_disk = 1.7e33
p_disk = []

p_disk = broken_pl(l, l_b_disk, n1_disk, n2_disk)

#-----------------------------------------------------------------------------
#NPTF - broken power law
n1_nptf = -0.66
n2_nptf = 18.2
l_b_nptf = 2.5e34
p_nptf = []

p_nptf = broken_pl(l, l_b_nptf, n1_nptf, n2_nptf)

#****************************************************************************
#PLOTS

#power laws plot

fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.loglog(l,p_pl1, color = 'lightblue', label = 'Wavelet 1')
ax1.loglog(l, p_pl2, color = 'green', label = 'Wavelet 2 ')
ax1.loglog(l, p_glc, color = 'darkorange', linestyle = ls1, label= 'GLC')
ax1.loglog(l, p_gce, color = 'fuchsia', linestyle = ls1, label = 'GCE')
ax1.loglog(l, p_aic, color = 'red' , linestyle = ls1 , label = 'AIC')
ax1.loglog(l, p_disk, color = 'gray' , linestyle = ls2, label = 'Disk')
ax1.loglog(l, p_nptf, color = 'black', linestyle = ls2,  label = 'NPTF')
ax2.loglog(f_pl, p_pl1, color = 'lightblue')

#graphical config
ax1.set_yticks(np.geomspace(1.0e-45, 1.e-29, 5))
ax1.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax2.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))

ax1.set_xlabel("Luminosity L [erg / s]")
ax2.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]') 
ax1.set_ylabel(r'dN / dL')

plt.ylim(1.0e-45 , 1.2e-29)
ax1.set_xlim(min, max)
ax2.set_xlim(min*r, max*r)
ax1.legend()

plt.savefig(os.path.join('pl.png'))

#----------------------------------------------------------------------------
#e-fold in L plot

fig2 , ax3 = plt.subplots()
ax4 = ax3.twiny()

ax3.loglog(l, l*p_pl1, color = 'lightblue', label = 'Wavelet 1')
ax3.loglog(l, l*p_pl2, color = 'green', label = 'Wavelet 2 ')
ax3.loglog(l, l*p_glc, color = 'darkorange', linestyle = ls1, label= 'GLC')
ax3.loglog(l, l*p_gce, color = 'fuchsia', linestyle = ls1, label = 'GCE')
ax3.loglog(l, l*p_aic, color = 'red' , linestyle = ls1 , label = 'AIC')
ax3.loglog(l, l*p_disk, color = 'gray' , linestyle = ls2, label = 'Disk')
ax3.loglog(l, l*p_nptf, color = 'black', linestyle = ls2,  label = 'NPTF')
ax4.loglog(f_pl, l * p_pl1, color = 'lightblue')

#graphical config
ax3.set_yticks(np.geomspace(1.0e-7, 1.0e2, 4))
ax3.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax4.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))

ax3.set_xlabel("Luminosity L [erg / s]")
ax4.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]')
ax3.set_ylabel("LdN / dL [erg / s]")

plt.ylim(1.0e-7, 1.0e3)
ax3.set_xlim(min, max)
ax4.set_xlim(min*r, max*r)
ax3.legend()

plt.savefig(os.path.join('1efold.png'))

#----------------------------------------------------------------------------
#2e-folds in L plot

fig3 , ax5 = plt.subplots()
ax6 = ax5.twiny()

ax5.loglog(l, l*l*p_pl1, color = 'lightblue', label = 'Wavelet 1')
ax5.loglog(l, l*l*p_pl2, color = 'green', label = 'Wavelet 2 ')
ax5.loglog(l, l*l*p_glc, color = 'darkorange', linestyle = ls1, label= 'GLC')
ax5.loglog(l, l*l*p_gce, color = 'fuchsia', linestyle = ls1, label = 'GCE')
ax5.loglog(l, l*l*p_aic, color = 'red' , linestyle = ls1 , label = 'AIC')
ax5.loglog(l, l*l*p_disk, color = 'gray' , linestyle = ls2, label = 'Disk')
ax5.loglog(l, l*l*p_nptf, color = 'black', linestyle = ls2,  label = 'NPTF')
ax6.loglog(f_pl, l*l* p_pl1, color = 'lightblue')

#graphical config
ax5.set_yticks(np.geomspace(1.0e28, 1.0e34, 3))
ax5.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax6.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))

ax5.set_xlabel("Luminosity L [erg / s]")
ax6.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]')
ax5.set_ylabel(r' $\mathregular{L^2}$ dN / dL [$\mathregular{erg^2}$ / $\mathregular{s^2}$]')

plt.ylim(1.0e27, 1.0e35)
ax5.set_xlim(min, max)
ax6.set_xlim(min*r, max*r)
ax5.legend()
plt.savefig(os.path.join('2efold.png'))