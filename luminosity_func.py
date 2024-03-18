

#import

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os

#****************************************************************************
#LUMINOSITY FUNCTION EVALUATION

min = 1.0e29      #erg s^(-1)     #min for graph
max = 1.0e38       #erg s^(-1)     #max for graph
r = 1.11e-46       #cm^(-2)        #flux/lum ratio

l= np.linspace(min, max,10000)     #luminosity
f_pl = l * r                       #flux

#power law - wavelet 1
l_m1 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M1 = 1.0e35  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha1 = 1.94  #slope



p_pl1 = l**(-alpha1) * np.exp(-l/l_M1) / (gamma((l_m1/l_M1)) * l_M1**(1.0-alpha1)) #power law lum funcs

#-----------------------------------------------------------------------------
#power law - wavelet 2
l_m2 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M2 = 7.0e34  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha2 = 1.95  #slope

p_pl2 = l**(-alpha2)  *np.exp(-l/l_M2) / (gamma((l_m2/l_M2)) * l_M2**(1.0-alpha2)) #power law lum funcs

#****************************************************************************
#PLOTS

#power law

fig, ax1 = plt.subplots(figsize= (16,9))
ax2 = ax1.twiny()

ax1.set_yticks(np.linspace(1.0e-45, 1.e-29, 5))
ax1.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax2.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))


ax1.set_xlabel("Luminosity L [erg / s]")
ax2.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]')
ax1.set_ylabel("dN / dL")

ax1.loglog(l,p_pl1, color = 'lightblue')
ax1.loglog(l, p_pl2, color = 'green')
ax2.loglog(f_pl, p_pl1, color = 'lightblue')
ax2.loglog(f_pl, p_pl2, color = 'green')

#fig.tight_layout()
plt.ylim(1.0e-45 , 1.0e-29)

plt.savefig(os.path.join('pl.png'))

#----------------------------------------------------------------------------
#e-fold

fig2 , ax3 = plt.subplots(figsize = (15,15))
ax4 = ax3.twiny()

ax3.set_yticks(np.geomspace(1.0e-7, 1.0e2, 4))
ax3.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax4.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))

ax3.set_xlabel("Luminosity L [erg / s]")
ax4.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]')
ax3.set_ylabel("LdN / dL")

ax3.loglog(l,l * p_pl1)
ax3.loglog(l, l * p_pl2)
#ax4.loglog(f_pl,l * p_pl1)

fig2.tight_layout()
plt.ylim(1.0e-7, 1.5e3)

plt.savefig(os.path.join('1efold.png'))