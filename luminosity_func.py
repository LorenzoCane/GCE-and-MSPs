

#import

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os

#****************************************************************************
min = 1.0e29
max = 1.0e37

#power law - wavelet 1
l_m1 = 1.0e29  #L_min : low flux step-func cutoff
l_M1 = 1.0e35  #L_MAX : high flux exp cutoff
alpha1 = 1.94  #slope

l= np.linspace(min, max,100000)


p_pl = l**(-alpha1)*np.exp(-l/l_M1) / (gamma((l_m1/l_M1)) * l_M1**(1.0-alpha1))

fig, ax = plt.subplots()
plt.ylim(1.0e-45,1.0e-25)
plt.xlim(min, 1.0e38)



ax.loglog(l,p_pl)

plt.savefig(os.path.join('pl.png'))