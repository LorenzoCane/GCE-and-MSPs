

#import
import time 
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import gammaincc
import os
from gce import gNRW2, sgNRW, cmtokpc, power_law, log_norm, broken_pl_arr, broken_pl, l_log, l_bpl #, n_integrand

#****************************************************************************
#USEFUL VALUES AND DEF
start_time = time.monotonic()

f = open('lum_func.txt', 'w')      #output file
f.write('luminosity_func.py results \n')
f.write('************************************************* \n \n')

#luminosity values
min = 1.0e29       #erg s^(-1)     #luminosity min 
max = 1.0e38       #erg s^(-1)     #luminosity max
l_th = 1.0e34      #erg s^(-1)     # luminosity threshold
#integration conditions
abs_err = 0.0
rel_err = 1.0e-6
div_numb = 100
#ROI bounds
b_min = np.deg2rad(2)              #ROI latitude min value
b_max = np.deg2rad(20)             #ROI latitude max value 
l_min = 0                          #ROI long. min value
l_max = b_max                      #ROI long. max value

#astronomical constants
rs = 20            #kpc            #scale radius (gNFW) 
g = 1.2            #kpc            #gamma-sloper (gNFW) 
rc = 8.5           #kpc            #Earth-GC distance
f_obs = 1.8e-9     #erg s^(-1)     #observed flux 

#Flux/Lum ratio
options = {'epsabs': abs_err, 'epsrel' : rel_err, 'limit' : div_numb}
num = 4*integrate.nquad(gNRW2, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, g, rc), opts=options)
den = 4* integrate.nquad(sgNRW, [[1.0e-6 , np.infty], [l_min, l_max], [b_min, b_max]] , args=(rs, g, rc), opts = options)
#print(den[0], "  ", den[1])
#print(num[0], "  ", num[1])
#!!! den is given in kpc^-2 !!!!
r = cmtokpc(cmtokpc(num[0] / den[0] / 4 /np.pi))       #cm^(-2)        #flux/lum ratio

line = ['Flux/Lum ratio: ', '\n','F/L = ', str(r), ' cm^(-2) \n']
for l in line:
    f.write(l)

f.write('-------------------------------------------------- \n \n')

#linestyle

ls1 = '--'
ls2 = 'dotted'

#luminosity and flux 
l= np.geomspace(min, max,10000)    #luminosity
f_l = l * r                       #flux

#****************************************************************************
#BENCHMARKS EVALUATION
f.write('Number of total PSs and number of resolved PSs: \n')
#wavelet 1 - power law 
l_m1 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M1 = 1.0e35  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha1 = 1.94  #slope

p_pl1 = power_law(l , alpha1 , l_m1 , l_M1, True) #ming power law lum func

d = integrate.quad(power_law, min, max, args=(alpha1-1 , l_m1 , l_M1, True), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(power_law, l_th, max, args=(alpha1 , l_m1 , l_M1, True), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(d[0], "  ", d[1])
f.write('\nWavelet 1: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

f.write('\n')

#----------------------------------------------------------------------------g
#wavelet 2 - power law 
l_m2 = 1.0e29  #erg s^(-1)  #L_min : low flux step-func cutoff
l_M2 = 7.0e34  #erg s^(-1)   #L_MAX : high flux exp cutoff
alpha2 = 1.5  #slope

p_pl2 = power_law(l, alpha2, l_m2, l_M2, False) #bartles power law lum func

d = integrate.quad(power_law, min, max, args=(alpha2-1 , l_m2 , l_M2, False), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(power_law, l_th, max, args=(alpha2 , l_m2 , l_M2, False), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(d[0], "  ", d[1])

f.write('\n\nWavelet 2: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))



#-----------------------------------------------------------------------------
#GLC - log normal
l_0_glc = 8.8e33
sigma_glc = 0.62 

p_glc = log_norm(l, l_0_glc, sigma_glc)  #log norm from global cluster obs

d = integrate.quad(l_log, min, max, args=(l_0_glc, sigma_glc), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(log_norm, l_th, max, args=(l_0_glc, sigma_glc), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(n[0], "  ", n[1])

f.write('\n\nGLC: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

#-----------------------------------------------------------------------------
#GCE - log normal
l_0_gce = 1.3e32
sigma_gce = 0.70

p_gce = log_norm(l, l_0_gce, sigma_gce)  #log norm from bulge obs

d = integrate.quad(l_log, min, max, args=(l_0_gce, sigma_gce), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(log_norm, l_th, max, args=(l_0_gce, sigma_gce), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(n[0], "  ", n[1])

f.write('\n\nGCE: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

#-----------------------------------------------------------------------------
#AIC - log normal
l_0_aic = 4.3e30
sigma_aic = 0.94

p_aic = log_norm(l, l_0_aic, sigma_aic)  #log norm from bulge obs

d = integrate.quad(l_log, min, max, args=(l_0_aic, sigma_aic), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(log_norm, l_th, max, args=(l_0_aic, sigma_aic), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(n[0], "  ", n[1])

f.write('\n\nAIC: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

#-----------------------------------------------------------------------------
#NPTF - broken power law
n1_nptf = -0.66
n2_nptf = 18.2
l_b_nptf = 2.5e34
norm_nptf= (1-n1_nptf)*(1-n2_nptf) / l_b_nptf / (n1_nptf - n2_nptf)
p_nptf = []

p_nptf = broken_pl_arr(l, norm_nptf, l_b_nptf, n1_nptf, n2_nptf)

#Search of integral upper bound
#int_max = 1.0e34
#contr = 1
#while contr > 1.0e-50:
#    int_max *= 2
#    contr = broken_pl(int_max, norm_nptf, l_b_nptf, n1_nptf, n2_nptf)
#
#print(int_max, "   ", contr)

int_max =3.2e35 #int upper bound as found before

d = integrate.quad(l_bpl, min, int_max, args=(norm_nptf, l_b_nptf, n1_nptf, n2_nptf), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(broken_pl, l_th, int_max, args=(norm_nptf, l_b_nptf, n1_nptf, n2_nptf), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(n[0], "  ", n[1])

f.write('\n\nNPTF: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

#-----------------------------------------------------------------------------
#Disk - broken power law
n1_disk = 0.97
n2_disk = 2.60
l_b_disk = 1.7e33
l_m_disk = 1.0e30
l_M_disk = 1.0e37
a = (n1_disk - n2_disk) - (1-n2_disk)*(l_m_disk/l_b_disk)**(1-n1_disk)+ (1-n1_disk)*(l_M_disk/l_b_disk)**(1-n2_disk)
norm_disk = (1-n1_disk)*(1-n2_disk) / l_b_disk / (n1_disk - n2_disk)
p_disk = []

p_disk = broken_pl_arr(l, norm_disk, l_b_disk, n1_disk, n2_disk)


norm_disk_lim = (1-n1_disk)*(1-n2_disk) / l_b_disk / a

#num_range = np.geomspace(l_th, l_M_disk,10000)
#den_range = np.geomspace(l_m_disk, l_M_disk, 10000)
d = integrate.quad(l_bpl, l_m_disk, l_M_disk, args=(norm_disk_lim, l_b_disk, n1_disk, n2_disk), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
n = integrate.quad(broken_pl, l_th, l_M_disk, args=(norm_disk_lim, l_b_disk, n1_disk, n2_disk), epsabs = abs_err, epsrel = rel_err, limit=div_numb)
#print(n[0], "  ", n[1])

f.write('\n\nDisk: \n')
f.write('N_GCE = ')
f.write(str(f_obs / r / d[0]))
f.write('\n')
f.write('N_r = ')
f.write(str(f_obs / r / d[0]*n[0]))

#****************************************************************************
#PLOTS

#lum func plot

fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.loglog(l,p_pl1, color = 'lightblue', label = 'Wavelet 1')
ax1.loglog(l, p_pl2, color = 'green', label = 'Wavelet 2 ')
ax1.loglog(l, p_glc, color = 'darkorange', linestyle = ls1, label= 'GLC')
ax1.loglog(l, p_gce, color = 'fuchsia', linestyle = ls1, label = 'GCE')
ax1.loglog(l, p_aic, color = 'red' , linestyle = ls1 , label = 'AIC')
ax1.loglog(l, p_disk, color = 'gray' , linestyle = ls2, label = 'Disk')
ax1.loglog(l, p_nptf, color = 'black', linestyle = ls2,  label = 'NPTF')
ax2.loglog(f_l, p_pl1, color = 'lightblue')

#graphical config
ax1.set_yticks(np.geomspace(1.0e-45, 1.e-29, 5))
ax1.set_xticks(np.geomspace(1.0e30, 1.0e38, 5))
ax2.set_xticks(np.geomspace(1.0e-15, 1.0e-9, 4))

ax1.set_xlabel("Luminosity L [erg / s]")
ax2.set_xlabel(r'Flux F [erg / $\mathregular{cm^2}$ / s]') 
ax1.set_ylabel(r'dN / dL')

plt.ylim(1.0e-45 , 1.2e-27)
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
ax4.loglog(f_l, l * p_pl1, color = 'lightblue')

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
ax6.loglog(f_l, l*l* p_pl1, color = 'lightblue')

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

#----------------------------------------------------------------------------
f.close()
end_time = time.monotonic()
print(timedelta(seconds= end_time - start_time))