#!/usr/bin/env python

import numpy as np
import time

import sys, os
sys.path.insert(1, "/home/arthur/Dropbox/python")
from cpartition import *

new_dirs = ['C_profiles', 'C_avg', 'pos_extremities', 'C_extremities', 'final_aust']
for directory in new_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

"""Carbon partitioning from martensite to austenite assuming CCE between fcc and bcc"""

basename = os.path.basename(__file__).replace('.py', '')
c0 = w2x(.25e-2)
T_C = 350.

n_time = 20000
total_time = 1.
dt = total_time/n_time
t = (np.arange(n_time) + 1)*dt

mart = BCC(T_C=T_C, dt=dt, z0=-.10, zn=0, n=25, c0=c0, n_time=n_time, tdata='../thermo/Fe-C/350-BCC.TXT')
aust = FCC(T_C=T_C, dt=dt, z0=0, zn=.03, n=250, c0=c0, n_time=n_time, tdata='../thermo/Fe-C/350-FCC.TXT')
int1 = Interface(domain1=mart, domain2=aust, type_int='fixed.fluxes')

from scipy.optimize import newton
import matplotlib.pyplot as plt

# def stefan(xfcc, xfcc1, xbcc1, k):
#     def f(x): return (int1.CCE(x) - xbcc1)*mart.D()/(k*aust.D(x)) + (x - xfcc1)
#     xfcc = newton(func=f, x0=xfcc)
#     xbcc = int1.CCE(xfcc)
#     return xbcc, xfcc

# k_ = 10**np.linspace(-1, 3, 100)
# xbcc = []
# xfcc = []
# for k in k_:
#     xbcc_, xfcc_ = stefan(c0, c0, c0, k)
#     xbcc.append(xbcc_)
#     xfcc.append(xfcc_)
# plt.plot(k_, xbcc)

plt.show()

aust_final = np.zeros(n_time)
aust_untransf = np.zeros(n_time)

f_log = open(basename + '.log', 'w')
string = log_header(f_log, c0=c0, T_C=T_C, n_time=n_time, total_time=total_time, domains={'mart': mart, 'aust': aust}, interfaces={'int1': int1})
print(string)
f_log.write(string + '\n')
f_log.flush()

t_start = time.time()
for i in range(n_time):
    # Calculate compositions at the interfaces
    int1.stefan_local_equilibrium(poly_deg=3)

    # FDM
    mart.FDM_implicit(bc0=(-1.5,2.,-.5,0.), bcn=(1.,0.,0.,int1.ci_bcc))
    aust.FDM_implicit(bc0=(1.,0.,0.,int1.ci_fcc), bcn=(1.5,-2.,.5,0))
    
    # Update position of interfaces and interpolate compositions
    mart.update_grid()
    aust.update_grid()

    aust_untransf[i] = aust.L/0.13
    aust_final[i] = aust.fraction_Troom(aust_untransf[i])

    if i > 0:
        pwr = int(np.log10(i))
        if (i+1) == 2*10**pwr or (i+1) == 6*10**pwr:
            z, c, cavg, strct = merge_domains((mart, aust), True)

            J_bcc, J_fcc = int1.flux(nnodes=3)
            D_bcc, D_fcc = int1.D_interface()
            string = '%6d: t=%.2e, r_bcc=%.2e r_fcc=%.2e, ci_bcc*=%f, ci_fcc*=%f, cavg*=%f, J=%g, t_elapsed=%.2f' % (i+1, t[i], np.max(mart.r()), np.max(aust.r()), mart.c[-1]/c0, aust.c[0]/c0, cavg/c0, np.abs((J_bcc-J_fcc)/J_fcc), time.time()-t_start)
            print(string)
            f_log.write(string + '\n')
            f_log.flush()

            lab = label(t[i])
            np.savetxt(fname=os.path.join('C_profiles', basename+'_t='+lab.replace(' ', '')+'.txt'), X=list(zip(z,c,strct)), fmt='%.6e %.6e %i')

string = 'Time elapsed: %.2f s' % (time.time() - t_start)
print(string)
f_log.write(string + '\n\n')
f_log.close()

np.savetxt(fname=os.path.join('C_avg', basename+'.txt'), X=list(zip(t, mart.cavg, aust.cavg)), fmt='%.6e', header='t mart aust')
np.savetxt(fname=os.path.join('C_extremities', basename+'.txt'), X=list(zip(t, mart.ci0, mart.cin, aust.ci0, aust.cin)), fmt='%.6e', header='t mart.ci0 mart.cin aust.ci0 aust.cin')
np.savetxt(fname=os.path.join('final_aust', basename+'.txt'), X=list(zip(t, aust_final, aust_untransf)), header='t aust.final aust.untransf')
