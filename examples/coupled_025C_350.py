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

"""Coupled martensite (normal CCE) bainitic ferrite (local equilibrium)"""

basename = os.path.basename(__file__).replace('.py', '')
c0 = w2x(.25e-2)
T_C = 350.

n_time = 20000
total_time = 1.
dt = total_time/n_time
t = (np.arange(n_time) + 1)*dt
mart = BCC(T_C=T_C, dt=dt, z0=-.10, zn=0, n=30, c0=c0, n_time=n_time, tdata='../thermo/Fe-C/350-BCC.TXT')
aust = FCC(T_C=T_C, dt=dt, z0=0, zn=.03, n=200, c0=c0, n_time=n_time, tdata='../thermo/Fe-C/350-FCC.TXT')
ferr = BCC(T_C=T_C, dt=dt, z0=.03, zn=.03, n=10, n_time=n_time, c0=0., tdata='../thermo/Fe-C/350-BCC.TXT', E=WBs(T_C))
int1 = Interface(domain1=mart, domain2=aust, type_int=1)
int2 = Interface(domain1=aust, domain2=ferr, type_int=3)

ferr.c[:] = int2.CCE(aust.c[-1])

aust_final = np.zeros(n_time)
aust_untransf = np.zeros(n_time)

f_log = open(basename + '.log', 'w')
string = log_header(f_log, c0=c0, T_C=T_C, n_time=n_time, total_time=total_time, domains={'mart': mart, 'aust': aust, 'ferr': ferr}, interfaces={'int1': int1, 'int2': int2})
print(string)
f_log.write(string + '\n')
f_log.flush()

t_start = time.time()
for i in range(n_time):
    # Calculate compositions at the interfaces
    int1.stefan_local_equilibrium(poly_deg=3)
    int2.v = 1e6*int2.chem_driving_force()*int2.M()/ferr.Vm
    int2.stefan_local_equilibrium(poly_deg=3)
    
    # FDM
    mart.FDM_implicit(bc0=(-1.5,2.,-.5,0.), bcn=(1.,0.,0.,int1.ci_bcc))
    aust.FDM_implicit(bc0=(1.,0.,0.,int1.ci_fcc), bcn=(1.,0.,0.,int2.ci_fcc))
    ferr.c[:] = int2.ci_bcc

    # Update position of interfaces and interpolate compositions
    mart.update_grid()
    aust.update_grid(vn=int2.v)
    ferr.update_grid(v0=int2.v)
    
    aust_untransf[i] = aust.L/0.13
    aust_final[i] = aust.fraction_Troom(aust_untransf[i])

    if i > 0:
        pwr = int(np.log10(i))
        if (i+1) == 2*10**pwr or (i+1) == 6*10**pwr:
            z, c, cavg, strct = merge_domains((mart, aust, ferr), return_structure=True)
            
            string = '%5d: t=%.2f, r_bcc=%.2e, r_fcc=%.2e, cavg*=%f, t_elapsed=%.2fs' % (i+1, t[i], np.max(mart.r()), np.max(aust.r()), cavg/c0, time.time()-t_start)
            print(string)
            f_log.write(string + '\n')
            f_log.flush()

            lab = label(t[i])
            np.savetxt(fname=os.path.join('C_profiles', basename+'_t='+lab.replace(' ', '')+'.txt'), X=list(zip(z,c,strct)), fmt='%.6e')

string = 'Time elapsed: %.2f s' % (time.time() - t_start)
print(string)
f_log.write(string + '\n\n')
f_log.close()

np.savetxt(fname=os.path.join('C_avg', basename+'.txt'), X=list(zip(t, mart.cavg, aust.cavg, ferr.cavg)), fmt='%.6e', header='t mart aust ferr')
np.savetxt(fname=os.path.join('pos_extremities', basename+'.txt'), X=list(zip(t, mart.s0, mart.sn, aust.s0, aust.sn, ferr.s0, ferr.sn)), fmt='%.6e', header='t mart.s0 mart.sn aust.s0 aust.sn ferr.s0 ferr.sn')
np.savetxt(fname=os.path.join('C_extremities', basename+'.txt'), X=list(zip(t, mart.ci0, mart.cin, aust.ci0, aust.cin, ferr.ci0, ferr.cin)), fmt='%.6e', header='t mart.ci0 mart.cin aust.ci0 aust.cin ferr.ci0 ferr.cin')
np.savetxt(fname=os.path.join('final_aust', basename+'.txt'), X=list(zip(t, aust_final, aust_untransf)), header='t aust.final aust.untransf')

