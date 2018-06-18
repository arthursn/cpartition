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

"""Growth of bainitic ferrite"""

basename = os.path.basename(__file__).replace('.py', '')
wc0 = 0.25e-2
c0 = w2x(wc0)
T_C = 350.

n_time = 20000
total_time = 10.
dt = total_time/n_time
t = (np.arange(n_time) + 1)*dt
aust = FCC(T_C=T_C, dt=dt, z=np.linspace(0, .03, 200), c0=c0, n_time=n_time, tdata='../thermo/Fe-C/350-FCC.TXT')
ferr = BCC(T_C=T_C, dt=dt, z=np.linspace(.03, .03, 10), c0=0., n_time=n_time, tdata='../thermo/Fe-C/350-BCC.TXT', E=WBs(T_C))
int1 = Interface(domain1=aust, domain2=ferr, type_int='mobile.mmode')
ferr.c[:] = int1.CCE(c0)

aust_final = np.zeros(n_time)
aust_untransf = np.zeros(n_time)
f_log = open(basename + '.log', 'w')
string = log_header(f_log, c0=c0, T_C=T_C, n_time=n_time, total_time=total_time, domains={'aust': aust, 'ferr': ferr}, interfaces={'int1': int1})
print(string)
f_log.write(string + '\n')
f_log.flush()

t_start = time.time()
for i in range(n_time):
    # Calculate compositions at the interfaces
    int1.v = 1e6*int1.chem_driving_force()*int1.M()/ferr.Vm
    int1.stefan_local_equilibrium(poly_deg=3)

    J_fcc1 = int1.v*(int1.ci_fcc - int1.ci_bcc)

    #Solve FDM in each domain
    aust.FDM_implicit(bcn=(1.5,-2.,.5,-J_fcc1*aust.dz/aust.D(C=aust.c[-1])))
    ferr.c[:] = int1.ci_bcc

    #Update position of interfaces and interpolate compositions
    aust.update_grid(vn=int1.v)
    ferr.update_grid(v0=int1.v)
    
    aust_untransf[i] = aust.L/0.03
    aust_final[i] = aust.fraction_Troom(aust_untransf[i])
    if i > 0:
        pwr = int(np.log10(i))
        if (i+1) == 2*10**pwr or (i+1) == 6*10**pwr or (i+1) == 3000:
            z, c, cavg, strct = merge_domains((aust, ferr), return_structure=True)
            
            string = '%5d: t=%.2f, r_fcc=%.2e, cavg*=%f, t_elapsed=%.2fs' % (i+1, t[i], np.max(aust.r()), cavg/c0, time.time()-t_start)
            print(string)
            f_log.write(string + '\n')
            f_log.flush()

            lab = label(t[i])
            np.savetxt(fname=os.path.join('C_profiles', basename+'_t='+lab.replace(' ', '')+'.txt'), X=list(zip(z,c,strct)), fmt='%.6e')

string = 'Time elapsed: %.2f s' % (time.time() - t_start)
print(string)
f_log.write(string + '\n\n')
f_log.close()

np.savetxt(fname=os.path.join('C_avg', basename+'.txt'), X=list(zip(t, aust.cavg, ferr.cavg)), fmt='%.6e', header='t aust ferr')
np.savetxt(fname=os.path.join('pos_extremities', basename+'.txt'), X=list(zip(t, aust.s0, aust.sn, ferr.s0, ferr.sn)), fmt='%.6e', header='t aust.s0 aust.sn ferr.s0 ferr.sn')
np.savetxt(fname=os.path.join('C_extremities', basename+'.txt'), X=list(zip(t, aust.ci0, aust.cin, ferr.ci0, ferr.cin)), fmt='%.6e', header='t aust.ci0 aust.cin ferr.ci0 ferr.cin')
np.savetxt(fname=os.path.join('final_aust', basename+'.txt'), X=list(zip(t, aust_final, aust_untransf)), header='t aust.final aust.untransf')

