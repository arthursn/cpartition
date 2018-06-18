#!/usr/bin/env python

import numpy as np
import time

import sys
import os
sys.path.insert(1, "/home/arthur/Dropbox/python")
from cpartition import *

new_dirs = ['C_profiles', 'C_avg',
            'pos_extremities', 'C_extremities', 'final_aus2']
for directory in new_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

"""Coupled model for Fe-0.80C alloy"""

basename = os.path.basename(__file__).replace('.py', '')
wc0 = 0.8e-2
c0 = w2x(wc0)
T_C = 375.

n_time = 20000
total_time = 100.
dt = total_time/n_time
t = (np.arange(n_time) + 1)*dt

mart = BCC(T_C=T_C, dt=dt, z=np.linspace(0, .5, 20), c0=c0,
           n_time=n_time, tdata='../thermo/Fe-C/375-BCC.TXT')
aus1 = FCC(T_C=T_C, dt=dt, z=np.linspace(.5, .75, 100), c0=c0,
           n_time=n_time, tdata='../thermo/Fe-C/375-FCC.TXT')
fer1 = BCC(T_C=T_C, dt=dt, z=np.linspace(.75, .75, 10), c0=0.,
           n_time=n_time, tdata='../thermo/Fe-C/375-BCC.TXT', E=WBs(T_C))
aus2 = FCC(T_C=T_C, dt=dt, z=np.linspace(.75, 1., 100), c0=c0,
           n_time=n_time, tdata='../thermo/Fe-C/375-FCC.TXT')
fer2 = BCC(T_C=T_C, dt=dt, z=np.linspace(1., 1., 10), c0=0.,
           n_time=n_time, tdata='../thermo/Fe-C/375-BCC.TXT'), E=WBs(T_C))

int1 = Interface(domain1=mart, domain2=aus1, type_int='fixed.fluxes')
int2 = Interface(domain1=aus1, domain2=fer1, type_int='mobile.mmode')
int3 = Interface(domain1=fer1, domain2=aus2, type_int='mobile.mmode')
int4 = Interface(domain1=aus2, domain2=fer2, type_int='mobile.mmode')

# fixed composition set by CCEtheta in austenite at the interface

g = lambda x: int1.fcc.f(x) - int1.bcc.f(x)
lo = max(min(int1.fcc.muC), min(int1.bcc.muC))
up = min(max(int1.fcc.muC), max(int1.bcc.muC))
muC = bisect(g, lo, up, xtol=1e-3)
int1.ci_fcc = int1.fcc.mu2x['C'](muC + WBs(T_C))
print(x2wp(int1.ci_fcc))
sys.exit()
fer1.c[:] = int3.CCE(c0)
fer2.c[:] = int4.CCE(c0)

aus2_final = np.zeros(n_time)
aus2_untransf = np.zeros(n_time)

f_log = open(basename + '.log', 'w')
string = log_header(f_log, c0=c0, T_C=T_C, n_time=n_time,
                    total_time=total_time,
                    domains={'aus2': aus2, 'fer2': fer2},
                    interfaces={'int4': int4})
print(string)
f_log.write(string + '\n')
f_log.flush()

t_start = time.time()
for i in range(n_time):
    # interface velocities at the mobile interfaces
    int2.v = 1e6*int2.chem_driving_force()*int2.M()/fer1.Vm
    int2.stefan_local_equilibrium(poly_deg=3)

    int3.v = 1e6*int3.chem_driving_force()*int3.M()/fer1.Vm
    int3.stefan_local_equilibrium(poly_deg=3)

    int4.v = 1e6*int4.chem_driving_force()*int4.M()/fer2.Vm
    int4.stefan_local_equilibrium(poly_deg=3)

    # Solve FDM in each domain
    # mart: martensite 
    J, = int1.flux('fcc')
    mart.FDM_implicit(bcn=(1.5, -2., .5, -J*mart.dz/mart.D()))

    # aus1: first austenite block
    J = int2.v*(int2.ci_fcc - int2.ci_bcc)  # net carbon flux due to interface movement
    aus1.FDM_implicit(bc0=(1, 0, 0, int1.ci_fcc),
                      bcn=(1.5, -2., .5, -J*aus1.dz/aus1.D(C=aus1.c[-1])))
    
    # fer1: first bainitic ferrite plate
    fer1.c[:] = np.linspace(int2.ci_bcc, int3.ci_bcc, fer1.n)

    # aus2: second austenite block
    J0 = int3.v*(int3.ci_bcc - int3.ci_fcc)  # net carbon flux due to interface movement
    Jn = int4.v*(int4.ci_fcc - int4.ci_bcc)  # net carbon flux due to interface movement
    aus2.FDM_implicit(bc0=(1.5, -2., .5, -J0*aus2.dz/aus2.D(C=aus2.c[0])),
                      bcn=(1.5, -2., .5, -Jn*aus2.dz/aus2.D(C=aus2.c[-1])))
    
    # fer2: second bainitic ferrite plate
    fer2.c[:] = int4.ci_bcc

    # Update position of interfaces and interpolate compositions
    aus1.update_grid(vn=int2.v)
    fer1.update_grid(v0=int2.v, vn=int3.v)
    aus2.update_grid(v0=int3.v, vn=int4.v)
    fer2.update_grid(v0=int4.v)

    # Some extra stuff
    aus2_untransf[i] = aus2.L/0.03
    aus2_final[i] = aus2.fraction_Troom(aus2_untransf[i])
    if i > 0:
        pwr = int(np.log10(i))
        if (i+1) == 2*10**pwr or (i+1) == 6*10**pwr or (i+1) == 3000:
            z, c, cavg, strct = merge_domains((mart, aus1, fer1, aus2, fer2),
                                              return_structure=True)

            string = '%5d: t=%.2f, r_fcc=%.2e, cavg*=%f, t_elapsed=%.2fs' % (
                i+1, t[i], np.max(aus2.r()), cavg/c0, time.time()-t_start)
            print(string)
            f_log.write(string + '\n')
            f_log.flush()

            lab = label(t[i])
            np.savetxt(fname=os.path.join('C_profiles', basename+'_t=' + lab.replace(' ', '')+'.txt'),
                       X=list(zip(z, c, strct)), fmt='%.6e')

string = 'Time elapsed: %.2f s' % (time.time() - t_start)
print(string)
f_log.write(string + '\n\n')
f_log.close()

np.savetxt(fname=os.path.join('C_avg', basename+'.txt'),
           X=list(zip(t, aus2.cavg, fer2.cavg)), fmt='%.6e', header='t aus2 fer2')
np.savetxt(fname=os.path.join('pos_extremities', basename+'.txt'),
           X=list(zip(t, aus2.s0, aus2.sn, fer2.s0, fer2.sn)), fmt='%.6e', header='t aus2.s0 aus2.sn fer2.s0 fer2.sn')
np.savetxt(fname=os.path.join('C_extremities', basename+'.txt'),
           X=list(zip(t, aus2.ci0, aus2.cin, fer2.ci0, fer2.cin)), fmt='%.6e', header='t aus2.ci0 aus2.cin fer2.ci0 fer2.cin')
np.savetxt(fname=os.path.join('final_aus2', basename+'.txt'),
           X=list(zip(t, aus2_final, aus2_untransf)), header='t aus2.final aus2.untransf')
