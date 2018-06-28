import numpy as np
import time

import os
import sys
from cpartition import *
from scipy.interpolate import interp1d

basename = os.path.basename(__file__).replace('.py', '')

c0 = 3.34414e-02
T_C = 375.

control_itsteps = ControlIterationSteps([5e-5, 5e-4, 5e-3], [0, 2, 100, 1000])
total_time = control_itsteps.total_time
n_time = control_itsteps.ntime
dt = control_itsteps.dt
each = 200
control_itsteps.print_summary()

tdata_fcc = '../thermo/FoFo/375-FCC.TXT'
tdata_bcc = '../thermo/FoFo/375-BCC.TXT'

mart = BCC(T_C=T_C, dt=dt, z=np.linspace(-1.16, -.66, 50), c0=c0,
           tdata=tdata_bcc)
aust = FCC(T_C=T_C, dt=dt, z=np.linspace(-.66, 0, 200), c0=c0,
           tdata=tdata_fcc)

intf = Interface(domain1=mart, domain2=aust, type_int='fixed.balance')

log = SimulationLog(basename)
log.set_domains([('mart', mart), ('aust', aust)])
log.set_interfaces([('intf', intf)])
log.set_conditions(c0, T_C, total_time, n_time)
log.initialize(False)

for i in control_itsteps.itlist:
    if i in control_itsteps.itstepi and i > 0:
        control_itsteps.next_itstep()
        dt = control_itsteps.dt

        mart.dt = dt
        aust.dt = dt

    # intf.comp(poly_deg=2)
    intf.comp(c0)
    mart.FDM_implicit(bcn=(1., 0, 0, intf.ci_bcc))
    aust.FDM_implicit(bc0=(1., 0, 0, intf.ci_fcc))

    mart.update_grid(i)
    aust.update_grid(i)

    log.printit(i, each)

log.close()

log.save_cprofiles()
log.save_properties('cavg')
log.save_properties('ci*')
log.save_properties('s*')
