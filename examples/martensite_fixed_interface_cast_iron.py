import numpy as np
import os
from cpartition import (BCC, FCC, Interface,
                        ControlIterationSteps, SimulationLog)

basename = os.path.basename(__file__).replace('.py', '')

# Initial composition in at. fraction
c0 = 3.34414e-02
# Temperature in degrees C
T_C = 375.

control_itsteps = ControlIterationSteps([5e-5, 5e-4, 5e-3], [0, 2, 100, 1000])
total_time = control_itsteps.total_time
n_time = control_itsteps.ntime
dt = control_itsteps.dt
each = 200
control_itsteps.print_summary()

# Files containing the chemical potentials of bcc and fcc phases
tdata_fcc = os.path.join('..', 'thermo', 'cast_iron', '375-FCC.TXT')
tdata_bcc = os.path.join('..', 'thermo', 'cast_iron', '375-BCC.TXT')

# Instantiate BCC and FCC classes as mart and aust objects
# T_C: temperature; dt: Time step; z: positions of the nodes;
# c0: initial composition; tdata: location of file containing
# thermodynamical data (chemical potentials)
mart = BCC(T_C=T_C, dt=dt, z=np.linspace(-1.16, -.66, 50), c0=c0,
           tdata=tdata_bcc)
aust = FCC(T_C=T_C, dt=dt, z=np.linspace(-.66, 0, 200), c0=c0,
           tdata=tdata_fcc)

# Interface
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

    # Calculates interfacial compositions
    intf.comp(c0)
    # FDM iteration step
    mart.FDM_implicit(bcn=(1., 0, 0, intf.ci_bcc))
    aust.FDM_implicit(bc0=(1., 0, 0, intf.ci_fcc))

    # Update grid information
    mart.update_grid(i)
    aust.update_grid(i)

    # Print info relative to the iteration step
    log.printit(i, each)

log.close()

log.save_cprofiles()
log.save_properties('cavg')
log.save_properties('ci*')
log.save_properties('s*')
