import sys
import numpy as np
from itertools import cycle

__all__ = ['IterationStep', 'ControlIterationSteps']


class IterationStep(object):
    def __init__(self, dt, ti, tf, iti=0):
        self.dt = dt  # delta t
        self.ti = ti  # initial time
        self.tf = tf  # final time
        self.iti = iti

        self.ntime = int((self.tf - self.ti)/self.dt)
        self.t = self.ti + (np.arange(self.ntime) + 1)*self.dt
        self.tf = self.t[-1]

    @property
    def _itloc(self):
        return list(range(self.ntime))

    @property
    def itloc(self):
        return np.array(self._itloc)

    @property
    def _itglo(self):
        return [it + self.iti for it in self._itloc]

    @property
    def itglo(self):
        return np.array(self._itglo)


class ControlIterationSteps(object):
    def __init__(self, dtlist=[1e-1], tbreak=[0, 1]):
        self.dtlist = dtlist
        self.tbreak = tbreak

        self.get_intervals()
        self.initialize_itsteps()

        self.cy_itsteps = cycle(self.itsteps)
        self.curr_itstep = next(self.cy_itsteps)

    def get_intervals(self):
        # convert tbreak to timeintervals
        tintervals = []
        for i in range(len(self.tbreak) - 1):
            tintervals += [[self.tbreak[i], self.tbreak[i+1]]]

        if len(tintervals) == len(self.dtlist):
            self.tintervals = tintervals
        else:
            raise Exception('Lengths of dtlist and tbreak differ')

    def initialize_itsteps(self):
        # initialize IterationStep objects
        self.itsteps = []

        idxstep = []
        iti, itlist = 0, []
        itstepi, itstepf = [], []
        ti, tlist = self.tintervals[0][0], []

        for i, (dt, tint) in enumerate(zip(self.dtlist, self.tintervals)):
            # redefine tint[0] because of non homogeneous grids
            tint[0] = ti

            # initialize IterationStep
            itstep = IterationStep(dt, tint[0], tint[1], iti)
            # append itstep to list of itsteps
            self.itsteps.append(itstep)

            # iti is the iteration number at the beginning of
            # the new step
            iti += itstep.ntime
            ti = itstep.tf

            itlist += itstep._itglo
            itstepi.append(itstep.itglo[0])
            itstepf.append(itstep.itglo[-1])

            tlist.append(itstep.t)
            idxstep += [i]*itstep.ntime

        self.tlist = np.hstack(tlist)
        self.itlist = np.array(itlist)
        self.itstepi = itstepi
        self.itstepf = itstepf
        self.idxstep = idxstep

    def next_itstep(self):
        self.curr_itstep = next(self.cy_itsteps)

    @property
    def total_time(self):
        return self.tlist[-1]

    @property
    def ntime(self):
        return len(self.tlist)

    @property
    def dt(self):
        return self.curr_itstep.dt

    def which_itstep(self, it):
        idx = self.idxstep[it]
        return self.itsteps[idx]

    def print_summary(self, fstream=None):
        if fstream is None:
            fstream = sys.stdout

        for itstep in self.itsteps:
            fstream.write(('dt = {}, ti = {}, tf = {}, iti = {}, itf = {}\n').format(
                itstep.dt, itstep.ti, itstep.tf, itstep.itglo[0], itstep.itglo[-1]))
