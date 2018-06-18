import os
import numpy as np
import pandas as pd
import socket
import time
import matplotlib.pyplot as plt

from .conversion import x2wp
from .cpartition import BCC, FCC, Interface


class SimulationLog(object):
    def __init__(self, basename):
        self.basename = basename

    def set_domains(self, domains):
        self.domains = domains

    def update_domains(self, domains):
        pass

    def set_interfaces(self, interfaces):
        self.interfaces = interfaces

    def update_interfaces(self, interfaces):
        pass

    def set_conditions(self, c0, T_C, total_time, n_time):
        self.c0 = c0
        self.T_C = T_C
        self.total_time = total_time
        self.n_time = n_time
        self.dt = total_time/n_time

        self.t = np.linspace(0, total_time, n_time)
        self.zz = []  # list of positions z
        self.cc = []  # list of compositions c
        self.ss = []  # list of structures strct

    def merge_domains(self):
        """
        Merge domains (FCC and/or BCC objects)

        Returns
        -------
        zcas : tuple
            (z, c, strct, cavg) a tuple containing the arrays for the
            node positions z and the node compositions c, a float for the average
            composition cavg, and the structure in each node strct. strct is only
            returned if return_structure is assigned as True.        
        """

        z = [dom.z for name, dom in self.domains if dom.active]
        c = [dom.c for name, dom in self.domains if dom.active]
        strct = [[name]*dom.n for name, dom in self.domains if dom.active]

        L = [dom.L for name, dom in self.domains if dom.active]
        cavg = [dom.get_cavg() for name, dom in self.domains if dom.active]
        cavg = np.average(cavg, weights=L)

        z = np.hstack(z)
        c = np.hstack(c)
        strct = np.hstack(strct)

        return z, c, strct, cavg

    def initialize(self, each=1, flush=True):
        """
        Generates header for the log file
        """
        fname = self.basename + '.log'
        self.f_log = open(fname, 'w')

        self.set_each(each)
        self.set_flush(flush)

        self.t_start = time.time()

        string = self.f_log.name + ' @ ' + socket.gethostname() + '\n'
        string += time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()) + '\n\n'

        string += 'c0 = {:e} (approx {:.2f} wt.%)\n'.format(self.c0,
                                                            x2wp(self.c0))
        string += 'T_C = {:.1f} [oC]\n'.format(self.T_C)
        string += 'total_time = {:.1f} [s]; n_time = {:.0f}\n\n'.format(
            self.total_time, self.n_time)

        for name, dom in self.domains:
            if dom.active:
                string += '{}: structure = {}; E = {} [J/mol]'.format(
                    name, dom.structure, dom.E)
                string += '; type_D = {}'.format(dom.lcode[dom.type_D])
                string += '\n'
                string += '      thermodynamical data: {}\n'.format(dom.tdata)
                string += '      z0 = {:.2f} um; zn = {:.2f} um; n = {:d}\n'.format(
                    dom.z0, dom.zn, dom.n)

        for name, intf in self.interfaces:
            if intf.active:
                string += '{}: type_int = {}\n'.format(
                    name, intf.lcode[intf.type_int])

        print(string)
        self.f_log.write(string + '\n')
        if self.flush:
            self.f_log.flush()

    def set_each(self, each):
        if isinstance(each, int):
            self.each = each
        else:
            raise Exception('each is not an integer')

    def set_flush(self, flush):
        if isinstance(flush, bool):
            self.flush = flush
        else:
            raise Exception('flush is not a boolean')

    def print(self, it):
        if (it+1) % self.each == 0:
            z, c, strct, cavg = self.merge_domains()

            self.zz.append(z)
            self.cc.append(c)
            self.ss.append(strct)

            string = '{:5d}: t={:.3f}, cavg*={:g}'.format(
                it+1, self.t[it], cavg/self.c0)
            for name, dom in self.domains:
                if dom.active:
                    string += ', r_{:}={:g}'.format(name, dom.r.max())

            print(string)
            self.f_log.write(string + '\n')
            if self.flush:
                self.f_log.flush()

    def close(self):
        string = 'Time elapsed: {:.2f} s'.format(time.time() - self.t_start)
        print(string)
        self.f_log.write(string + '\n\n')
        self.f_log.close()

    def save_cprofiles(self, fname=None):
        gridsizes = [len(z) for z in self.zz]
        gridsize = max(gridsizes)

        zz = []
        cc = []
        ss = []

        for i in range(len(self.zz)):
            zz += list(self.zz[i])
            cc += list(self.cc[i])
            ss += list(self.ss[i])

            inc = gridsize - len(self.zz[i])
            if inc > 0:
                zz += [zz[-1]]*inc
                cc += [cc[-1]]*inc
                ss += [ss[-1]]*inc

        df = pd.DataFrame(dict(z=zz, c=cc, strct=ss),
                          columns=['z', 'c', 'strct'])

        if not fname:
            fname = self.basename + '_profiles.txt'

        n_out = self.n_time//self.each
        ti = self.dt*self.each
        tf = self.dt*n_out*self.each
        tstep = self.dt*self.each

        header = ('# n={:d}\n'
                  '# ti={:g} tf={:g} tstep={:g} ntime={:d}\n').format(gridsize, ti, tf, tstep, n_out)

        try:
            fileout = open(fname, 'w')
            fileout.write(header)
            df.to_csv(fileout, index=False, sep=' ', float_format='%.6e')
            fileout.close()
        except:
            raise
        else:
            print('File "{}" successfully created'.format(fname))

    def save_properties(self, prop, fname=None, **kwargs):
        df = pd.DataFrame(dict(t=self.t))

        for name, dom in self.domains:
            if prop == 'cavg':
                df[name] = dom.cavg
            elif prop == 's*':
                df[name+'.s0'] = dom.s0
                df[name+'.sn'] = dom.sn
            elif prop == 'ci*':
                df[name+'.ci0'] = dom.ci0
                df[name+'.cin'] = dom.cin

        if len(df.columns) > 1:
            if not fname:
                if prop == 'ci*':
                    directory = 'C_extremities'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    fname = os.path.join(directory, self.basename+'.txt')

                elif prop == 's*':
                    directory = 'pos_extremities'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    fname = os.path.join(directory, self.basename+'.txt')

                else:
                    directory = 'C_avg'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    fname = os.path.join(directory, self.basename+'.txt')

            try:
                df.to_csv(fname, index=False, sep=' ', float_format='%.6e')
            except:
                raise
            else:
                print('File "{}" successfully created'.format(fname))
        else:
            print('Nothing to save')


class CProfiles(object):
    def __init__(self, fname):
        self.fname = fname

        self.header = None
        self.n = None
        self.ti = None
        self.tf = None
        self.ntime = None
        self.tstep = None

        self.df = None  # pandas dataframe
        self.zz = None  # 2d array for position z
        self.cc = None  # 2d array for composition c
        self.tt = None  # 2d array for time t
        self.ss = None  # 2d array for structure strct

    def get_header(self):
        self.header = {}

        f = open(self.fname, 'r')

        line = f.readline()
        line = line.strip('# ').split()
        self.header.update(dict(x.split('=') for x in line))

        line = f.readline()
        line = line.strip('# ').split()
        self.header.update(dict(x.split('=') for x in line))

        f.close()

        self.n = int(self.header['n'])
        self.ntime = int(self.header['ntime'])
        self.ti = float(self.header['ti'])
        self.tf = float(self.header['tf'])
        self.tstep = float(self.header['tstep'])

    def get_time(self):
        if not self.header:
            self.get_header()

        _, self.tt = np.meshgrid(np.arange(self.n),
                                 np.linspace(self.ti, self.tf, self.ntime))

    def load_file(self):
        if not self.header:
            self.get_header()

        self.df = pd.read_table(self.fname, sep=' ', comment='#')

        self.zz = self.df['z'].values.reshape(-1, self.n)
        self.cc = self.df['c'].values.reshape(-1, self.n)

        try:
            self.ss = self.df['strct'].values.reshape(-1, self.n)
        except:
            pass

    def plot_colormap(self, ax=None, mirror=False, func=lambda x: x, **kwargs):
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if not self.df:
            self.load_file()

        if not self.tt:
            self.get_time()

        ax.pcolormesh(self.zz, self.tt, func(self.cc), **kwargs)
        if mirror:
            ax.pcolormesh(2*self.zz[:, -1].reshape(-1, 1) -
                          self.zz, self.tt, func(self.cc), **kwargs)

        return ax

    def plot_profiles(self, each, ax=None, mirror=False, func=lambda x: x, **kwargs):
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if not self.df:
            self.load_file()

        if not self.tt:
            self.get_time()

        for i in range(each-1, len(self.zz), each):
            z, c, t = self.zz[i], func(self.cc[i]), self.tstep*(i+1)

            if mirror:
                z = np.hstack([z, 2*z[-1] - z[::-1]])
                c = np.hstack([c, c[::-1]])

            ax.plot(z, c, label='t = {} s'.format(t))

        return ax
