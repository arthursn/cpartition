import os
import numpy as np
import pandas as pd
import socket
import time
import matplotlib.pyplot as plt

from .conversion import x2wp

__all__ = ['SimulationLog', 'CProfiles']


class SimulationLog(object):
    """
    Simulation log. Used for exporting simulation data.
    """

    def __init__(self, basename):
        self.basename = basename.strip('.py')

    def set_domains(self, domains):
        self.domains = domains

    def update_domains(self, domains):
        # Forgot what I was going to implement.
        # So far it's ok the way it is
        pass

    def set_interfaces(self, interfaces):
        self.interfaces = interfaces

    def update_interfaces(self, interfaces):
        # Forgot what I was going to implement.
        # So far it's ok the way it is
        pass

    def set_conditions(self, c0, T_C, total_time, n_time, reset=True):
        """Set conditions"""
        self.c0 = c0
        self.T_C = T_C
        self.total_time = total_time
        self.n_time = n_time
        self.dt = total_time/n_time

        # Uset reset=False if you don't want to erase the data
        if reset:
            # list of floats:
            self.t = []  # time
            self.cavg = []  # average composition
            # list of numpy.ndarray objects:
            self.zz = []  # list of positions z
            self.cc = []  # list of compositions c
            self.ss = []  # list of structures strct

    def set_flush(self, flush):
        if isinstance(flush, bool):
            self.flush = flush  # if True, flush the data from memory to the file stream
        else:
            raise Exception('flush is not a boolean')

    def merge_domains(self):
        """
        Merge domains (FCC and/or BCC objects)

        Returns
        -------
        tzcas : tuple
            (t, z, c, strct, cavg) a tuple containing the arrays for the time,
            node positions z, the node compositions c, the average composition
            cavg, and the structure in each node strct.
        """
        t, z, c, L, cavg, strct = [], [], [], [], [], []

        for name, dom in self.domains:
            if dom.active:
                t.append(dom.t[-1])
                z.append(dom.z)
                c.append(dom.c)
                strct.append([name]*dom.n)
                L.append(dom.L)
                cavg.append(dom.get_cavg())

        t = max(t)
        cavg = np.average(cavg, weights=L)

        z = np.hstack(z)
        c = np.hstack(c)
        strct = np.hstack(strct)

        return t, z, c, strct, cavg

    def initialize(self, flush=True, mode='w'):
        """
        Generates header for the log file
        """
        fname = self.basename + '.log'
        self.f_log = open(fname, mode)

        self.set_flush(flush)

        self.t_start = time.time()

        # printing hostname
        string = self.f_log.name + ' @ ' + socket.gethostname() + '\n'
        string += time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()) + '\n\n'

        # printing conditions (c0, temperature, time)
        string += 'c0 = {:e} (approx {:.2f} wt.%)\n'.format(self.c0,
                                                            x2wp(self.c0))
        string += 'T_C = {:.1f} [oC]\n'.format(self.T_C)
        string += 'total_time = {:.1f} [s]; n_time = {:.0f}\n\n'.format(
            self.total_time, self.n_time)

        # printing details of the simulation domains
        for name, dom in self.domains:
            if dom.active:
                string += '{}: structure = {}; E = {} [J/mol]'.format(
                    name, dom.structure, dom.E)
                string += '; type_D = {}'.format(dom.lcode[dom.type_D])
                string += '\n'
                string += '      thermodynamical data: {}\n'.format(dom.tdata)
                string += '      z0 = {:.2f} um; zn = {:.2f} um; n = {:d}\n'.format(
                    dom.z0, dom.zn, dom.n)

        # print details of the interfaces
        for name, intf in self.interfaces:
            if intf.active:
                string += '{}: type_int = {}\n'.format(
                    name, intf.lcode[intf.type_int])

        print(string)
        self.f_log.write(string + '\n')
        if self.flush:
            self.f_log.flush()

    def printit(self, it, *args, **kwargs):
        """
        Print details of the current iteration during the simulation.
        Use the argument 'criteria' to set the criteria for printing such
        data.
        """
        criteria = kwargs.pop('criteria', lambda it, each: (it+1) % each == 0)

        if criteria(it, *args):
            t, z, c, strct, cavg = self.merge_domains()

            self.t.append(t)
            self.cavg.append(cavg)
            self.zz.append(z)
            self.cc.append(c)
            self.ss.append(strct)

            string = '{:5d}: t={:.3f}, cavg*={:g}'.format(
                it+1, t, cavg/self.c0)
            for name, dom in self.domains:
                if dom.active:
                    string += ', r_{:}={:g}'.format(name, dom.r.max())

            print(string)
            self.f_log.write(string + '\n')
            if self.flush:
                self.f_log.flush()

    def close(self):
        """
        Close file stream
        """
        string = 'Time elapsed: {:.2f} s'.format(time.time() - self.t_start)
        print(string)
        self.f_log.write(string + '\n\n')
        self.f_log.close()

    def save_cprofiles(self):
        """
        Export carbon profiles to ascii txt files.
        Two files are generated:
        - <directory (C_profiles)>/<basename>_profiles.txt
        - <directory (C_profiles)>/<basename>_time.txt
        """
        gridsizes = [len(z) for z in self.zz]
        gridsize = max(gridsizes)

        zz = []
        cc = []
        ss = []

        # gridsizes may vary from iteration to iteration
        # the code below normalizes the gridsizes by
        # expanding the shorter ones
        for i in range(len(self.zz)):
            zz += list(self.zz[i])
            cc += list(self.cc[i])
            ss += list(self.ss[i])

            inc = gridsize - len(self.zz[i])
            if inc > 0:
                zz += [zz[-1]]*inc
                cc += [cc[-1]]*inc
                ss += [ss[-1]]*inc

        # pandas dataframes for time and carbon profiles
        df_time = pd.DataFrame(dict(t=self.t, cavg=self.cavg),
                               columns=['t', 'cavg'])
        df_profiles = pd.DataFrame(dict(z=zz, c=cc, strct=ss),
                                   columns=['z', 'c', 'strct'])

        # check if C_profiles dir exist. If not, creates one
        directory = 'C_profiles'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # saves dataframe time
        try:
            fname = os.path.join(
                'C_profiles', '{}_time.txt'.format(self.basename))
            df_time.to_csv(fname, index=False, sep=' ', float_format='%.6e')
        except Exception:
            raise
        else:
            print('File "{}" successfully created'.format(fname))

        # saves dataframe carbon profiles
        try:
            fname = os.path.join(
                'C_profiles', '{}_profiles.txt'.format(self.basename))
            header = '# n={:d}\n'.format(gridsize)

            fileout = open(fname, 'w')
            fileout.write(header)
            fileout.close()
            df_profiles.to_csv(fname, index=False, sep=' ',
                               float_format='%.6e', mode='a')
        except Exception:
            raise
        else:
            print('File "{}" successfully created'.format(fname))

    def save_properties(self, prop, fname=None, **kwargs):
        """
        Export simulation tracked properties to txt ascii file.
        The available tracked properties are:
          - cavg: average composition in each phase
          - ci*: interfacial compositions
          - s*: and interface positions
        """
        # if prop is in the form 'ci*' (interfacial comp)
        # or 's*' (interface position)
        if prop[-1] == '*':
            cols = [prop[:-1]+'0', prop[:-1]+'n']
        else:
            cols = [prop]

        t, columns = [], []
        # collect properties (as dataframe) for t and the selected
        # property prop (cavg, ci*, s*)
        for name, dom in self.domains:
            columns.append(dom.dataframe(cols=cols, prefix=name))
            t.append(dom.dataframe(cols=['t']))

        # merge time arrays and drop duplicates
        t = pd.concat(t).drop_duplicates()
        # concat properties dataframes
        columns = pd.concat(columns, axis=1)
        # merge t and columns into a dataframe df
        # (this is the one it will exported)
        df = t.join(columns)

        # if fname no provided, save to a standard file name in
        # the directory C_extremities, pos_extremities, and C_avg
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
                df.to_csv(fname, index=False, sep=' ',
                          float_format='%.6e', na_rep='nan')
            except Exception:
                raise
            else:
                print('File "{}" successfully created'.format(fname))
        else:
            print('Nothing to save')


class CProfiles(object):
    """
    Class for loading and plotting the carbon profiles
    """

    def __init__(self, basename, directory='C_profiles'):
        self.basename = basename.strip('.py')
        self.fname_time = os.path.join(
            directory, '{}_time.txt'.format(self.basename))
        self.fname_profiles = os.path.join(
            directory, '{}_profiles.txt'.format(self.basename))

        self.header = {}
        self.n = None
        self.ntime = None
        self.ti = None
        self.tf = None
        self.tstep = None

        self.t = []  # time as 1d array
        self.tt = []  # time as 2d array

        # this will eventually become a pandas dataframe for carbon profiles
        self.df_cprofiles = []
        self.zz = []  # 2d array for position z
        self.cc = []  # 2d array for composition c
        self.ss = []  # 2d array for structure strct

        # these will eventually become dataframes for interfacial composition,
        # interface position, average composition
        self.df_ci = []
        self.df_si = []
        self.df_cavg = []

    def read_header(self):
        """
        Read <basename>_profiles.txt header
        """
        self.header = {}

        f = open(self.fname_profiles, 'r')

        # read n (gridsize)
        line = f.readline()
        line = line.strip('# ').split()
        self.header.update(dict(x.split('=') for x in line))

        # read time parameters (optional)
        line = f.readline()
        if line[0] == '#':
            line = line.strip('# ').split()
            self.header.update(dict(x.split('=') for x in line))

        f.close()

        self.n = int(self.header['n'])
        try:
            self.ntime = int(self.header['ntime'])
            self.ti = float(self.header['ti'])
            self.tf = float(self.header['tf'])
            self.tstep = float(self.header['tstep'])
        except Exception:
            self.ntime = None
            self.ti = None
            self.tf = None
            self.tstep = None

    def load_time(self):
        """
        Load time data
        """
        if len(self.header) == 0:
            self.read_header()

        try:
            df = pd.read_csv(self.fname_time, sep=' ', comment='#')
            self.t = df['t'].values  # numpy array for time
        except Exception:
            self.t = np.linspace(self.ti, self.tf, self.ntime)

        _, self.tt = np.meshgrid(np.arange(self.n), self.t)

    def load_cprofiles(self):
        """
        Load carbon profiles data
        """
        if len(self.header) == 0:
            self.read_header()

        self.df_cprofiles = pd.read_csv(
            self.fname_profiles, sep=' ', comment='#')

        self.zz = self.df_cprofiles['z'].values.reshape(-1, self.n)
        self.cc = self.df_cprofiles['c'].values.reshape(-1, self.n)

        try:
            self.ss = self.df_cprofiles['strct'].values.reshape(-1, self.n)
        except Exception:
            pass

    def plot_colormap(self, ax=None, mirror=False, func=lambda x: x, **kwargs):
        """
        Plot carbon profiles data as colormap using matplotlib pcolormesh
        """
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if len(self.tt) == 0:
            self.load_time()

        if len(self.df_cprofiles) == 0:
            self.load_cprofiles()

        ax.pcolormesh(self.zz, self.tt, func(self.cc), **kwargs)
        if mirror:
            ax.pcolormesh(2*self.zz[:, -1].reshape(-1, 1) -
                          self.zz, self.tt, func(self.cc), **kwargs)

        return ax

    def where_tlist(self, tlist, appendto=[]):
        """
        Get indices of matching occurences of tlist in CPartition().t
        """
        # loading profiles if not defined yet
        if len(self.df_cprofiles) == 0:
            self.load_cprofiles()

        # loading time if not defined yet
        if len(self.t) == 0:
            self.load_time()

        # Search for tlist in self.t
        matches, = np.where(np.isin(self.t.astype(np.float32),
                                    np.array(tlist).astype(np.float32),
                                    assume_unique=True))
        appendto += list(matches)

        return appendto

    def label_phases(self, ax, t, labels=[('aus1', r'$\gamma_1$', -1),
                                          ('aus2', r'$\gamma_2$', -1),
                                          ('aust', r'$\gamma$', -1),
                                          ('mart', r"$\alpha'$", -1),
                                          ('fer1', r'$\alpha_{b1}$', 1),
                                          ('fer2', r'$\alpha_{b2}$', 1)],
                     mirror=False, **kwargs):
        """
        Label phases in a C profile plot
        """
        if isinstance(t, list):
            t = t[-1]
        j, = self.where_tlist([t], [])

        zbleft = self.zz[0][0]
        zbright = self.zz[0][-1]
        kwannotate = dict(size=9)
        kwannotate.update(kwargs)
        kwvline = dict(color='k', ls=':', lw='.8')
        zused = []

        for strct, lab, ypos in labels:
            sel = self.ss[j] == strct
            z = self.zz[j][sel]

            if len(z > 0):
                ymin, ymax = ax.get_ylim()
                va = 'bottom' if ypos > 0 else 'top'

                zclist = [.5*(z[0] + z[-1])]
                if mirror:
                    zclist += [2*zbright - zclist[0]]
                    if z[-1] == zbright:
                        zclist = [self.zz[0][-1]]

                for zc in zclist:
                    ax.annotate(s=lab, xy=(zc, ymax), xytext=(0, ypos),
                                textcoords='offset points',
                                ha='center', va=va, **kwannotate)
                ##############

                if z[0] != zbleft and z[0] not in zused:
                    ax.axvline(z[0], **kwvline)
                    if mirror:
                        ax.axvline(2*zbright - z[0], **kwvline)
                    zused.append(z[0])

                if z[-1] != zbright and z[-1] not in zused:
                    ax.axvline(z[-1], **kwvline)
                    if mirror:
                        ax.axvline(2*zbright - z[-1], **kwvline)
                    zused.append(z[-1])

        return ax

    def get_cprofile(self, i, mirror=False, func=lambda x: x):
        """
        Select C profile for a given time index i
        """
        z, c, t = self.zz[i], func(self.cc[i]), self.t[i]

        if mirror:
            z = np.hstack([z, 2*z[-1] - z[::-1]])
            c = np.hstack([c, c[::-1]])

        return z, c, t

    def plot_cprofiles(self, ax=None, mirror=False, func=lambda x: x, **kwargs):
        """
        Plot carbon profiles using matplotlib plot
        """
        sel = kwargs.pop('sel', [])  # list of indices to be plotted
        tlist = kwargs.pop('tlist', [])  # list of times to be plotted

        slc = kwargs.pop('slc', None)  # slice of the indices to be plotted
        # t-range. Passed to slice slc if slc is None
        tmin = kwargs.pop('tmin', None)
        tmax = kwargs.pop('tmax', None)
        each = kwargs.pop('each', None)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # loading profiles if not defined yet
        if len(self.df_cprofiles) == 0:
            self.load_cprofiles()

        # loading time if not defined yet
        if len(self.t) == 0:
            self.load_time()

        # append to sel the indices where t in tlist is also in self.t
        if len(tlist) > 0:  # if tlist is empty
            sel = self.where_tlist(tlist, appendto=sel)

        if slc is None:  # if slc is empty
            slc = slice(tmin, tmax, each)

        # if sel is still empty, then sel are the items
        if len(sel) == 0:
            sel = range(len(self.zz))

        # call plot for each item of the selection
        for i in sel[slc]:
            try:
                z, c, t = self.get_cprofile(i, mirror, func)

                lines = ax.plot(z, c, label='t = {:g} s'.format(t), **kwargs)
            except IndexError:
                print('Index {} is out of bounds'.format(i))
            except Exception:
                print('Unexpected error')
                raise

        return lines

    def plot_locus_interface(self, pairs, ax=None, mirror=False, func=lambda x: x, *args, **kwargs):
        """
        Plot locus ci* (interfacial composition) vs s* (interface position)
        """
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if len(pairs) > 0:
            if len(self.df_ci) == 0:
                fname = os.path.join(
                    'C_extremities', '{}.txt'.format(self.basename))
                self.df_ci = pd.read_csv(fname, sep=' ', comment='#')

            if len(self.df_si) == 0:
                fname = os.path.join(
                    'pos_extremities', '{}.txt'.format(self.basename))
                self.df_si = pd.read_csv(fname, sep=' ', comment='#')

            for xkey, ykey in pairs:
                try:
                    lines = ax.plot(self.df_si[xkey],
                                    func(self.df_ci[ykey]), *args, **kwargs)
                    if mirror:
                        lines = ax.plot(2*self.zz[0][-1] - self.df_si[xkey],
                                        func(self.df_ci[ykey]), *args, **kwargs)
                except KeyError:
                    print('Key error')
                except Exception as ex:
                    print('Unexpected error: {}'.format(ex))
                    raise

        return lines
