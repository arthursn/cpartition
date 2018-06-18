import numpy as np
import socket
import time

from .conversion import x2w, w2x, x2wp
from .cpartition import BCC, FCC, Interface


def label(t):
    """
    Formats time in a label format with 1 significant figure
    """
    if t < 1.:
        lab = '{:.1g} s'.format(t)
    else:
        lab = '{:.0f} s'.format(t)
    return lab


class SimulationBox(object):
    def __init__(self, t, *args):
        self.t = t

        if len(args) == 1:
            elements = args[0]
        elif len(args) == 2:
            if isinstance(args[-1][0], str):
                args = args[::-1]
            elements = zip(*args)

        self.names = []
        self.elements = []
        self.typeobj = []
        for k, el in elements:
            if isinstance(el, (BCC, FCC, Interface)):
                self.names.append(k)
                self.elements.append(el)
                self.typeobj.append(type(el))

    def concat_properties(self, prop):
        X = []
        header = []
        for name, el, to in zip(self.names, self.elements, self.typeobj):
            if to is not Interface:
                if prop == 'cavg':
                    X += [el.cavg]
                    header += [name]
                elif prop == 's*':
                    X += [el.s0, el.sn]
                    header += [name+'.s0', name+'.sn']
                elif prop == 'ci*':
                    X += [el.ci0, el.cin]
                    header += [name+'.ci0', name+'.cin']

        X = [self.t] + X
        header = ' '.join(['t'] + header)
        return list(zip(*X)), header

    def save_properties(self, fname, prop, **kwargs):
        X, header = self.concat_properties(prop)
        if len(X) > 1:
            fmt = kwargs.pop('fmt', '%.6e')
            comments = kwargs.pop('comments', '')
            np.savetxt(fname=fname, X=X, fmt=fmt, header=header,
                       comments=comments, **kwargs)
        else:
            print('Nothing to save')


def log_header(f_log, **kwargs):
    """
    Generates header for the log file
    """
    string = f_log.name + ' @ ' + socket.gethostname() + '\n'
    string += time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()) + '\n\n'

    c0 = kwargs.get('c0', 0)
    string += 'c0 = {:.3e} ({:.2f} wt.%)\n'.format(c0, 100*x2w(c0))
    string += 'T_C = {:.1f} [oC]\n'.format(kwargs.get('T_C', 0))
    string += 'total_time = {:.1f} [s]; n_time = {:.0f}\n\n'.format(
        kwargs.get('total_time', 0), kwargs.get('n_time', 0))

    domains = kwargs.get('domains', {})
    for name, dom in list(domains.items()):
        string += '{}: structure = {}; E = {} [J/mol]'.format(
            name, dom.structure, dom.E)
        if dom.structure == 'fcc':
            string += '; type_D = {}'.format(dom.lcode[dom.type_D])
        string += '\n'
        string += '      thermodynamical data: {}\n'.format(dom.tdata)
        string += '      z0 = {:.2f} [um]; zn = {:.2f} [um]; n = {:d}\n'.format(
            dom.z0, dom.zn, dom.n)

    interfaces = kwargs.get('interfaces', {})
    for name, intr in list(interfaces.items()):
        string += '{}: type_int = {}\n'.format(
            name, intr.lcode[intr.type_int])

    return string


def save_cprofiles(basename, zz, cc, n_time, dt, each):
    gridsizes = [len(z) for z in zz]
    gridsize = max(gridsizes)

    for i in range(len(zz)):
        inc = gridsize - len(zz[i])
        zz[i] = np.pad(zz[i], (0, inc), 'edge')
        cc[i] = np.pad(cc[i], (0, inc), 'edge')

    zz = np.array(zz).ravel()
    cc = np.array(cc).ravel()

    np.savetxt(fname=basename + '_profiles.txt', X=list(zip(zz, cc)),
               header=('# n={}\n'
                       '# ti={} tf={} tstep={} ntime={}\n'
                       'z c').format(gridsize, dt*each, dt*n_time, dt*each, n_time//each),
               fmt='%.6e', comments='')
