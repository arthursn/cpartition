#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import os
from itertools import cycle
from cpartition import *

markers = [u'o', u's', u'v', u'8', u'+', u'*', u'h', u'd']
markercycler = cycle(markers)

def removekey(d, key):
    try:
        del d[key]
    except:
        pass

def plot_comparison(files, cols=(0,1), **kwargs):
    tup = True
    try:
        cols[0][0]
        if len(cols) != len(files):
            raise NameError('lenDiffer')
    except NameError:
        print 'Lengths of "cols" and "files" differ.'
        raise
    except:
        tup = False
        col = cols

    func = kwargs.get('func', lambda x: x)
    removekey(kwargs, 'func')

    funcApply = kwargs.get('funcApply', [True]*len(files))
    removekey(kwargs, 'funcApply')
    if len(files) != len(funcApply):
        print 'Lengths of "files" and "funcApply" differ.'
        raise

    labels = kwargs.get('labels', [None]*len(files))
    removekey(kwargs, 'labels')

    colors = kwargs.get('colors', None)
    removekey(kwargs, 'colors')

    markers = kwargs.get('markers', None)
    removekey(kwargs, 'markers')

    mirror = kwargs.get('mirror', False)
    removekey(kwargs, 'mirror')

    if colors != None:
        colorcycler = cycle(colors)
    if markers != None:
        markercycler = cycle(markers)

    for i in xrange(len(files)):
        if tup == True:
            col = cols[i]

        data = np.loadtxt(files[i], usecols=col)
        x = data[:,0]
        y = data[:,1]

        if mirror == True:
            offset = 2*x[-1]
            x = np.hstack([x, offset-x[::-1]])
            y = np.hstack([y, y[::-1]])
        if funcApply[i] == True:
            y = func(y)
        try:
            if colors != None:
                kwargs['color'] = next(colorcycler)
            if markers != None:
                kwargs['marker'] = next(markercycler)
            plt.plot(x, y, label=labels[i], **kwargs)
        except:
            print 'Unexpected error while plotting data. Check if "label" variable is ok.'

def plot_profiles(basename, t_set, **kwargs):
    only_fcc = kwargs.get('only_fcc', False)
    removekey(kwargs, 'only_fcc')

    mirror = kwargs.get('mirror', False)
    removekey(kwargs, 'mirror')

    cint = 0
    # colors = np.linspace(.7, 0, len(t_set))
    # colorcycler = cycle(colors)
    for t in t_set:
        lab = label(t)
        filename = os.path.join('C_profiles', basename + '_t=' + lab.replace(' ', '') + '.txt')

        try:
            print filename
            data = np.loadtxt(filename)
            z, c = data[:,0], data[:,1]
            if mirror == True:
                offset = 2*z[-1]
                z = np.hstack([z, offset-z[::-1]])
                c = np.hstack([c, c[::-1]])
            if data.shape[1] == 3 and only_fcc == True:
                p = data[:,2]
                sel = (p == 1)
                z, c = z[sel], c[sel]

            plt.plot(z, x2wp(c), label=lab, **kwargs)
            # plt.plot(z, x2wp(c), label=lab, color='k', marker=next(markercycler), mfc='none', ms=4, lw=0.5, **kwargs)
        except:
            print 'No such file "%s"' % filename
            pass

        m = np.argmax(c)
        if c[m] > cint:
            zint, cint = z[m], c[m]

    return (zint, x2wp(cint))

x2wp = lambda x: 100*x2w(x)

