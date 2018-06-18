# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "/home/arthur/Dropbox/python")
from cpartition import x2wp

import glob

matplotlib.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size': 16})

basename = 'mart_031C_400'
t_set=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
xlim=(-.188/2, .05/2)
ylim=(0, 3.5)
# figsize=(5.5, 4.5)

for t in t_set:
    try:
        fname = 'C_profiles/' + basename + '_t=' + str(t) + 's.txt'
        z, c, ph = np.loadtxt(fname).T
        plt.plot(z, x2wp(c), label=str(t) + ' s')
    except:
        pass

plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(u'Position (µm)')
plt.ylabel('Carbon content (wt.%)')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(basename + '_Cprofiles.png', dpi=300)
plt.show()
plt.close()
