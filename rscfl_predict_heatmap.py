#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import sys
import json
import re
import os
import os.path
import scipy.stats as st
import numpy as np
import pickle
from pylab import *
from functools import partial
from sklearn import gaussian_process

pd.options.display.mpl_style = 'default'
rcParams['figure.figsize'] = 22,10

train_file = open(sys.argv[1], 'rb')
trained_gauss = pickle.load(train_file)
train_file.close()

gp = trained_gauss

hmap = pd.read_pickle(sys.argv[2])

hmap['xen_cyc'] = hmap['xen_cyc'].astype('uint64')
hmap['load'] = hmap['load'].astype(int)
hmap['no_vms'] = hmap['no_vms'].astype(int)

ghmap = hmap.groupby(by=['no_vms', 'load'])

vm_loadset = set([vm for vm, ld in ghmap.groups.keys()])
os_loadset = set([ld for vm, ld in ghmap.groups.keys()])
hmad_data = pd.DataFrame(np.zeros((len(vm_loadset), len(os_loadset))), columns=list(os_loadset))

if len(sys.argv)>4:
    prefix = sys.argv[4]
else:
    prefix = "."

for (vm, ld), gr in ghmap:
    x = np.atleast_2d(gr['xen_cyc'])
    if not os.path.isfile(os.path.abspath(prefix + "/%d.%d.hmap" % (vm, ld))):
        f = open(prefix + "%d.%d.hmap" % (vm, ld), 'w')
        y_pred, MSE = gp.predict(x.T, eval_MSE=True)
        pickle.dump(y_pred, f)
        f.close()
    else:
        f = open(prefix + "%d.%d.hmap" % (vm, ld), 'r')
        y_pred = picle.load(f)
        f.close()
    hmad_data.at[vm-1, ld] = np.percentile(y_pred, 99)

hmap_plt = np.array(hmad_data)
pcolormesh(hmap_plt)
colorbar()
savefig(sys.argv[3])
