#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import sys
import json
import re
import os
import scipy.stats as st
import numpy as np
import pickle
from pylab import *
from functools import partial
from sklearn import gaussian_process

pd.options.display.mpl_style = 'default'
rcParams['figure.figsize'] = 22,10

print(sys.argv[1])
bm_dataset = pd.read_pickle(sys.argv[1])
virt_dataset = pd.read_pickle(sys.argv[2])
if len(sys.argv) > 3:
    f = open(sys.argv[3], "w")
else:
    f = open("trained_gauss2.pickle", "w")

bm_dataset['no_vms']=bm_dataset['no_vms'].astype(int)
bm_dataset['xen_cyc'] = bm_dataset['xen_cyc'].astype('uint64')
bm_dataset['wct_us'] = bm_dataset['wct_us'].astype('uint64')
virt_dataset['no_vms']=virt_dataset['no_vms'].astype(int)
virt_dataset['xen_cyc'] = virt_dataset['xen_cyc'].astype('uint64')
virt_dataset['wct_us'] = virt_dataset['wct_us'].astype('uint64')

virt_dataset['expl'] = pd.Series(virt_dataset['wct_us'] * 100 / virt_dataset['srv_lat_us'], index=virt_dataset.index)
virt_dataset=virt_dataset[ (virt_dataset['expl'] <= 100)]

print("Starting data processing")
lat_dim = 'srv_lat_cyc'

hist = virt_dataset[lat_dim].sort(inplace=False)
ix = pd.Index(hist)
p_100 = len(hist)

def get_percentiles(x):
        pos = ix.get_loc(x)
        try:
         cnt = pos.stop - pos.start
         total = sum([x+1 for x in list(range(pos.start, pos.stop))])
         pct = total * 100 / (p_100 * cnt) # average out rankings
        except:
         pct = (pos + 1) * 100 / p_100

        return pct


percentiles = virt_dataset[lat_dim].apply(get_percentiles)
bm_percentiles = np.percentile(bm_dataset[lat_dim], percentiles)

dependent_var = virt_dataset[lat_dim] - bm_percentiles

#fig1 = plt.figure()
#bm_dataset['srv_lat_cyc'].hist(bins=100)
#fig1.savefig("./bm_hist")

#fig2 = plt.figure()
#plt.scatter(virt_dataset['xen_cyc'], dependent_var, c=virt_dataset['expl'], marker='+')
#plt.colorbar()
#xmax = virt_dataset['xen_cyc'].quantile(0.99)
#fig2.axes[0].set_xlim(0, xmax)
#savefig("training_ds.png")

#c_list = [[False], [False]*3, [True]*6, [False]*2, [True]*4]
c_list = [[False]*12, [True]*1, [False]*4]
c_list = [item for sublist in c_list for item in sublist]
virt_dataset=virt_dataset.ix[:,c_list].drop_duplicates()
virt_dataset=virt_dataset[ virt_dataset['xen_cyc'] < 1e9 ]

# sampling
train_samples = 1200
rows = np.random.choice(virt_dataset.index, train_samples, replace=False)
virt_dataset=virt_dataset.ix[rows]
dependent_var = dependent_var.ix[rows]

fig2 = plt.figure()
plt.scatter(virt_dataset['xen_cyc'], dependent_var, marker='+')
xmax = virt_dataset['xen_cyc'].quantile(0.99)
fig2.axes[0].set_xlim(0, xmax)
savefig("training_ds.png")

# training gaussian process
gp= gaussian_process.GaussianProcess(theta0=0.5e-2, thetaL=1e-3, thetaU=0.5, nugget=0.2, random_start=10)
virt_mx = virt_dataset.as_matrix()
gp.fit(virt_mx, dependent_var)
pickle.dump(gp, f)


# plotting resulting model (and some predictions)
x = np.atleast_2d(np.linspace(0, virt_dataset['xen_cyc'].max(), 1000)).T
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

fig = plt.figure()
plt.plot(virt_mx.ravel(), dependent_var, 'r+', markersize=7, label=u'observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
#plt.axis([0, 4e7, 0, 9e7])
plt.legend(loc='upper left')
plt.savefig("gaussian_process6.png")
