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
base_path = ""
if len(sys.argv) > 3:
    f = open(sys.argv[3], "w")
    base_path = os.path.dirname(sys.argv[3])
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
virt_dataset=virt_dataset[ virt_dataset['xen_cyc'] < 1e9 ]

percentiles = virt_dataset[lat_dim].apply(get_percentiles)
bm_percentiles = np.percentile(bm_dataset[lat_dim], percentiles)

#dependent_var = virt_dataset[lat_dim] - bm_percentiles
virt_dataset['y'] = pd.Series(virt_dataset[lat_dim] - bm_percentiles, index=virt_dataset.index)

c_list = [[False]*12, [True]*1, [False]*4]
c_list = [item for sublist in c_list for item in sublist]
#virt_dataset=virt_dataset.ix[:,c_list].drop_duplicates()

# binning
bsz = 100000
bins = range(virt_dataset['xen_cyc'].min(), virt_dataset['xen_cyc'].max(), bsz)
v_group = virt_dataset.groupby(np.digitize(virt_dataset['xen_cyc'], bins))
train_array = []
for name, gr in v_group:
    g_dict = {}
    g_dict['x']=gr['xen_cyc'].mean()
    g_dict['y']=gr['y'].median()
    train_array.append(g_dict)

train_data = pd.DataFrame(train_array)
train_data = train_data[train_data['x'] < 2e8]

# sampling
#train_samples = 600
#rows = np.random.choice(virt_dataset.index, train_samples, replace=False)
#virt_dataset=virt_dataset.ix[rows]
#dependent_var = dependent_var.ix[rows]

fig2 = plt.figure()
plt.scatter(train_data['x'], train_data['y'] , marker='+')
xmax = train_data['x'].quantile(0.99)
fig2.axes[0].set_xlim(0, xmax)
savefig(os.path.join(base_path, "training.scatter.png"))

# training gaussian process
ng = 1
gp= gaussian_process.GaussianProcess(theta0=20, thetaL=1, thetaU=50, nugget=ng, random_start=10)
virt_mx = np.atleast_2d(train_data['x'])
gp.fit(virt_mx.T, train_data['y'])
pickle.dump(gp, f)


# plotting resulting model (and some predictions)
x = np.atleast_2d(np.linspace(0, 2e8, 1000)).T
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)
rsigma = 0.7e7 * ng * ng

fig = plt.figure()
plt.plot(virt_mx.ravel(), train_data['y'], 'r+', markersize=7, label=u'observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 3.5 * (sigma),
                        (y_pred + 3.5 * (sigma))[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.axis([0, 2e8, 0, 2e8])
plt.legend(loc='upper left')
plt.savefig(os.path.join(base_path,"training.predict.png"))
