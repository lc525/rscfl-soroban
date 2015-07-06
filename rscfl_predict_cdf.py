#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
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
rcParams['figure.figsize'] = 10,5

train_file = open(sys.argv[1], 'rb')
trained_gauss = pickle.load(train_file)
train_file.close()

gp = trained_gauss

fig, ax = plt.subplots(1)
ax.set_axis_bgcolor("w")
#ax.set_yscale('log')

colors = ['k', 'b']
styles = ['solid', 'dashed']
labels = ['Low contention', 'AV scan running']
fs = []
i = 0

for f in [sys.argv[2], sys.argv[3]]:
    dataset = pd.read_pickle(f)
    print(dataset['srv_lat_us'].describe(percentiles=[0.90, 0.95, 0.99, 0.999, 0.99999, 1]))

    dataset['xen_cyc'] = dataset['xen_cyc'].astype('uint64')
    print(dataset['xen_cyc'].describe(percentiles=[0.90, 0.95, 0.99, 0.999, 0.99999, 1]))

    x = np.atleast_2d(dataset['xen_cyc'])

    y_pred, MSE = gp.predict(x.T, eval_MSE=True)
    sigma = np.sqrt(MSE)
    print(len(dataset[dataset['xen_cyc']==0])*100/len(dataset))

    to_plot = pd.Series(y_pred / 3.3e6)
    
    print(to_plot.describe(percentiles=[0.90, 0.95, 0.99, 0.999, 0.99999, 1]))
    #fs = to_plot.hist(bins=np.logspace(0.1, 10, 200))

    fs.append(to_plot.hist(bins=200, color=colors[i], label=labels[i], linestyle=(styles[i]),
                           ax=ax, cumulative=True, linewidth=2,
                      histtype='step', normed=True))
    plt.hold(True)
    i += 1

grid('off')

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

#plt.yscale('log', nonposy='clip')

plt.xlabel('Hypervisor-induced latency (ms)',fontsize=25, color='black', labelpad=10)
plt.title('CDFs of hypervisor-induced latency', color='black', y=1.1)

plt.tight_layout()


plt.axis([30,68,0,1])
#x = np.atleast_2d(np.linspace(0, 7e7, 1000)).T
#y_pred, MSE = gp.predict(x, eval_MSE=True)
#sigma = np.sqrt(MSE)

#fig = plt.figure()
#plt.plot(x, y_pred, 'b-', label=u'Prediction')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred - 3.500 * sigma,
#                        (y_pred + 3.500 * sigma)[::-1]]),
#         alpha=.4, fc='b', ec='None', label='95% confidence interval#')
artists = []
artists.append(mlines.Line2D([], [], color='k', label='Low contention'))
artists.append(mlines.Line2D([], [], color='b', label='AV scan running', linestyle='dashed'))
legend = plt.legend(loc=4, handles=artists)
frame = legend.get_frame()
frame.set_color('white')
frame.set_facecolor('white')

fig.savefig(sys.argv[4])
