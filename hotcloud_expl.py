#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import sys
import os
import json
import re
from pylab import *
from functools import partial

pd.options.display.mpl_style = 'default'
rcParams['figure.figsize'] = 22,10

dataset = pd.read_pickle(sys.argv[1])
dataset['no_vms'] = dataset['no_vms'].astype(int)

dataset['expl'] = pd.Series(dataset['wct_us'] * 100 / dataset['srv_lat_us'],
                            index=dataset.index)

dataset = dataset[(dataset['group_ex'] != 'STOP') & (dataset['expl'] <= 100)]

fn_base = os.path.splitext(os.path.basename(sys.argv[1]))[0]

fig_ax = dataset['expl'].plot(kind='hist', bins=200)
fig = fig_ax.get_figure()
fig_ax.set_xlim(0,100)
fig.savefig(fn_base + "_all.png")

plt.figure()
# tail latency
qt = dataset['srv_lat_us'].quantile(0.9)
df = dataset[dataset['srv_lat_us'] >= qt]
fig_ax2 = df['expl'].plot(kind='hist', bins=200)
fig_ax2.set_xlim(0,100)
fig2 = fig_ax2.get_figure()
fig2.savefig(fn_base + "_tail.png")
