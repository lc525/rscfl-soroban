#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import sys
import json
import re
import os
from pylab import *
from functools import partial

pd.options.display.mpl_style = 'default'
rcParams['figure.figsize'] = 22,10

print(sys.argv[1])
dataset = pd.read_pickle(sys.argv[1])
dataset['no_vms'] = dataset['no_vms'].astype(int)

# given a line and a point, determine whether the point is above or below the
# line.
#
# returns  1 if point is "above"
# returns -1 if point is "below"
def line_val(line, pt_x, pt_y):
    ret = line[0]*pt_x + line[1]*pt_y + line[2]
    if ret > 0:
        return 1
    else:
        return -1

def pd_line_val(line, v):
    ret = line[0]*v[0] + line[1]*v[1] + line[2]
    if ret > 0:
        return 'L'
    else:
        return 'S'

def get_line(pt1_x, pt1_y, pt2_x, pt2_y):
    a = pt2_y - pt1_y
    b = pt1_x - pt2_x
    c = (pt1_y * pt2_x) - (pt2_y * pt1_x)
    return [a, b, c]

def mask(df, f):
    return df[f(df)]

#plt.ioff()
fig = plt.figure()
plt.scatter(dataset['srv_lat_cyc'], dataset['xen_cyc'], marker='+',
#         markeredgecolor='#4e5183', markerfacecolor='#cecef8', markersize=6.0,
   #      linestyle='', label='A',
 alpha=0.2, c=dataset['xen_bl'])

xd = dataset['srv_lat_cyc'].quantile(0.99)
xmax_ns = xd
ymax_cyc = xmax_ns
plt.plot([0, xmax_ns], [0, ymax_cyc], 'k-')
plt.axis([0, 2e8, 0, 2e8])
fig.set_tight_layout(True)
#plt.show()

fn_base = os.path.splitext(os.path.basename(sys.argv[1]))[0]

savefig(fn_base + "_scatter.png")

