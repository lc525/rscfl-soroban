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
#dataset['no_vms'] = dataset['no_vms'].astype(int)

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
plt.plot(dataset['srv_lat_cyc'], dataset['xen_cyc'], marker='+',
         markeredgecolor='#4e5183', markerfacecolor='#cecef8', markersize=6.0,
         linestyle='', label='A')
#plt.axis([0,1e5,0,1e11])
dataset['srv_lat_cyc'].hist(bins=40,range=(0,1250))
print(dataset['srv_lat_cyc'].describe(percentiles=[0,0.90,0.95,0.99]))

fn_base = os.path.splitext(os.path.basename(sys.argv[1]))[0]

savefig(fn_base + "_scatter.png")

