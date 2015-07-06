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
rcParams['figure.figsize'] = 12,6

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
fig, ax = plt.subplots(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

ax.set_axis_bgcolor("w")
plt.scatter(dataset['srv_lat_cyc'], dataset['xen_cyc'], marker='o', s=8.0,
            cmap='gnuplot', edgecolor='',
#         markeredgecolor='#4e5183', markerfacecolor='#cecef8', markersize=6.0,
   #      linestyle='', label='A',
 alpha=0.7, c=dataset['xen_bl'])

xd = dataset['srv_lat_cyc'].quantile(0.999)
xmax_ns = 2e8
ymax_cyc = xmax_ns
plt.plot([0, xmax_ns], [0, ymax_cyc], 'k-')
plt.plot([0, xmax_ns], [0, ymax_cyc*0.66], 'k--')
plt.text(xmax_ns - 0.41e8, 1.1e8, "66\%")
plt.plot([0, xmax_ns], [0, ymax_cyc*0.3], 'k--')
plt.text(xmax_ns - 0.41e8, 0.53e8, "30\%")
plt.axis([0.7e8, 1.65e8, 0, 1.3e8])
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
#plt.show()
cb = colorbar()
cb.set_alpha(1)
cb.ax.tick_params(axis='y', colors='black')
cb.set_label('xen \# schedules due to blocking', color='black', fontsize=22, labelpad=12)
cb.draw_all()
plt.ylabel('Xen scheduled-out (cycles)',fontsize=25, color='black', labelpad=10)
plt.xlabel('Server-side latency (cycles)',fontsize=25, color='black', labelpad=10)
plt.title('Xen scheduling measurements', color='black', y=1.1)
fig.set_tight_layout(True)
fn_base = os.path.splitext(os.path.basename(sys.argv[1]))[0]

fig.savefig(sys.argv[2])

