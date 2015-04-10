#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import sys
import json
from pylab import *
from functools import partial
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

# nfs-only experiments, varying file size and ab concurrency
# also, one test involves another machine continuously updating
# the file served by lighttpd.
#
# each element of this array must be the name of a directory storing
# data for the given experiment.
experiment_sets=["hypervisor_tests"]

# for each element in experiment_sets, provide an array of data files
# containing /rscfl data for that experiment
experiment_data_files=[["exp_rscfl_hyp2.dat"]]

# for each element in experiment_data_files, provide the corresponding
# file containing markers, using the same array nesting.
experiment_mark_files=[["exp_marks_hyp2.dat"]]

pd.options.display.mpl_style = 'default'
rcParams['figure.figsize'] = 22,10

data_dict = []

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

# load experimental data into a pandas DataFrame
for ix, ex_dir in enumerate(experiment_sets):
    for ix_f, ex_data_filen in enumerate(experiment_data_files[ix]):
        ex_mark_filen = experiment_mark_files[ix][ix_f]

        ex_data_file = open(ex_dir+"/"+ex_data_filen)
        ex_mark_file = open(ex_dir+"/"+ex_mark_filen)

        ex_json_data = json.load(ex_data_file)
        ex_json_mark = json.load(ex_mark_file)

        ex_data_file.close()
        ex_mark_file.close()

        no_machines=0
        load = 0
        iter = 0

        mk_items = ex_json_mark['marks']
        mk_dict = {}
        for mark in mk_items:
            mk_dict[mark['at_id']] = mark['note']

        ex_items = ex_json_data['data']
        c_groupex = mk_dict[0]
        for req_data in ex_items:

            # simulate hypervisor/load increases
            if iter % 5000 == 0:
                no_machines = no_machines + 1
                load = 0
            if iter % 500 == 0:
                load = load + 1

            iter = iter + 1

            req_dict = {}
            req_id = req_data['id']
            req_dict['id'] = req_id
            if req_id in mk_dict:
                c_groupex = mk_dict[req_id] # tag point with active marker
            #subsystems
            try:
                s_data = req_data['sdata']
                for s in s_data:
                    req_dict[str(s['s'])] = s['cycles']
                    req_dict[str(s['s'])+"_h_out_us"] = s['hso_wc']
                    req_dict[str(s['s'])+"_p_evch"] = s['p_evch']
                #totals
                req_dict['t_wc_us'] = req_data['t_wc_us']
                req_dict['t_cyc'] = req_data['t_cyc']
                req_dict['t_hyp_ev'] = req_data['t_hs']
                req_dict['t_hyp_cyc'] = req_data['t_hs_c']
            except:
                req_dict['t_wc_us'] = 0
                print("no rscfl data\n")
            finally:
                req_dict['srv_lat_us'] = req_data['srv_lat_us']
                req_dict['vms'] = no_machines
                req_dict['load'] = load
                data_dict.append(req_dict)

# dimensions of dataframe table:
#
#   +---- cycles for subsystems s1 .... sn
#  /
# si_cyc | si_hwct_out | ... | group_ex | srv_lat_us | t_wc_us | t_hyp_ev | t_hyp_cyc
#
dataset = pd.DataFrame(data_dict)
dataset = dataset.fillna(0)
lat_quantile = partial(np.percentile, q=90)
agg = np.max
lat_map = pd.tools.pivot.pivot_table(dataset, values='srv_lat_us',
                                     index=['vms'], columns=['load'],
                                     aggfunc=agg)

lat_map_np = np.array(lat_map)

# logarithmic plot
#pcolormesh(lat_map_np, norm=LogNorm(vmin=dataset['srv_lat_us'].min(), vmax=dataset['srv_lat_us'].max()))
#colorbar()
#colorbar(format=LogFormatterMathtext())

# linear plot
pcolormesh(lat_map_np)
colorbar()
