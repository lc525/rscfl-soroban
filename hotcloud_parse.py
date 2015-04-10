#!/usr/bin/python
import matplotlib

import pandas as pd
import sys
import os
import json
import re
import gc
from pylab import *

if len(sys.argv) > 1:
    rootdir=sys.argv[1]
else:
    rootdir=os.path.dirname(os.path.realpath(__file__))
print("Parsing rscfl data files in  %s" %rootdir)

ix = set()
to_process = set()

ix_path = os.path.join(rootdir, "plot_dat.ix")
ix_file = open(ix_path, 'a+')
ix_file.seek(0)
for line in ix_file:
    ix.add(line.strip())
ix_file.close()

ix_file = open(ix_path, 'a+')
for root, subFolders, files in os.walk(rootdir):
    for filename in files:
        if os.path.splitext(os.path.basename(filename))[1] != '.dat':
            continue
        file_path = os.path.relpath(os.path.join(root,filename), rootdir)
        if file_path not in ix:
            to_process.add(os.path.relpath(file_path, rootdir))

print("Processing %d (new) data files" % len(to_process))
for datafile_path in to_process:
    print("Processing " + datafile_path)
    for stage in ['subsys', 'totals']:
        #do processing
        data_dict = []
        subsys_dict = []
        err = 0

        dirname = os.path.dirname(datafile_path)
        datafile_basename = os.path.splitext(os.path.basename(datafile_path))[0]
        mark_file = os.path.join(dirname, datafile_basename + ".mdat")

        ex_data_file=open(datafile_path)
        ex_mark_file=open(mark_file)

        ex_json_data = json.load(ex_data_file)
        ex_json_mark = json.load(ex_mark_file)

        ex_data_file.close()
        ex_mark_file.close()

        mk_items = ex_json_mark['marks']
        mk_dict = {}
        mark_reg=re.compile("vms_([^_]*)_.*")
        for mark in mk_items:
            m = mark_reg.match(mark['note'])
            if m:
                no_vms = m.group(1)
            else:
                no_vms = -1
            mk_dict[mark['at_id']] = (no_vms, mark['note'])

        ex_items = ex_json_data['data']
        (no_vms, c_groupex) = mk_dict[0]
        for req_data in ex_items:
            req_dict = {}
            s_dict = {}
            req_id = req_data['id']
            if req_id in mk_dict:
                (no_vms, c_groupex) = mk_dict[req_id] # tag point with active marker
#            if no_vms == -1:
#                continue
            #subsystems
            try:
                s_data = req_data['sdata']
                s_dict['id'] = req_id
                req_dict['id'] = req_id
                if stage == 'subsys':
                    for s in s_data:
                        req_dict[str(s['s'])] = s['cyc']
                        req_dict[str(s['s']) + "_sch_us"] = s['sch_us']
                        req_dict[str(s['s']) + "_sch_cyc"] = s['sch_cyc']
                        req_dict[str(s['s']) + "_xenc_pnd"] = s['xenc_pnd']
                        req_dict[str(s['s']) + "_xen_bl"] = s['xen_bl']
                        req_dict[str(s['s']) + "_xen_yl"] = s['xen_yl']
                        req_dict[str(s['s']) + "_xen_sch_us"] = s['xen_sch_us']
                        req_dict[str(s['s']) + "_xen_crmin"] = s['xen_crmin']
                        req_dict[str(s['s']) + "_xen_crmax"] = s['xen_crmax']
                elif stage == 'totals':
                    #totals
                    req_dict['wct_us'] = req_data['wct_us']
                    req_dict['cyc'] = req_data['cyc']
                    req_dict['sch_us'] = req_data['sch_us']
                    req_dict['sch_cyc'] = req_data['sch_cyc']
                    req_dict['xen_ev'] = req_data['xen_ev']
                    req_dict['xen_cyc'] = req_data['xen_cyc']
                    req_dict['xenc_pnd'] = req_data['xenc_pnd']
                    req_dict['xen_bl'] = req_data['xen_bl']
                    req_dict['xen_yl'] = req_data['xen_yl']
                    req_dict['xen_crmin'] = req_data['xen_crmin']
                    req_dict['xen_crmax'] = req_data['xen_crmax']
            except:
                req_dict['wct_us'] = 0
                print("no rscfl data\n")
            finally:
                req_dict['srv_lat_us'] = req_data['srv_lat_us']
                req_dict['srv_lat_cyc'] = req_data['srv_lat_cyc']
                req_dict['group_ex'] = c_groupex
                req_dict['no_vms'] = no_vms
                data_dict.append(req_dict)

        dataset = pd.DataFrame(data_dict)
        dataset = dataset.fillna(0)

        dataset.to_pickle(os.path.join(dirname, datafile_basename + "." + stage
                                       + ".sdat"))

        del data_dict
        del dataset
        gc.collect()
    ix_file.write(datafile_path + "\n")
