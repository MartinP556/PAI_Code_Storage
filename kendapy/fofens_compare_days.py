#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . F O F E N S _ C O M P A R E _ D A Y S
#  generate plots from data produced by fofens_eval for different days
#
#  2019.3 L.Scheck

from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os, sys, getpass, subprocess, argparse, time, re, gc, hashlib
import pickle
from kendapy.ekf import Ekf, tables
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range, Timeaxis
from kendapy.fofens_eval import fofens_eval, otvn_combinations, fofens_average
from kendapy.colseq import colseq

def plot_day_comparison( days, xt, xt_ref, afee, veri_time_min=180, uv=True, figsize=(3,4) ) :

    # afee contents:
    # afee[day][expid][veri_time_min][obstype][varname]['overall']['n_obs']
    # afee[day][expid][veri_time_min][obstype][varname]['overall']['rmse']
    # afee[day][expid][veri_time_min][obstype][varname]['overall']['bias']

    print('plot_day_comparison : sorting variables...')
    day = days[0]
    expid = day+'.'+xt_ref
    obstypes_containing = {}
    #for veri_time_min in afee[day][expid] :
    for obstype in afee[day][expid][veri_time_min] :
        for varname in afee[day][expid][veri_time_min][obstype] :
            if not varname in obstypes_containing :
                obstypes_containing[varname] = {}
            obstypes_containing[varname][obstype] = True

    varname_groups = {}
    for varname in obstypes_containing :
        print( ' -- ', varname, ' is measured by ', obstypes_containing[varname].keys())
        if uv :
            if varname.startswith('U') :
                varname_groups[ varname.replace('U','U,V ' ).replace('10M','10m').strip() ] = [varname, varname.replace('U','V')]
            elif varname.startswith('V') :
                pass
            else :
                varname_groups[ varname ] = [ varname ]
        else :
            varname_groups[ varname ] = [ varname ]

    print(' -> varname groups : ', varname_groups)


    fig, ax = plt.subplots(figsize=figsize)
    xmax = 0
    y = 0
    lbl = True
    yticks = []
    yticklabels = []
    for vg in sorted(varname_groups.keys()) :
        varnames = varname_groups[vg]
        obstypes = obstypes_containing[varnames[0]].keys()
        for obstype in obstypes :
            for iday, day in enumerate(days) :
                # averages individual varname results
                expid     = day+'.'+xt
                expid_ref = day+'.'+xt_ref
                ref = afee[day][expid_ref][veri_time_min][obstype]
                tst = afee[day][expid    ][veri_time_min][obstype]

                n_obs     = np.array([ tst[varname]['overall']['n_obs'] for varname in varnames ]).sum()
                n_obs_ref = np.array([ ref[varname]['overall']['n_obs'] for varname in varnames ]).sum()
                if( n_obs != n_obs_ref ) :
                    raise ValueError('n_obs and n_obs_ref do not agree!')

                rmse_ref = np.sqrt( np.array([ (ref[varname]['overall']['rmse']**2)*ref[varname]['overall']['n_obs'] for varname in varnames ]).sum() / n_obs )
                rmse_tst = np.sqrt( np.array([ (tst[varname]['overall']['rmse']**2)*tst[varname]['overall']['n_obs'] for varname in varnames ]).sum() / n_obs )
                drmse_rel = (rmse_tst-rmse_ref)/rmse_ref
                print( vg, obstype, day, rmse_tst, rmse_ref, drmse_rel )

                yd = y + 0.4*iday
                ax.plot( (0,drmse_rel*100), (yd,yd), ['#ff7f2a','#2ac2ff'][iday], linewidth=8, solid_capstyle='butt',
                         label=day.replace('0605','JUNE 5').replace('0529','MAY 29') if lbl else None )
                ax.text( 0, yd, ' '+str(n_obs)+' ', va='center', ha='right' if drmse_rel > 0 else 'left', color='k', fontsize=8 )
                xmax = np.ceil(np.maximum( abs(drmse_rel)*100, xmax ))

            yticks.append(y+0.2)
            yticklabels.append( '{} [{}]'.format(vg,obstype.replace('PILOT','PROFILER')) )
            lbl = False

            y += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim((-xmax, xmax))
    tst_name = xt.replace('104','VISCONV')
    ref_name = xt_ref.replace('101','CONV')
    ax.set_xlabel(r'(RMSE$_{\rm %s}$ - RMSE$_{\rm %s}$) / RMSE$_{\rm %s}$ [%%]' % (tst_name,ref_name,ref_name))
    ax.legend(loc='center right', fontsize=8)
    #ax.grid()
    fig.savefig('drmse_overview_{:d}min.pdf'.format(veri_time_min), bbox_inches='tight')
    plt.close(fig)

        #print(vg, ' : ', varnames, ' from ', obstypes)


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Evaluate FOF file ensembles')

    parser.add_argument( '-d', '--days',       dest='days',   help='coma-separated list of days, e.g. 0605,0529', default='0605,0529' )
    parser.add_argument( '-x', '--experiment', dest='xt',     help='experiment type id', default='104' )
    parser.add_argument( '-r', '--reference',  dest='xt_ref', help='reference  type id', default='101' )

    parser.add_argument(       '--recompute',   dest='recompute',  help='do not read cache files', action='store_true' )

    parser.add_argument( '-A', '--area-filter',  dest='area_filter',  help='area filter for observations', default='auto' )
    parser.add_argument( '-S', '--state-filter', dest='state_filter', help='state filter for observations', default='active' )
    parser.add_argument( '-V', '--variables',    dest='varnames',    help='comma-separated list of variables  to be considered (default: all)', default='' )
    parser.add_argument( '-O', '--obstypes',     dest='obstypes',    help='comma-separated list of obs. types to be considered (default: all)', default='' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument( '-I', '--input-path',  dest='input_path',  help='path to the directory containing the log files', default='' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )

    args = parser.parse_args()

    afee = {}

    days = args.days.split(',')
    for day in days :

        logfiles = [ day+'.'+args.xt_ref, day+'.'+args.xt ]
        for i, lfn in enumerate(logfiles) :
            logfiles[i] += '/run_cycle_'+logfiles[i]+'.log'
            logfiles[i] = os.path.join( args.input_path, logfiles[i] )

        xps = []
        for i, logfile in enumerate(logfiles) :
            xp = Experiment(logfile)
            print('experiment %s : %s' % ( xp.settings['exp'], xp.description() ))
            xps.append( xp )

        # process all forecasts between start_time and end_time and all veri_times
        start_time = xps[0].lfcst_start_times[0]
        end_time   = xps[0].lfcst_start_times[-1]
        fcst_start_times = [f for f in xps[0].lfcst_start_times if (Time14(f) >= Time14(start_time)) & (Time14(f) <= Time14(end_time))]
        assint_min = int(xps[0].settings['ASSINT'])//60
        #veri_times_min = range( assint_min, (int(xps[0].lfcst_settings['FCTIME'])//60)+1, assint_min )
        veri_times_min = list(range( 0, (int(xps[0].lfcst_settings['FCTIME'])//60)+1, assint_min))

        # get data
        fee =  fofens_eval( xps, fcst_start_times, veri_times_min, lfcst=True, common_obs = True,
                           obstypes = args.obstypes.split(',') if args.obstypes != '' else None,
                           varnames = args.varnames.split(',') if args.varnames != '' else None,
                           state_filter = args.state_filter,
                           recompute = args.recompute )
        otvn, fst, vt = otvn_combinations(fee)
        print( day, ' >>> ', otvn, fst, vt)
        afee[day] = fofens_average(fee)

    for veri_time_min in veri_times_min :
        if veri_time_min > 0 :
            plot_day_comparison( days, args.xt, args.xt_ref, afee, veri_time_min=veri_time_min, uv=True )



