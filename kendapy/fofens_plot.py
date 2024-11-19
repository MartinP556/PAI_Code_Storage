#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . F O F E N S _ P L O T
#  generate plots from data produced by fofens_eval
#
#  2018.2 L.Scheck

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

#-----------------------------------------------------------------------------------------------------------------------
def comparison_barplot( data, cats, subcats, filename, cat_info=None, title=None, legend_title=None, colors=None,
                        xlim=None, figsize=(5,8) ) :
    """
    Generate comparison bar plot
    :param data:     shape (n_cat,n_subcat)
    :param cats:     string list of length n_cats
    :param subcats:  string list of length n_subcats
    :param cat_info: optional string list of length n_cats, to be displayed next to categories in small print
    :param title:    optional plot title
    :return: 
    """

    print(data.shape, len(cats), len(subcats))

    n_cats, n_subcats = data.shape

    if colors is None :
        colors = [ colseq(i,n_subcats) for i in range(n_subcats) ]

    fig, ax = plt.subplots( figsize=figsize )
    y0 = np.arange(n_cats)
    for isc, sc in enumerate(subcats) :
        ax.barh( y0 + float(isc-(n_subcats-1)/2.0)/(n_subcats+1), data[:,isc], 1.0/float(n_subcats+1),
                 align='center', color=colors[isc], edgecolor='', label=sc )

    ax.set_yticks(y0)
    ax.set_yticklabels(cats)
    ax.set_ylim((-0.5,n_cats-0.5))
    ax.plot( (0,0), ax.get_ylim(), 'k', zorder=0, alpha=0.5 )

    if not xlim is None :
        ax.set_xlim(xlim)
    for i in range(n_cats+1) :
        ax.plot( ax.get_xlim(), [i-0.5]*2, 'k', zorder=0, alpha=0.25 )

    if not cat_info is None :
        xlm = ax.get_xlim()
        for i in range(n_cats) :
            ax.text( xlm[0] + 0.01*(xlm[1]-xlm[0]), i-((n_subcats-1)/2.0)/(n_subcats+1), cat_info[i], fontsize=10 )
    if not title is None : fig.suptitle(title)
    ax.legend( frameon=False, title=legend_title, fontsize=10, loc='upper right')
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)

#-----------------------------------------------------------------------------------------------------------------------
def plot_error_overview( fee, refexp=None, common_obs=True, figsize=(10,7), colors=None, names=None, n_obs_min=20,
                         file_type='png' ) :

    if refexp is None :
        refexp = list(fee.keys())[0]
    print('selected %s as reference experiment...' % refexp)

    otvn, fst, vt = otvn_combinations(fee)

    print('veri_times : ', vt)
    print('fcst_start_times : ', fst)

    # for each fcst start time and each verification time : plot changes in rmse with respect to the reference run .....

    for veri_time_min in vt[1:] :
        for fcst_start_time in fst :
            veri_time = (Time14(fcst_start_time)+Time14(veri_time_min*60)).string14()
            print('fcst_start_time = %s, veri_time = %d min = %s' % (fcst_start_time,veri_time_min,veri_time))

            drmse = {}
            n_obs = {}
            otvncombs = []
            for obstype in sorted(otvn.keys()) :
                if not obstype in fee[refexp][fcst_start_time][veri_time] :
                    continue
                for varname in sorted(list(otvn[obstype])) :
                    if not varname in fee[refexp][fcst_start_time][veri_time][obstype] :
                        continue

                    n_obs_ref = fee[refexp][fcst_start_time][veri_time][obstype][varname]['n_obs']
                    if n_obs_ref < n_obs_min :
                        continue

                    otvncomb = obstype+' / '+varname
                    otvncombs.append(otvncomb)
                    drmse[otvncomb] = {}
                    n_obs[otvncomb] = n_obs_ref
                    rmse_ref = fee[refexp][fcst_start_time][veri_time][obstype][varname]['overall']['rmse']

                    expids = []
                    for expid in fee :
                        if expid == refexp :
                            continue
                        expids.append(expid)
                        drmse[otvncomb][expid] = (fee[expid][fcst_start_time][veri_time][obstype][varname]['overall']['rmse']-rmse_ref)/rmse_ref
                        print(otvncomb, drmse[otvncomb][expid])

            compdat = np.array([ [drmse[otvncomb][expid] for expid in expids ] for otvncomb in otvncombs] )
            comparison_barplot( compdat, otvncombs, [ expid+':'+names[expid] for expid in expids ],
                                colors = [ colors[expid] for expid in expids ],
                                cat_info=[ str(n_obs[otvncomb]) for otvncomb in otvncombs ],
                                legend_title='%s + %d min'%(fcst_start_time,veri_time_min), xlim=(-0.2,0.2),
                                title='relative change in RMSE with respect to %s' % names[refexp],
                                filename='drmse_%s_plus%04dmin.%s'%(fcst_start_time,veri_time_min,file_type) )

    # plot changes in rmse with respect to the reference run, averaged over all start times ............................

    afee = fofens_average(fee)
    for veri_time_min in veri_times_min :

        if veri_time_min > 0 : # no FOF results available at t=0...
            drmse = {}
            n_obs = {}
            otvncombs = []
            for obstype in sorted(otvn.keys()) :
                if not obstype in afee[refexp][veri_time_min] :
                    continue
                for varname in sorted(list(otvn[obstype])) :
                    if not varname in afee[refexp][veri_time_min][obstype] :
                        continue
                    print('>>>>>', list(afee[refexp][veri_time_min][obstype][varname].keys()))
                    n_obs_ref = afee[refexp][veri_time_min][obstype][varname]['overall']['n_obs']
                    if n_obs_ref < n_obs_min :
                        continue

                    otvncomb = obstype+' / '+varname
                    otvncombs.append(otvncomb)
                    drmse[otvncomb] = {}
                    n_obs[otvncomb] = n_obs_ref
                    rmse_ref = afee[refexp][veri_time_min][obstype][varname]['overall']['rmse']

                    expids = []
                    for expid in fee :
                        if expid == refexp :
                            continue
                        expids.append(expid)
                        drmse[otvncomb][expid] = (afee[expid][veri_time_min][obstype][varname]['overall']['rmse']-rmse_ref)/rmse_ref
                        print(otvncomb, drmse[otvncomb][expid])

            compdat = np.array([ [drmse[otvncomb][expid] for expid in expids ] for otvncomb in otvncombs] )
            comparison_barplot( compdat, otvncombs, [ names[expid] for expid in expids ],
                                colors = [ colors[expid] for expid in expids ],
                                cat_info=[ str(n_obs[otvncomb]) for otvncomb in otvncombs ],
                                legend_title='%d min forecast'%(veri_time_min), xlim=(-0.1,0.1),
                                title='average relative change in RMSE with respect to %s' % names[refexp],
                                filename='drmse_%s-%s_plus%04dmin.%s'%(fcst_start_times[0],fcst_start_times[-1],
                                                                       veri_time_min,file_type) )



#-----------------------------------------------------------------------------------------------------------------------
def plot_error_evolution( fee, common_obs=True, figsize=(10,7), colors=None, names=None, cycle=True, n_obs=True, plot_spread=False,
                          exclude_obstypes=['TEMP'], file_type='png' ) :

    otvn, fst, vt = otvn_combinations(fee)

    font = FontProperties()
    boldfont = font.copy()
    boldfont.set_weight('bold')

    for obstype in sorted(otvn.keys()) :
        if obstype in exclude_obstypes :
            continue

        for varname in sorted(list(otvn[obstype])) :
            print('PLOTTING ', obstype, varname)

            # RMSE / SPREAD / BIAS
            fig, ax = plt.subplots(figsize=figsize)
            ta = Timeaxis()
            for expid in fee :
                labeled = False
                for fcst_start_time in fee[expid] :
                    veri_times = sorted(fee[expid][fcst_start_time].keys())
                    if obstype in fee[expid][fcst_start_time][veri_times[0]] and varname in fee[expid][fcst_start_time][veri_times[0]][obstype] :
                        t_veri_times = ta.convert(veri_times)

                        # long forecast results
                        rmse   = [ fee[expid][fcst_start_time][t][obstype][varname]['overall']['rmse']   for t in veri_times[1:] ]
                        bias   = [ fee[expid][fcst_start_time][t][obstype][varname]['overall']['bias']   for t in veri_times[1:] ]
                        #print veri_times, t_veri_times

                        ax.plot( t_veri_times[1:], rmse, linewidth=2, color=colors[expid], label=names[expid] if not labeled else None )
                        labeled = True
                        ax.plot( t_veri_times[1:], bias, '--', linewidth=2, color=colors[expid] )
                        if plot_spread :
                            spread = [ fee[expid][fcst_start_time][t][obstype][varname]['overall']['spread'] for t in veri_times[1:] ]
                            ax.plot( t_veri_times[1:], spread, ':', linewidth=2, color=colors[expid] )

                        # add first point from analysis
                        ana_rmse = [ fee[expid][fcst_start_time][t][obstype][varname]['overall_ekf']['ana_rmse'] for t in veri_times ]
                        ana_bias = [ fee[expid][fcst_start_time][t][obstype][varname]['overall_ekf']['ana_bias'] for t in veri_times ]
                        ax.plot( t_veri_times[:2], [ana_rmse[0],rmse[0]], linewidth=2, color=colors[expid] )
                        ax.plot( t_veri_times[:2], [ana_bias[0],bias[0]], '--', linewidth=2, color=colors[expid] )

                        if cycle : # plot also cycling results
                            fg_rmse  = [ fee[expid][fcst_start_time][t][obstype][varname]['overall_ekf']['rmse']     for t in veri_times ]
                            fg_bias  = [ fee[expid][fcst_start_time][t][obstype][varname]['overall_ekf']['bias']     for t in veri_times ]

                            for i, t in enumerate(t_veri_times[:-1]) :
                                ax.plot( (t_veri_times[i],t_veri_times[i+1],t_veri_times[i+1]),
                                         (ana_rmse[i],fg_rmse[i+1],ana_rmse[i+1]), linewidth=1, color=colors[expid], alpha=0.5 )
                                ax.plot( (t_veri_times[i],t_veri_times[i+1],t_veri_times[i+1]),
                                         (ana_bias[i],fg_bias[i+1],ana_bias[i+1]), '--', linewidth=1, color=colors[expid], alpha=0.5 )

                        if n_obs and expid == list(fee.keys())[-1] :
                            ylm = ax.get_ylim()
                            ytxt = ylm[0] + 0.01*(ylm[1]-ylm[0])
                            for i in range(len(t_veri_times)) :
                                ax.text( t_veri_times[i], ytxt, str(fee[expid][fcst_start_time][veri_times[i]][obstype][varname]['n_obs']),
                                         fontsize=8, ha='center' )

            ax.legend(frameon=False, fontsize=12)
            ax.grid()
            ta.set_tickmarks(ax,margin=0.03)

            xlm, ylm = ax.get_xlim(), ax.get_ylim()
            ax.text( xlm[0]+0.02*(xlm[1]-xlm[0]), ylm[0]+0.98*(ylm[1]-ylm[0]), '%s / %s' % (obstype,varname),
                     fontsize=24, color='#999999', va='top', fontproperties=boldfont, zorder=0 )
            ax.plot( xlm, (0,0), color='k', alpha=0.5, linewidth=0.5, zorder=0 )

            ax.set_xlabel('time')
            ax.set_ylabel('bias (dashed), spread (dotted), rmse (solid)')
            fig.savefig('fofens_eevo_'+obstype+'_'+varname+'.'+file_type, bbox_inches='tight')
            plt.close(fig)

    #print veri_time, res[expid][fcst_start_time][veri_time]['AIREP']['RH']['overall']['rmse'], '|',



#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Evaluate FOF file ensembles')

    parser.add_argument( '-E', '--evolution',  dest='evolution',  help='generate error evolution plots', action='store_true' )
    parser.add_argument( '-o', '--overview',   dest='overview',   help='generate error overview plots',  action='store_true' )
    parser.add_argument( '-P', '--profiles',   dest='profiles',   help='generate error profile plots',   action='store_true' )

    parser.add_argument( '-s', '--start-time',    dest='start_time', help='(first) fcst start time',    default='' )
    parser.add_argument( '-e', '--end-time',      dest='end_time',   help='last    fcst start time',    default='' )
    parser.add_argument( '-a', '--analysis-time', dest='veri_time',  help='analysis time',      default='' )
    parser.add_argument( '-c', '--cycle',         dest='cycle',      help='plot also cycle results',   action='store_true' )
    parser.add_argument(       '--individual',    dest='individual_obs', help='use individual set of observations for each experiment',   action='store_true' )
    parser.add_argument( '-r', '--reference',     dest='refexp', help='reference experiment (default: the first one)', default='' )

    parser.add_argument(       '--recompute',   dest='recompute',  help='do not read cache files', action='store_true' )

    parser.add_argument(       '--colors',      dest='colors',     help='comma-separated list of colors for the experiments', default='' )
    parser.add_argument(       '--names',       dest='names',      help='comma-separated list of names (use ,, for commas within the names)', default='' )

    parser.add_argument( '-A', '--area-filter',  dest='area_filter',  help='area filter for observations', default='auto' )
    parser.add_argument( '-S', '--state-filter', dest='state_filter', help='state filter for observations', default='active' )
    parser.add_argument( '-V', '--variables',    dest='varnames',    help='comma-separated list of variables  to be considered (default: all)', default='' )
    parser.add_argument( '-O', '--obstypes',     dest='obstypes',    help='comma-separated list of obs. types to be considered (default: all)', default='' )

    parser.add_argument(       '--dwd2lmu',     dest='dwd2lmu',     help='convert settings',  action='store_true' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument( '-I', '--input-path',  dest='input_path',  help='path to the directory containing the log files', default='' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    logfiles = args.logfile
    for i, lfn in enumerate(logfiles) :
        if not lfn.endswith('.log') : # asssume it is an experiment id and not a log file
            logfiles[i] += '/run_cycle_'+logfiles[i]+'.log'
        if args.input_path != '' : # add input path
            logfiles[i] = os.path.join( args.input_path, logfiles[i] )

    xps = []
    for i, logfile in enumerate(logfiles) :
        xp = Experiment(logfile)
        print('experiment %s : %s' % ( xp.settings['exp'], xp.description() ))
        xps.append( xp )

    # process all forecasts between start_time and end_time and all veri_times
    start_time = args.start_time if args.start_time != '' else xps[0].lfcst_start_times[0]
    end_time   = args.end_time   if args.end_time   != '' else xps[0].lfcst_start_times[-1]
    fcst_start_times = [f for f in xps[0].lfcst_start_times if (Time14(f) >= Time14(start_time)) & (Time14(f) <= Time14(end_time))]
    assint_min = int(xps[0].settings['ASSINT'])//60
    #veri_times_min = range( assint_min, (int(xps[0].lfcst_settings['FCTIME'])//60)+1, assint_min )
    veri_times_min = list(range( 0, (int(xps[0].lfcst_settings['FCTIME'])//60)+1, assint_min))

    # get data
    fee = fofens_eval( xps, fcst_start_times, veri_times_min, lfcst=(not args.cycle),
                       common_obs = not args.individual_obs,
                       obstypes = args.obstypes.split(',') if args.obstypes != '' else None,
                       varnames = args.varnames.split(',') if args.varnames != '' else None,
                       state_filter = args.state_filter,
                       recompute = args.recompute )

    if args.colors != '' :
        colors = { xp.expid : args.colors.split(',')[i] for i, xp in enumerate(xps) }
    else :
        colors = { xp.expid : colseq(i,len(xps)) for i, xp in enumerate(xps) }
    if args.names != '' :
        names  = { xp.expid : args.names.split(',')[i]  for i, xp in enumerate(xps) }
    else :
        names  = { xp.expid : xp.expid for i, xp in enumerate(xps) }
    print('using names ', names)

    if args.evolution :
        plot_error_evolution( fee, common_obs=not args.individual_obs, colors=colors, names=names, cycle=args.cycle,
                              file_type=args.file_type )

    if args.overview :
        plot_error_overview( fee, refexp = args.refexp if not args.refexp=='' else xps[0].expid,
                             common_obs=not args.individual_obs, colors=colors, names=names, file_type=args.file_type  )


