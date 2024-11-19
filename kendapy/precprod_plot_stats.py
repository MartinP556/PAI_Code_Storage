#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P R E C P R O D _ P L O T _ S T A T S
#  plot precipitation statistics for KENDA experiments
#
#  2018.1 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

import sys, os, argparse
from numpy import *
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.colseq import colseq
from kendapy.precprod_stats import get_precprod_lfcst_stats

#-------------------------------------------------------------------------------
def error_evo_plot( xp, times, stats, lfcst_stats, titlestr='', ekf=True, verbose=True,
                    start_plot=True, finish_plot=True, color='k', style='' ) :

    if start_plot and finish_plot :
        comparison = False
    else :
        comparison = True

    cycavg = False # plot values averaged over cycles?

    if comparison :
        colrmse        = color
        colrmse_dots   = color
        colbias        = color
        colbias_dots   = color
        collfcst       = color
        colspread      = None
        colspread_dots = None

        rmse_label=xp.settings['EXPID']
        spread_label=None
        bias_label=None

    else :
        colrmse        = '#000000'
        colrmse_dots   = 'b'
        colbias        = '#999999'
        colbias_dots   = '#ee9900'
        collfcst       = '#ee6699'
        colspread      = '#999999'
        colspread_dots = '#00ee99'

        rmse_label='RMSE'
        spread_label='spread'
        bias_label='bias'

        if 'talk' in style :
            colspread_dots = None
            colbias_dots   = '#ee0099'
        if 'nodots' in style :
            colspread_dots = None
            colbias_dots   = None
            colrmse_dots   = None

    if ekf :
        # get data from ekfRAD files
        ekfevo = xp.compute_error_evolution( times, obs_types=['RAD'], variables=['REFL'], state_filter='valid' )['RAD']['REFL']
        #        {'RAD': {'REFL': {'20160605100000': {'fgmean': {'bias': -0.0030769857, 'rmse': 0.08585836}, 'fgspread': {'mean': 0.10906658}...

    if start_plot :
        plt.figure(1,figsize=(10,5))
        plt.clf()

    hmin = Time14(times[0]).dayhour()
    h = hmin
    add_label=True
    for i, t in enumerate(times) :

        if verbose :
            print('error_evo_plot: processing fcst start time #%d = %s' % ( i, t ))
            #print '                available fcst times: ', sorted(stats[t].keys())
            #print '                visop output is available at ', h+xp.visop_fcsttimes()/60.0

        plt.figure(1)

        if not lfcst_stats is None :
            if t in lfcst_stats :
                print(lfcst_stats[t])
                lfcst_t_min = sorted(lfcst_stats[t].keys())
                lfcst_t_hour = Time14(t).dayhour() + array(lfcst_t_min)/60.0
                print(lfcst_t_hour)
                plt.plot( lfcst_t_hour, [ lfcst_stats[t][f]['rmse'] for f in lfcst_t_min ], '-', color=collfcst, linewidth=3 )
                plt.plot( lfcst_t_hour, [ lfcst_stats[t][f]['bias'] for f in lfcst_t_min ], '-', color=collfcst, linewidth=1 )

        if not colrmse is None :
            plt.plot( h+xp.visop_fcsttimes()/60.0, [ stats[t][f]['rmse']   for f in sorted(stats[t].keys()) ], '.-', color=colrmse, linewidth=1.5,  label=rmse_label   if add_label else None  )
        if not colspread is None :
            plt.plot( h+xp.visop_fcsttimes()/60.0, [ stats[t][f]['spread'] for f in sorted(stats[t].keys()) ], '--', color=colspread, label=spread_label if add_label else None  )
        if not colbias is None :
            plt.plot( h+xp.visop_fcsttimes()/60.0, [ stats[t][f]['bias']   for f in sorted(stats[t].keys()) ], '.-', color=colbias, linewidth=0.5,  label=bias_label   if add_label else None  )

        if ekf :
            if cycavg :
                fgmean = sqrt((array([ stats[t][f]['rmse'] for f in sorted(stats[t].keys())[1:] ])**2).mean())
                plt.plot( (h+xp.visop_fcsttimes()[1]/60.0,h+xp.visop_fcsttimes()[-1]/60.0), (fgmean,fgmean), 'r' )

            # EKF values: correspond to mean values in the interval ]t,t+ASSINT]
            if i < len(times)-1 and times[i+1] in ekfevo :
                if cycavg :
                    # EKF values averaged over assimilation window
                    plt.scatter( h+xp.visop_fcsttimes()[1:].mean()/60.0, ekfevo[times[i+1]]['fgmean']['rmse'], marker='x', color='r' )
                    plt.scatter( h+xp.visop_fcsttimes()[1:].mean()/60.0, ekfevo[times[i+1]]['anamean']['rmse'], marker='o', color='r' )
                    plt.plot( (h+xp.visop_fcsttimes()[1]/60.0,h+xp.visop_fcsttimes()[-1]/60.0), (ekfevo[times[i+1]]['anamean']['rmse'],ekfevo[times[i+1]]['anamean']['rmse']), color='r' )

                #plt.scatter( h+xp.visop_fcsttimes()[1:].mean()/60.0, ekfevo[times[i+1]]['fgens']['spread'], marker='x', color=colspread )
                #plt.scatter( h+xp.visop_fcsttimes()[1:].mean()/60.0, ekfevo[times[i+1]]['anaens']['spread'], marker='o', color=colspread )

                # EKF values at analysis times
                ms = 30
                if not colrmse_dots is None :
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['fgmean']['rmse_last'],   s=ms, marker='x', color=colrmse_dots )
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['anamean']['rmse_last'],  s=ms, marker='o', color=colrmse_dots )
                    plt.scatter( h+xp.visop_fcsttimes()[0]/60.0, stats[t][sorted(stats[t].keys())[0]]['rmse'], s=ms, marker='s', color=colrmse_dots )
                    #plt.gca().arrow( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['fgmean']['rmse_last'],
                    #                 0, ekfevo[times[i+1]]['anamean']['rmse_last'] - ekfevo[times[i+1]]['fgmean']['rmse_last'],
                    #                 color=colrmse_dots, length_includes_head=True, width=0.001, head_width=0.01)

                if not colbias_dots is None :
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['fgmean']['bias_last'],   s=ms, marker='x', color=colbias_dots )
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['anamean']['bias_last'],  s=ms, marker='o', color=colbias_dots )
                    plt.scatter( h+xp.visop_fcsttimes()[0]/60.0, stats[t][sorted(stats[t].keys())[0]]['bias'], s=ms, marker='s', color=colbias_dots )
                    #plt.gca().arrow( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['fgmean']['bias_last'],
                    #                 0, ekfevo[times[i+1]]['anamean']['bias_last'] - ekfevo[times[i+1]]['fgmean']['bias_last'],
                    #                 color=colbias_dots, length_includes_head=True, width=0.001, head_width=0.01)

                if not colspread_dots is None :
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['fgens']['spread_last'],  s=ms, marker='x', color=colspread_dots )
                    plt.scatter( h+xp.visop_fcsttimes()[-1]/60.0, ekfevo[times[i+1]]['anaens']['spread_last'], s=ms, marker='o', color=colspread_dots )

        add_label=False
        h += float(xp.settings['ASSINT'])/3600.0
    hmax = h
    hdur = hmax-hmin
    pltrng = (hmin-0.05*hdur,hmax+0.05*hdur)

    if finish_plot :
        plt.figure(1)
        plt.plot( pltrng, (0,0), '-.k', linewidth=0.3 )
        plt.ylim((-0.05,0.25))
        plt.xlim(pltrng)
        plt.xlabel('t [h] UTC')
        plt.ylabel('reflectance bias / spread / rmse')
        plt.legend(frameon=False)
        plt.title(titlestr)
        plt.grid()
        plt.savefig('error_evo_'+('lfcst_' if args.lfcst else '') \
                                +('noekf_' if args.no_ekf else '') \
                                +(style+'_' if style != '' else '') \
                                +channel+'_i2o-'+args.i2o_settings+'.png')


#-------------------------------------------------------------------------------
def clc_evo_plot( xp, times, clc, lfcst_clc, titlestr='' ) :

    plt.figure(1,figsize=(10,5))
    plt.clf()
    hmin = Time14(times[0]).dayhour()
    h = hmin
    add_label=True
    for i, t in enumerate(times) :
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['ensmean30'] for f in sorted(clc[t].keys()) ], 'k',               label='ensmean' if add_label else None  )
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['det30']     for f in sorted(clc[t].keys()) ], color='#999999',   label='det'     if add_label else None  )
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['obs30']     for f in sorted(clc[t].keys()) ], 'k',  linewidth=2, label='obs'     if add_label else None )
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['ensmean60'] for f in sorted(clc[t].keys()) ], 'k' )
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['det60']     for f in sorted(clc[t].keys()) ], color='#999999' )
        plt.plot( h+xp.visop_fcsttimes()/60.0, [ clc[t][f]['obs60']     for f in sorted(clc[t].keys()) ], 'k',  linewidth=2 )

        if not lfcst_clc is None :
            if t in lfcst_clc :
                lfcst_t_min = sorted(lfcst_clc[t].keys())
                lfcst_t_hour = Time14(t).dayhour() + array(lfcst_t_min)/60.0
                plt.plot( lfcst_t_hour, [ lfcst_clc[t][f]['ensmean30'] for f in lfcst_t_min ], ':', color='#990099' )
                plt.plot( lfcst_t_hour, [ lfcst_clc[t][f]['ensmean60'] for f in lfcst_t_min ], ':', color='#cc00cc' )
                plt.plot( lfcst_t_hour, [ lfcst_clc[t][f]['det30'] for f in lfcst_t_min ], color='#990099' )
                plt.plot( lfcst_t_hour, [ lfcst_clc[t][f]['det60'] for f in lfcst_t_min ], color='#cc00cc' )

        add_label=False
        h += float(xp.settings['ASSINT'])/3600.0
    hmax = h
    plt.plot( (hmin,hmax), (0,0), '-.k', linewidth=0.3 )
    plt.ylim((0,1))
    plt.legend(loc='upper right',title='domain cloud cover', frameon=False)
    plt.xlabel('t [h UTC]')
    plt.ylabel('cloud dover')
    plt.title(titlestr)
    plt.savefig('clc_evo_'+channel+'.png')


#-------------------------------------------------------------------------------
def multiscale_evo_plot( xp, times, mstat, lfcst_mstat, titlestr='', verbose=True ) :

    #import colorsys
    scales = mstat['scales']

    # evolution of rmse and drmse/dt of the ensemble mean for all scales
    plt.figure(1,figsize=(10,5))
    plt.clf()
    plt.figure(2,figsize=(10,5))
    plt.clf()

    hmin = Time14(times[0]).dayhour()
    h = hmin
    add_label=True
    add_label_lfcst=True
    for i, t in enumerate(times) :
        for iscl, s in enumerate(scales) :
            #hue = float(iscl)/len(scales)
            #col = colorsys.hls_to_rgb( hue, 0.6-0.2*exp(-((hue-0.35)/0.1)**2), 0.7 )
            col = colseq( iscl, len(scales) )
            km = s*6

            plt.figure(1)
            if not lfcst_mstat is None :
                if t in lfcst_mstat :
                    lfcst_t_min = sorted(lfcst_mstat[t].keys())
                    lfcst_t_hour = Time14(t).dayhour() + array(lfcst_t_min)/60.0
                    plt.plot( lfcst_t_hour, [ lfcst_mstat[t][f][s]['rmse'] for f in lfcst_t_min ],
                              color=col, linewidth=3, label='%dkm'%km if add_label_lfcst else None, zorder=2)
                    #plt.plot( lfcst_t_hour, [ lfcst_mstat[t][f][s]['rmse'] for f in lfcst_t_min ], '--', color=col )
                    if s == scales[-1] :
                        add_label_lfcst=False

                col='#cccccc'
                add_label = False

            plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['rmse'] for f in sorted(mstat[t].keys()) ],
                      color=col, label='%dkm'%km if add_label else None, linewidth=2, zorder=1 )


            plt.figure(2)
            rmses = [ mstat[t][f][s]['rmse'] for f in sorted(mstat[t].keys()) ]
            drmse = rmses[-1]-rmses[-2]
            dt    = (xp.visop_fcsttimes()[-1]-xp.visop_fcsttimes()[-2])/60.0
            plt.plot( (h+xp.visop_fcsttimes()[-2]/60.0,h+xp.visop_fcsttimes()[-1]/60.0), [drmse/dt,drmse/dt],
                      color=col, label='%dkm'%km if add_label else None, linewidth=2 )

        add_label=False
        h += float(xp.settings['ASSINT'])/3600.0
    hmax = h+3


    plt.figure(1)
    plt.plot( (hmin,hmax), (0,0), '-.k', linewidth=0.3 )
    plt.ylim((0,0.22))
    plt.xlim((hmin,hmax))
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel('t [h UTC]')
    plt.ylabel('rmse')
    plt.title(titlestr)
    plt.grid()
    if lfcst_mstat is None :
        plt.savefig('multiscale_rmse_evo_'+channel+'.png', bbox_inches='tight')
    else :
        plt.savefig('multiscale_rmse_evo_lfcst_'+channel+'.png', bbox_inches='tight')

    plt.figure(2)
    plt.plot( (hmin,hmax), (0,0), '-.k', linewidth=0.3 )
    plt.ylim((-0.05,0.1))
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel('t [h UTC]')
    plt.ylabel('drmse/dt')
    plt.title(titlestr)
    plt.grid()
    plt.savefig('multiscale_drmse_dt_evo_'+channel+'.png')

    #...................................................................................................................

    # rmse evolution of all ensemble member (indiv. plot for each scale)
    plt.figure(1,figsize=(10,5))
    for iscl, s in enumerate(scales) :
        km = s*6
        plt.clf()
        h = hmin
        add_label=True
        for i, t in enumerate(times) :
            plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['spread'] for f in sorted(mstat[t].keys()) ],
                      '--', color='#999999', label='spread' if add_label else None, linewidth=2 )

            for m in range(1,xp.n_ens+1) :
                plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['rmse_mem%03d'%m] for f in sorted(mstat[t].keys()) ],
                          color='#000000', alpha=0.2, linewidth=1,
                          label='rmse members' if (add_label and m==1) else None )

            if not lfcst_mstat is None :
                if t in lfcst_mstat :
                    lfcst_t_min = sorted(lfcst_mstat[t].keys())
                    lfcst_t_hour = Time14(t).dayhour() + array(lfcst_t_min)/60.0
                    for m in range(1,xp.n_ens+1) :
                        plt.plot( lfcst_t_hour, [ lfcst_mstat[t][f][s]['rmse_mem%03d'%m] for f in lfcst_t_min ], color='#0066cc', alpha=0.2 )

            plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['rmse'] for f in sorted(mstat[t].keys()) ],
                      color='r', label='rmse mean' if add_label else None, linewidth=2 )
            add_label=False
            h += float(xp.settings['ASSINT'])/3600.0

        plt.plot( (hmin,hmax), (0,0), '-.k', linewidth=0.3 )
        plt.ylim((0,0.3))
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('t [h UTC]')
        plt.ylabel('rmse')
        plt.title(titlestr + (" scale %dkm"%km))
        plt.grid()
        plt.savefig('multiscale_rmse_s%02d_evo_%s.png' % (s,channel))

    for iscl, s in enumerate([scales[0]]) : # bias is scale-independent...
        km = s*6
        plt.clf()
        h = hmin
        add_label=True
        for i, t in enumerate(times) :
            for m in range(1,xp.n_ens+1) :
                plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['bias_mem%03d'%m] for f in sorted(mstat[t].keys()) ],
                          color='#000000', alpha=0.2, linewidth=1 )
            plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['bias'] for f in sorted(mstat[t].keys()) ],
                      color='k', label='%dkm'%km if add_label else None, linewidth=2 )
            add_label=False
            h += float(xp.settings['ASSINT'])/3600.0

        plt.plot( (hmin,hmax), (0,0), '-.k', linewidth=0.3 )
        #plt.ylim((0,0.22))
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('t [h UTC]')
        plt.ylabel('bias')
        plt.title(titlestr)
        plt.grid()
        plt.savefig('multiscale_bias_s%02d_evo_%s.png' % (s,channel))

#-------------------------------------------------------------------------------
def multiscale_comparison_plot( xps, rss, cycle=True, colors=None, names=None, xrange=None, yrange=None, figsize=(10,5) ) :

    times  = rss[list(rss.keys())[0]]['times']
    scales = rss[list(rss.keys())[0]]['mstat']['scales']
    print('times ', times)
    print('scales ', scales)

    hmin = Time14(times[0]).dayhour()
    hmax = hmin

    # individual error/bias evolution plot for each scale

    for iscl, s in enumerate(scales) :
        km = s*6

        # create new plot
        #fig, ax = plt.subplots()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)


        for i,l in enumerate(xps.keys()) :
            if not names is None :
                expname = names[i]
            else :
                expname = xps[l].settings['EXPID']
            xp = xps[l]
            rs = rss[l]
            mstat = rs['mstat']
            if 'lfcst_mstat' in  rs :
                lfcst_mstat = rs['lfcst_mstat']
            else :
                lfcst_mstat = None
            if colors is None :
                col = colseq(i,len(xps))
            else :
                col = colors[i]

            add_label=True
            h = hmin
            for i, t in enumerate(times) :

                if not lfcst_mstat is None :
                    if t in lfcst_mstat :
                        lfcst_t_min = sorted(lfcst_mstat[t].keys())
                        lfcst_t_hour = Time14(t).dayhour() + array(lfcst_t_min)/60.0
                        ax.plot( lfcst_t_hour, [ lfcst_mstat[t][f][s]['rmse'] for f in lfcst_t_min ], color=col, linewidth=2,
                                 label=expname if add_label else None )
                        ax.plot( lfcst_t_hour, [ lfcst_mstat[t][f][s]['bias'] for f in lfcst_t_min ], '--', color=col, linewidth=2 )
                        add_label=False

                if cycle :
                    plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['rmse'] for f in sorted(mstat[t].keys()) ],
                              color=col, linewidth=1, alpha=0.5 )
                    plt.plot( h+xp.visop_fcsttimes()/60.0, [ mstat[t][f][s]['bias'] for f in sorted(mstat[t].keys()) ], '--',
                              color=col, linewidth=1, alpha=0.5 )

                #add_label=False
                h += float(xp.settings['ASSINT'])/3600.0
            hmax = maximum( h, hmax )

        if xrange is None :
            xmin = hmin-1
            xmax = hmax+2
        else :
            xmin, xmax = xrange
        ax.plot(   ( xmin, xmax ), (0,0), '--', color='#999999', linewidth=1, zorder=0 )
        ax.set_xlim( xmin, xmax )

        if not yrange is  None :
            ax.set_ylim( yrange[0], yrange[1] )
        else :
            ax.set_ylim( -0.05, 0.22 )

        ax.legend(loc='center right', frameon=False)
        ax.set_xlabel('t [h UTC]')
        ax.set_ylabel('ens. mean bias (dashed) and rmse (solid)')
        ax.set_title("reflectance, scale=%dkm"%km)
        ax.grid()
        fig.savefig('multiscale_rmse_comparison_s%02d_evo_%s.png' % (s,channel))


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate VISOP plots for KENDA experiment')

    parser.add_argument( '-E', '--evolution',   dest='error_evolution',   help='generate error evolution plots', action='store_true' )
    parser.add_argument(       '--no-ekf',      dest='no_ekf',            help='do not use ekf data in error evolution plots', action='store_true' )

    parser.add_argument( '-c', '--clc',         dest='clc_evolution',  help='plot clc evolution', action='store_true' )
    parser.add_argument( '-n', '--not-all-members',  dest='not_all_members',  help='do not generate plots for each of the members', action='store_true' )
    parser.add_argument( '-m', '--multiscale',   dest='multiscale',        help='multiscale metrics plots', action='store_true' )
    #parser.add_argument(       '--scales',      dest='scales', default='1,2,3,4,5,6,7'

    parser.add_argument( '-C', '--compare',     dest='compare',     help='generate comparison plots with data from all specified experiments',
                                                                    action='store_true' )
    parser.add_argument(       '--no-cycle',    dest='no_cycle',    help='show only results from long forecasts', action='store_true' )

    parser.add_argument( '-l', '--lfcst',       dest='lfcst',      help='take also long forecasts (not only cycling results) into account',   action='store_true' )

    parser.add_argument( '-T', '--transform',   dest='transform',  help='transform images to observations',  action='store_true' )
    parser.add_argument(       '--i2o-settings',   dest='i2o_settings',  help='visop_i2o settings string [default: the one from the experiment settings]',  default='experiment' )

    parser.add_argument(       '--channel',     dest='channel',    help='channel[s] (comma-separated list, deafult=VIS006)', default='VIS006' )

    parser.add_argument(       '--style',       dest='style',      help='plot style', default='' )
    parser.add_argument(       '--colors',      dest='colors',     help='comma-separated list of colors for the experiments', default='' )
    parser.add_argument(       '--names',       dest='names',      help='comma-separated list of names (use ,, for commas within the names)', default='' )
    parser.add_argument(       '--xrange',      dest='xrange',     type=str, help='xmin,xmax for the RMSE plots', default='' )
    parser.add_argument(       '--yrange',      dest='yrange',     type=str, help='ymin,ymax for the RMSE plots', default='' )
    parser.add_argument(       '--figsize',     dest='figsize',    type=str, help='witdh,height for the RMSE plots', default='' )

 #   parser.add_argument( '-V', '--variables',   dest='variables',   help='comma-separated list of variables  to be considered (default: all)', default='' )
 #   parser.add_argument( '-O', '--obs-types',   dest='obs_types',   help='comma-separated list of obs. types to be considered (default: all)', default='' )
    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default='' )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default='' )
    parser.add_argument( '-d', '--delta-time',  dest='delta_time',  help='time interval', default='' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='auto' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    

    parser.add_argument(       '--recompute',   dest='recompute',   help='ignore/overwrite cache files', action='store_true' )

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    if args.colors != '' :
        colors = args.colors.split(',')
    else :
        colors = None

    if args.names != '' :
        names = [ n.replace('_COMMA_',',') for n in args.names.replace(',,','_COMMA_').split(',') ]
    else :
        names = None

    if args.yrange != '' :
        yrange = [ float(x) for x in args.yrange.replace('\-','-').split(',') ]
        #yrange = [-0.06,0.16]
    else :
        yrange = None

    if args.xrange != '' :
        xrange = [ float(x) for x in args.xrange.replace('\-','-').split(',') ]
    else :
        xrange = None

    if args.figsize != '' :
        figsize = [ float(x) for x in args.figsize.split(',') ]
    else :
        figsize = None

    # process all log files

    xps = {}
    rss = {}
    for logfile in args.logfile :

        print()
        print("processing file %s ..." % logfile)

        xp = Experiment(logfile)
        xps[logfile] = xp
        print('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
               xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] ))

        titlestr = "%s T_win=%4.2fh e_o=%s h_loc=%s superobb=%s " % (
                    xp.settings['EXPID'],
                    float(xp.settings['ASSINT'])/3600.0,
                    xp.settings['VISOP_ERROR'],
                    xp.settings['VISOP_HLOC'],
                    xp.settings['VISOP_SUPEROBB'])

        # set some default values
        if args.output_path != '' :
            if args.output_path != 'auto' :
                output_path = args.output_path+'/'
            else :
                output_path = xp.settings['PLOT_DIR']+'/visop/'
                if not os.path.exists(output_path) :
                    os.system('mkdir '+output_path)
        else :
            output_path = ''

        # determine time range
        if args.start_time != '' :
            start_time = args.start_time
        else :
            for t in xp.veri_times :
                t14 = Time14(t)
                if t14.hour() > int(xp.settings['VISOP_START_HOUR']) and t14.hour() < int(xp.settings['VISOP_STOP_HOUR']) :
                    start_time = t
                    break
                if t14.hour() == int(xp.settings['VISOP_START_HOUR']) :
                    if t14.minute() >= int(xp.settings['VISOP_START_MINUTE']) :
                        start_time = t
                        break
                if t14.hour() == int(xp.settings['VISOP_STOP_HOUR']) :
                    if t14.minute() <= int(xp.settings['VISOP_STOP_MINUTE']) :
                        start_time = t
                        break
        print('selecting start time = ', start_time)

        if args.end_time != '' :
            end_time = args.end_time
        else :
            end_time = xp.fcst_start_times[-1]
            end_time = end_time[:8] + ("%02d" % (int(xp.settings['VISOP_STOP_HOUR'])-1)) + end_time[10:]
        print('selecting end time = ', end_time)

        if args.delta_time != '' :
            delta_time = args.delta_time
        else :
            delta_time = xp.settings['ASSINT']
        times = time_range( start_time, end_time, delta_time  )
        print('fcst start times: ', times)

        # loop over channels and times, read data and generate plots
        for channel in args.channel.split(',') :

            # get data
            rs = get_precprod_lfcst_stats( xp, times=times, recompute=args.recompute )
            rss[logfile] = rs

            if not args.compare :
                if args.error_evolution : stats = rs['stats']
                if args.clc_evolution   : clc   = rs['clc']
                if args.multiscale      : mstat = rs['mstat']
                if args.lfcst :
                    if args.error_evolution : lfcst_stats = rs['lfcst_stats']
                    if args.clc_evolution   : lfcst_clc   = rs['lfcst_clc']
                    if args.multiscale      : lfcst_mstat = rs['lfcst_mstat']

                # generate plots

                if args.error_evolution :
                    error_evo_plot( xp, times, stats, lfcst_stats if args.lfcst else None, titlestr=titlestr, style=args.style )

                if args.clc_evolution :
                    clc_evo_plot( xp, times, clc, lfcst_clc if args.lfcst else None, titlestr=titlestr )

                if args.multiscale :
                    multiscale_evo_plot( xp, times, mstat, lfcst_mstat if args.lfcst else None, titlestr=titlestr )

        # end of channel loop ......................................................................
    # end of logfile loop ..........................................................................

    if args.compare :
        if args.error_evolution :
            for i, l in enumerate(sorted(rss.keys())) :
                error_evo_plot( xps[l], rss[l]['times'], rss[l]['stats'], rss[l]['lfcst_stats'] if args.lfcst else None,
                                ekf=False,
                                start_plot = True if i == 0 else False,
                                finish_plot = True if i == len(rss)-1 else False,
                                color = colseq(i,len(rss)),
                                style=args.style )
            for i, l in enumerate(sorted(rss.keys())) :
                print(i, l)

        if args.multiscale :
            multiscale_comparison_plot( xps, rss, cycle = not args.no_cycle, colors=colors, names=names,
                                        xrange=xrange, yrange=yrange, figsize=figsize )

