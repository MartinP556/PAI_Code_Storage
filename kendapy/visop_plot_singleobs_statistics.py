#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . V I S O P _ P L O T _ S I N G L E O B S _ S T A T I S T I C S
#  plot common statistics for several single observation KENDA+VISOP experiments
#
#  2018.3 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

import sys, os, argparse, pickle
import time as tttime
from numpy import *
from scipy import stats
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.colseq import colseq
from kendapy.visop_fcst import get_visop_fcst_stats
from kendapy.visop_plot_images import plot_reflectance
from kendapy.cosmo_state import CosmoState
from kendapy.binplot import binplot

def plot_singleobs_statistics( sobs, args, figsize=(3.5,3.5) ) :

    if args.mark != '' :
        mark    = dict( [ (p.split(':')[0],p.split(':')[2]) for p in args.mark.split(',') ] )
        marknum = dict( [ (p.split(':')[0],p.split(':')[1]) for p in args.mark.split(',') ] )
        print('mark colors  : ', mark)
        print('mark numbers : ', marknum)
    else :
        mark = {}
        marknum = {}

    obsids = sorted(sobs.keys())
    markcol = [ mark[i] if i in mark else 'k' for i in obsids ]

    for obsid in sobs :
        print(obsid, '<--' if obsid in mark else '')

    # tauw, taui

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    for i in marknum :
        ax.text( sobs[i]['tauw_mean_fg']+2, sobs[i]['taui_mean_fg']-0.1, marknum[i], ha='left', va='top', fontsize=12, color=mark[i], zorder=10 )
    ax.scatter( [ sobs[i]['tauw_mean_fg'] for i in obsids ],
                [ sobs[i]['taui_mean_fg'] for i in obsids ],
                color=[ '{:.2f}'.format(sobs[i]['obs']) for i in obsids ],
                edgecolor=[ mark[i] if i in mark else 'k' for i in obsids ],
                s=50, cmap='gray' )
    if args.label :
        for i in obsids :
            ax.text( sobs[i]['tauw_mean_fg']+1, sobs[i]['taui_mean_fg'], i.replace('.92s','/').replace('TNSb_','#'),
                     ha='left', va='center', fontsize=8 )

    #if args.file_type == 'png' :
    #    for i in obsids :
    #        ax.text( sobs[i]['fgmean'] - sobs[i]['obs'], abs(sobs[i]['anamean_real'] - sobs[i]['obs']) - abs(sobs[i]['fgmean'] - sobs[i]['obs']),
    #                 i.replace('.92s','/').replace('TNSb_','#'), fontsize=8 )
    #ax.set_xlim((0.1,100))
    #ax.set_ylim((0.01,10))
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_w$')
    ax.set_ylabel(r'$\tau_i$')
    ax.grid()
    #ax.legend(frameon=False,loc='lower right')
    #ax.set_title(['FG mean ','ANA mean'][iveri]+mvar)
    fig.savefig( 'tauw_vs_taui.'+args.file_type, bbox_inches='tight' )
    plt.close(fig)


    # reflectance error reduction   |B-O| - |A-O|

    fgdep       = [ sobs[i]['fgmean'] - sobs[i]['obs'] for i in obsids ]
    derefl      = [ abs(sobs[i]['fgmean'] - sobs[i]['obs']) - abs(sobs[i]['anamean']      - sobs[i]['obs']) for i in obsids ]
    derefl_real = [ abs(sobs[i]['fgmean'] - sobs[i]['obs']) - abs(sobs[i]['anamean_real'] - sobs[i]['obs']) for i in obsids ]

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    k_marked = []
    for k, i in enumerate(obsids) :
        ax.plot( [fgdep[k]]*2, (derefl[k],derefl_real[k]), color=markcol[k], zorder=9 if i in marknum else 1 )
        if i in marknum :
            ax.text( fgdep[k], derefl[k]+0.035, marknum[i], ha='center', va='top', fontsize=12, color=mark[i], zorder=10 )
            k_marked.append(k)

    ax.scatter( fgdep, derefl, color='w', edgecolor=markcol, label='linear' )
    ax.scatter( fgdep, derefl_real, color=markcol, edgecolor=markcol, label='nonlinear' )

    ax.scatter( [fgdep[k] for k in k_marked], [derefl[k]      for k in k_marked], color='w',
                edgecolor=[markcol[k] for k in k_marked], zorder=10 )
    ax.scatter( [fgdep[k] for k in k_marked], [derefl_real[k] for k in k_marked], color=[markcol[k] for k in k_marked],
                edgecolor=[markcol[k] for k in k_marked], zorder=10 )

    if args.label :
        for i in obsids :
            ax.text( sobs[i]['fgmean'] - sobs[i]['obs'],
                     abs(sobs[i]['fgmean'] - sobs[i]['obs']) - abs(sobs[i]['anamean_real'] - sobs[i]['obs']) - 0.01,
                     i.replace('.101s','/').replace('TNSc_','#'), fontsize=5, zorder=1, ha='left', va='top' )
    xlm = array(ax.get_xlim())
    xmx = abs(xlm).max()
    ax.set_xlim((-xmx*1.1,xmx*1.1))
    ax.plot( ax.get_xlim(), (0,0), '--k', linewidth=0.5 )
    ylm = ax.get_ylim()
    ax.plot( (0,0), ylm, '--k', linewidth=0.5 )
    ax.set_ylim(ylm)
    ax.set_xlabel(r'${\cal R}_B - {\cal R}_O$')
    ax.set_ylabel(r'$|{\cal R}_B - {\cal R}_O| - |{\cal R}_A - {\cal R}_O|$')
    ax.legend(frameon=True,loc='upper center')
    #ax.set_title(['FG mean ','ANA mean'][iveri]+mvar)
    fig.savefig( 'refl_error_reduction.'+args.file_type, bbox_inches='tight' )
    plt.close(fig)

    # change in T / RH errors

    meancol='#00cc00'

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    for i in marknum :
        ax.text( sobs[i]['T_delta_rmse']+0.01, sobs[i]['RH_delta_rmse'], marknum[i], ha='left', va='center', fontsize=12, color=mark[i], zorder=10 )
    ax.scatter( [ sobs[i]['T_delta_rmse']  for i in obsids ],
                [ sobs[i]['RH_delta_rmse'] for i in obsids ], color=markcol, edgecolor=None )
    if args.file_type == 'png' :
        for i in obsids :
            ax.text( sobs[i]['T_delta_rmse'], sobs[i]['RH_delta_rmse'],
                     i.replace('.92s','/').replace('TNSb_','#'), fontsize=8 )

    det = array([ sobs[i]['T_delta_rmse']  for i in obsids ])
    der = array([ sobs[i]['RH_delta_rmse']  for i in obsids ])
    ax.plot( (det.mean()-det.std()/2,det.mean()+det.std()/2), [der.mean()]*2, color=meancol )
    ax.plot( [det.mean()]*2, (der.mean()-der.std()/2,der.mean()+der.std()/2), color=meancol )

    ax.set_xlabel(r'$\Delta\epsilon_T$ [K]')
    ax.set_ylabel(r'$\Delta\epsilon_{RH}$')

    #xlm = array(ax.get_xlim())
    #xmx = abs(xlm).max()
    #ax.set_xlim((-xmx*1.1,xmx*1.1))
    xlm = array(ax.get_xlim())

    #ylm = array(ax.get_ylim())
    #ymx = abs(ylm).max()
    #ax.set_ylim((-ymx*1.1,ymx*1.1))
    ylm = array(ax.get_ylim())

    ax.plot( xlm, (0,0), ':k', linewidth=0.5 )
    ax.plot( (0,0), ylm, ':k', linewidth=0.5 )
    ax.set_xlim(xlm)
    ax.set_ylim(ylm)

    fig.savefig( 'T_RH_delta_rmse.'+args.file_type, bbox_inches='tight' )
    plt.close(fig)


    # change in U / V errors

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    for i in marknum :
        ax.text( sobs[i]['U_delta_rmse']-0.01, sobs[i]['V_delta_rmse'], marknum[i], ha='right', va='center', fontsize=12, color=mark[i], zorder=10 )
    ax.scatter( [ sobs[i]['U_delta_rmse']  for i in obsids ],
                [ sobs[i]['V_delta_rmse'] for i in obsids ], color=markcol, edgecolor=None )
    if args.file_type == 'png' :
        for i in obsids :
            ax.text( sobs[i]['U_delta_rmse'], sobs[i]['V_delta_rmse'],
                     i.replace('.92s','/').replace('TNSb_','#'), fontsize=8 )

    deu = array([ sobs[i]['U_delta_rmse']  for i in obsids ])
    dev = array([ sobs[i]['V_delta_rmse']  for i in obsids ])
    ax.plot( (deu.mean()-deu.std()/2,deu.mean()+deu.std()/2), [dev.mean()]*2, color=meancol )
    ax.plot( [deu.mean()]*2, (dev.mean()-dev.std()/2,dev.mean()+dev.std()/2), color=meancol )

    ax.set_xlabel(r'$\Delta\epsilon_U$ [m/s]')
    ax.set_ylabel(r'$\Delta\epsilon_V$ [m/s]')

    #xlm = array(ax.get_xlim())
    #xmx = abs(xlm).max()
    #ax.set_xlim((-xmx*1.1,xmx*1.1))
    xlm = array(ax.get_xlim())

    #ylm = array(ax.get_ylim())
    #ymx = abs(ylm).max()
    #ax.set_ylim((-ymx*1.1,ymx*1.1))
    ylm = array(ax.get_ylim())

    ax.plot( xlm, (0,0), ':k', linewidth=0.5 )
    ax.plot( (0,0), ylm, ':k', linewidth=0.5 )
    ax.set_xlim(xlm)
    ax.set_ylim(ylm)

    fig.savefig( 'U_V_delta_rmse.'+args.file_type, bbox_inches='tight' )
    plt.close(fig)


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate VISOP plots for KENDA experiment')

    # parser.add_argument( '-E', '--evolution',   dest='error_evolution',   help='generate error evolution plots', action='store_true' )
    # parser.add_argument(       '--no-ekf',      dest='no_ekf',            help='do not use ekf data in error evolution plots', action='store_true' )
    #
    # parser.add_argument( '-o', '--obsloc',      dest='plot_obsloc',      help='plot observation locations',         action='store_true' )
    # parser.add_argument( '-S', '--stamps',      dest='stamps',           help='generate stamp collection plot for each observation', action='store_true' )
    # parser.add_argument( '-R', '--radio',       dest='radio',            help='include radio sond information', action='store_true' )
    #
    # parser.add_argument( '-i', '--increments',  dest='plot_increments',  help='plot fg + ana state and increments', action='store_true' )
    # parser.add_argument( '-P', '--points',      dest='points',           help='indices of single obs. points for which increments are to be plotted', default='all' )
    # parser.add_argument( '-L', '--location',    dest='location',         help='plot results for observation closest to the location specified as <lon>,<lat>', default='' )
    #
    # parser.add_argument( '-c', '--clc',         dest='clc_evolution',  help='plot clc evolution', action='store_true' )
    # parser.add_argument( '-n', '--not-all-members',  dest='not_all_members',  help='do not generate plots for each of the members', action='store_true' )
    #parser.add_argument( '-m', '--multiscale',   dest='multiscale',        help='multiscale metrics plots', action='store_true' )
    #parser.add_argument(       '--scales',      dest='scales', default='1,2,3,4,5,6,7'

#    parser.add_argument( '-C', '--compare',     dest='compare',     help='generate comparison plots with data from all specified experiments',                                                                    action='store_true' )
#    parser.add_argument(       '--no-cycle',    dest='no_cycle',    help='show only results from long forecasts', action='store_true' )

#    parser.add_argument( '-l', '--lfcst',       dest='lfcst',      help='take also long forecasts (not only cycling results) into account',   action='store_true' )

#    parser.add_argument( '-T', '--transform',   dest='transform',  help='transform images to observations',  action='store_true' )
#    parser.add_argument(       '--i2o-settings',   dest='i2o_settings',  help='visop_i2o settings string [default: the one from the experiment settings]',  default='experiment' )

#    parser.add_argument(       '--channel',     dest='channel',    help='channel[s] (comma-separated list, deafult=VIS006)', default='VIS006' )

#   parser.add_argument( '-V', '--variables',   dest='variables',   help='comma-separated list of variables  to be considered (default: all)', default='' )
#   parser.add_argument( '-O', '--obs-types',   dest='obs_types',   help='comma-separated list of obs. types to be considered (default: all)', default='' )
#    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default='' )
#    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default='' )
#   --mark 0605.92s10TNSb_3:r,0529.92s16TNSb_3:g,0605.92s16TNSb_3:b,0605.92s10TNSb_2:y
    parser.add_argument( '-m', '--mark',        dest='mark',        help='mark the sepcified obs using the specified colors', default='' )
    parser.add_argument( '-l', '--label',       dest='label',       help='label all observaations', action='store_true' )
    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='auto' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    

    parser.add_argument( 'obsfiles', metavar='obsfiles', help='observation files (*/observation.pickle)', nargs='*' )
    args = parser.parse_args()

    # process all log files

    sobs = {}
    for obsfile in args.obsfiles :
        print('reading %s ...' % obsfile)
        with open(obsfile,'r') as f :
            sobs.update( pickle.load(f) )

    plot_singleobs_statistics( sobs, args )
