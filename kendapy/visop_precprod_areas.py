#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . V I S O P _ P R E C P R O D _ A R E A S
#  compute and plot time evolution of domain area fractions where reflectance or precipitation exceed certain thresholds
#
#  2018.1 L.Scheck

from __future__ import absolute_import, division, print_function
from numpy import *
import numpy.ma as ma

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.precprod_ens import get_precprod_ens
from kendapy.cosmo_state import cosmo_grid
from kendapy.time14 import Time14

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    import sys
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from copy import copy
    from kendapy.cosmo_state import cosmo_grid

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate precipitation')

    parser.add_argument( '-d', '--day',       dest='day',   help='day, e.g. 0605 (default)', default='0605' )
    parser.add_argument( '-t', '--threshold', dest='level', help='precip threshold[s]',      default='1.0' )
    parser.add_argument( '-s', '--scale',     dest='win',   help='scale[s]',                 default='11' )

    parser.add_argument( '-X', '--experiment', dest='experiment', help='path to experiment log file', default='' )
    parser.add_argument( '-S', '--start-time', dest='start_time', help='start time, e.g. 201606050600', default='' )
    parser.add_argument( '-E', '--end-time',   dest='end_time',   help='end   time, e.g. 201606051800', default='' )

    parser.add_argument( '-o', '--output-time', dest='output_time', help='output time [minutes, relative to start time]', type=int, default=0 )

    parser.add_argument( '-f', '--fss',     dest='plot_fss',     help='plot fss evolution', action='store_true' )
    parser.add_argument(       '--mean-only',  dest='mean_only',  help='plot only mean, not all members', action='store_true' )
    parser.add_argument( '-F', '--fss-dwd', dest='plot_fss_dwd', help='reproduce dwd fss plots', action='store_true' )
    parser.add_argument( '-m', '--member',  dest='member',       help='member index (default: -1 -> use full ensemble)', type=int, default=-1 )
    parser.add_argument( '-B', '--belscl',  dest='plot_belscl',  help='plot believable scale evolution', action='store_true' )
    parser.add_argument(       '--belscl-min', dest='belscl_min',  help='mininum believable scale to be plotted', type=float, default=0 )
    parser.add_argument(       '--belscl-max', dest='belscl_max',  help='mininum believable scale to be plotted', type=float, default=200 )
    parser.add_argument(       '--belscl-log', dest='belscl_log',  help='generate logarithmic plot of believable scale (default: linear)', action='store_true' )

    parser.add_argument( '-P', '--preciprates', dest='preciprates', help='comma-separated list of precipitation rates [in units of mm/h]', default='0.1,1,5' )
    parser.add_argument( '-R', '--reflectances',  dest='reflectances',  help='comma-separated list of reflectance threshold values', default='0.3,0.5,0.7' )

    parser.add_argument( '-r', '--resolution',  dest='tresmin',       help='time resolution in minutes (default 60)', type=int, default=60 )

    parser.add_argument( '-l', '--lfcst',       dest='lfcst',      help='take also long forecasts (not only cycling results) into account',   action='store_true' )
    #parser.add_argument( '-I', '--input-path',  dest='input_path',  help='path to the directory containing the log files', default='' )
    parser.add_argument( 'logfile', metavar='logfile', help='log file name[s] (will be prepended by --input-path, if specified) or experiment id', nargs='*' )
    args = parser.parse_args()

    xps = [ Experiment(l) for l in args.logfile ]
    xp = xps[0]
    start_times = xp.get_fcst_start_times( tmin=args.start_time, tmax=args.end_time, lfcst=args.lfcst )
    output_times_min = xp.lfcst_visop_output_times_min if args.lfcst else xp.visop_output_times_min
    tresmin = output_times_min[1] - output_times_min[0]

    # loop over forecast start and output times
    t_obs = []
    precip = {}
    preciprates = args.preciprates.split(',')
    seviri = {}
    reflectances = args.reflectances.split(',')
    for tstart in start_times :
        for tout in output_times_min[:-1] :
            t_obs.append( str(Time14(tstart) + Time14(tout*60)) )

            # get precip product
            pp = get_precprod_ens( xp, tstart, tout, tresmin=tresmin, lfcst=args.lfcst, verbose=False, copy_mask=True, clear_mask=False )

            # compute fraction of radar-covered area where the precip rate exceeds a threshold value
            n_valid = pp['obs'].size - count_nonzero(pp['mask'])
            precip[t_obs[-1]] = {}
            precip[t_obs[-1]]['areafrac'] = {}
            for pr in preciprates :
                idcs = where( pp['obs'] > float(pr) )
                precip[t_obs[-1]]['areafrac'][pr] = len(idcs[0])/float(n_valid)
            print( t_obs[-1], 'PRECIP ', precip[t_obs[-1]]['areafrac'] )

            # get visop ensemble
            seviri[t_obs[-1]] = {}
            seviri[t_obs[-1]]['areafrac'] = {}
            vens = xp.get_visop( tstart, fcsttime=tout, lfcst=args.lfcst, preload=True )
            for refl in reflectances :
                seviri[t_obs[-1]]['areafrac'][refl] = vens.areafrac_obs(thres=float(refl))
                #seviri[t_obs[-1]]['areafrac_mean'][refl] = vens.areafrac_ens(thres=float(refl)).mean()
            print( t_obs[-1], 'REFL ', seviri[t_obs[-1]]['areafrac'] )

    fig, ax = plt.subplots(figsize=(3,5))
    hour = [ Time14(t).dayhour() for t in t_obs ] # e.g. 6:30 UTC = 6.5
    for pr in preciprates :
        pfrac = array([ precip[t]['areafrac'][pr] for t in t_obs ])
        ax.plot( hour, pfrac, 'b' )
        ax.text( hour[0], pfrac[0], " {:.2f}".format(pfrac.mean()), fontsize=8, color='b' )
    for refl in reflectances :
        rfrac = array([ seviri[t]['areafrac'][refl] for t in t_obs ])
        ax.plot( hour, rfrac, 'r' )
        ax.text( hour[-1], rfrac[-1], "{:.2f} ".format(rfrac.mean()), ha='right', fontsize=8, color='r' )
    ax.set_ylabel('area fraction')
    ax.set_xlabel('t [h UTC]')
    ax.set_ylim((0,1))
    fig.savefig(xp.expid+'_areafrac.png')



