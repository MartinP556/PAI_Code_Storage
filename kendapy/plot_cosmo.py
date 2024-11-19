#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P L O T _ C O S M O
#  plot COSMO fields
#
#  2017.6 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os, argparse
from numpy import *

from kendapy.time14 import Time14, time_range

def plot_field( cs, quan, fname ) :

    if quan == 'TOT_PREC' :
        vmin = -2
        vmax = 2
        logscl=True
    elif quan == 'TQV' :
        vmin = 0.5
        vmax = 2.0
        logscl=True
    else :
        vmin = None
        vmax = None
        logscl = False

    if logscl :
        qval = log10( cs[quan] + 10**vmin if not vmin is None else 0 )
    else :
        qval = cs[quan]

    plt.figure(1,figsize=(8,6))
    plt.clf()
    plt.imshow( qval, origin='lower', vmin=vmin, vmax=vmax )
    plt.colorbar()
    ny, nx = qval.shape
    plt.text( 0.03*nx, 0.97*ny, '%s [%s]' % (quan, cs.meta[quan]['units']),
             color='w', fontsize=10, ha='left', va='top' )
    plt.savefig( fname, bbox_inches='tight' )

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    from kendapy.experiment import Experiment

    parser = argparse.ArgumentParser(description='Generate COSMO plots for KENDA experiment')

    parser.add_argument( '-m', '--member',      dest='member',      help='member index (-1=obs,0=det,1,2,...N_ENS)', default='1' )

    parser.add_argument( '-V', '--variables',   dest='variables',   help='comma-separated list of variables to be plotted', default='' )

    parser.add_argument( '-l', '--lfcst-start', dest='lfcst_start', help='start time of long forecast to be used (not set -> use cycling results)', default='' )

    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time (for long forecasts: relative)', default='' )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time   (for long forecasts: relative)', default='' )
    parser.add_argument( '-d', '--delta-time',  dest='delta_time',  help='time interval', default='' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='auto' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    # process all log files

    xps = {}
    for logfile in args.logfile :

        print()
        print(("processing file %s ..." % logfile))

        xp = Experiment(logfile)
        xps[logfile] = xp
        print(('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
               xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] )))
        if args.lfcst_start != '' :
            print(('specified lfcst_start = ', args.lfcst_start))
            print(('available lfcst_start values : ', xp.lfcst_start_times))
            if not args.lfcst_start in xp.lfcst_start_times :
                raise ValueError('lfcst_start is invalid')
            print(('lfcst output times : ', xp.lfcst_sfc_output_times_min))

        if args.member == 'all' :
            members = list(range( 1, xp.n_ens+1))
        else :
            members = list(map( int, args.member.split(',') ))

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
                output_path = xp.settings['PLOT_DIR']+'/cosmo/'
                if not os.path.exists(output_path) :
                    os.system('mkdir '+output_path)
        else :
            output_path = ''

        # determine time range

        if args.start_time != '' :
            start_time = args.start_time
        else :
            if args.lfcst_start != '' :
                start_time = xp.lfcst_sfc_output_times_min[0]
            else :
                start_time = xp.veri_times[0]

        if args.end_time != '' :
            end_time = args.end_time
        else :
            if args.lfcst_start != '' :
                end_time = xp.lfcst_sfc_output_times_min[-1]
            else :
                end_time =  xp.veri_times[-1]

        if args.delta_time != '' :
            delta_time = args.delta_time
        else :
            if args.lfcst_start != '' :
                delta_time = xp.lfcst_sfc_output_times_min[1]-xp.lfcst_sfc_output_times_min[0]
            else :
                delta_time = xp.settings['ASSINT']

        if args.lfcst_start != '' :
            times = arange( int(start_time), int(end_time)+1, delta_time )
        else :
            times = time_range( start_time, end_time, delta_time  )

        print(('times', times))
        for t in times :
            for m in members :
                if args.lfcst_start != '' :
                    raise ValueError('FIXME time in get_cosmo')
                    cs = xp.get_cosmo( args.lfcst_start, prefix='lfff', suffix='sfc', member=m )
                    fcst_id = 'lfcst'+args.lfcst_start
                    tstr = str(Time14(t*60))
                else :
                    cs = xp.get_cosmo( t, prefix='lfff', suffix='sfc', member=m )
                    fcst_id = 'cycle'+t
                    tstr = ''

                #cs.list_variables()

                for v in args.variables.split(',') :
                    fname = output_path + fcst_id + '_' + tstr + '_' + v + ('_mem%03d'%m ) + '.png'
                    plot_field( cs, v, fname )
