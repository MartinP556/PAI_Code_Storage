#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . V I S O P _ P L O T _ S I N G L E O B S
#  plot statistics for single observation KENDA+VISOP experiments
#
#  2017.7 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Rectangle

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

#-----------------------------------------------------------------------------------------------------------------------
def summary_profile_plot( iobs, xp, lat, lon, obs, fgens, fgmean, anaens, anaens_real, anamean, anamean_real,
                          fgmean_so, anamean_so, fgspread_so, anaspread_so,
                          tauw_mean_fg, tauw_spread_fg, tauw_mean_ana, tauw_spread_ana,
                          taui_mean_fg, taui_spread_fg, taui_mean_ana, taui_spread_ana,
                          clc_tot_mean_fg, clc_tot_spread_fg, clc_tot_mean_ana, clc_tot_spread_ana,
                          ekf_temp, temp_rep_close_to_obs, temp_errors,
                          obsdir, time, args, short=True, dist_sep=True ) :

    # setup layout .....................................................................................................

    col_a = 'r'
    col_b = '#0000dd'
    col_i = '#990099'
    col_o = 'k'
    #col_temp_a = '#cc6600'
    #col_temp_b = '#0099cc'
    col_temp_b = '#0000dd'
    col_temp_a = 'r'
    col_temp_s = 'k'
    pmin=200

    lw=2.0
    lwn=1.0

    if short :


        if dist_sep : # reflectance distributions in separate plot
            fig = plt.figure( figsize=(8,2.5) )

            print( 'Plotting reflectance distributions in separate figures...' )
            mvars = ['QC','QI','CLC']
            gs2   = gridspec.GridSpec(nrows=1, ncols=len(mvars), left=0.06, right=0.59, hspace=0.0, wspace=0.0 )
            rvars = ['RH','T']
            gs3   = gridspec.GridSpec(nrows=1, ncols=len(rvars), left=0.61, right=0.98, hspace=0.0, wspace=0.0 )

        else :
            fig = plt.figure( figsize=(12,3) )
            gs1 = gridspec.GridSpec(nrows=1, ncols=1, left=0.05, right=0.17)

            #mvars = ['QC','QI','CLC']
            #gs2   = gridspec.GridSpec(nrows=1, ncols=len(mvars), left=0.23, right=0.58, hspace=0.0, wspace=0.0 )
            #rvars = ['RH','U','T']
            #gs3   = gridspec.GridSpec(nrows=1, ncols=len(rvars), left=0.60, right=0.98, hspace=0.0, wspace=0.0 )

            mvars = ['QC','QI','CLC']
            gs2   = gridspec.GridSpec(nrows=1, ncols=len(mvars), left=0.23, right=0.65, hspace=0.0, wspace=0.0 )
            rvars = ['RH','T']
            gs3   = gridspec.GridSpec(nrows=1, ncols=len(rvars), left=0.67, right=0.98, hspace=0.0, wspace=0.0 )

    else :
        fig = plt.figure( figsize=(19,4) )

        gs1 = gridspec.GridSpec(nrows=1, ncols=1, left=0.05, right=0.15)

        #gs2   = gridspec.GridSpec(nrows=1, ncols=7, left=0.20, right=0.7, hspace=0.0, wspace=0.0 )
        #mvars = ['QC','QI','CLC','RELHUM','U','V','T']
        #gs3   = gridspec.GridSpec(nrows=1, ncols=4, left=0.75, right=0.98, hspace=0.0, wspace=0.0 )
        #rvars = ['RH','U','V','T']

        mvars = ['QC','QI','CLC','RELHUM','T']
        gs2   = gridspec.GridSpec(nrows=1, ncols=len(mvars), left=0.20, right=0.6, hspace=0.0, wspace=0.0 )
        rvars = ['RH','U','V','T']
        gs3   = gridspec.GridSpec(nrows=1, ncols=len(rvars), left=0.65, right=0.98, hspace=0.0, wspace=0.0 )

    print( '***** SUMMARY_PROFILE_PLOT: ', mvars, rvars  )

    axs = {}
    if not dist_sep :
        axs['REFL'] = fig.add_subplot(gs1[:,:])

    maxs = {}
    for i, mvar in enumerate(mvars) :
        maxs[mvar] = fig.add_subplot(gs2[:,i], sharey = maxs[mvars[0]] if i > 0 else None )
        if i > 0 :
            plt.setp(maxs[mvar].get_yticklabels(), visible=False)

    raxs = {}
    for i, rvar in enumerate(rvars) :
        raxs[rvar] = fig.add_subplot(gs3[:,i], sharey = raxs[rvars[0]] if i > 0 else None )
        if i > 0 :
            plt.setp(raxs[rvar].get_yticklabels(), visible=False)


    # plot reflectance ensembles .......................................................................................

    nens = fgens.shape[0]
    dx = 0.3*( arange(nens) - (nens-1)*0.5 ) / ((nens-1)*0.5)
    symsz = 8
    alphasym = 0.5

    #...also as separate plt...
    Rfig, Rax = plt.subplots( figsize=(2.5,2.5) )
    if dist_sep :
        axxes = [Rax]
    else :
        axxes = [axs['REFL'], Rax]
    for axx in axxes :

        if fgmean[iobs] > 0.2 : # makes only sense for paper cases, sorry...
            ytxt = 0.04
        else :
            ytxt = 0.91

        axx.text( 0.65, ytxt, xp.expid.replace('0605','June 5, ').replace('0529','May 29, ').replace('.101s',' ').replace('TNSe','UTC, ') \
                              + ('%.1f$^{\circ}$E/%.1f$^{\circ}$N' % (lon[iobs],lat[iobs])), fontsize=10 ) # +'/'+str(iobs)
        #axx.text( 0.75, 0.85, 'lat=%.1f lon=%.1f' % (lat[iobs],lon[iobs]), fontsize=10 )

        c=col_o
        axx.barh( obs[iobs], 3, float(xp.settings['VISOP_ERROR']), 0.5, align='center', color=c,
                  alpha=0.1, edgecolor='none' )
        axx.plot( (0.5,3.5), [obs[iobs]]*2, color=c, linewidth=lw )
        axx.text( 0.6, obs[iobs]+0.01, 'O', color=c )

        c=col_b
        axx.scatter( [1]*fgens.shape[0] + dx, fgens[:,iobs], color=c, alpha=alphasym, s=symsz, edgecolors='none' )
        axx.plot( (0.5,1.5), [fgmean[iobs]]*2, color=c, linewidth=lw, solid_capstyle='butt' )
        sprd = fgens[:,iobs].std()
        axx.add_patch( Rectangle((1-0.4, fgmean[iobs]-sprd/2), 0.8, sprd, edgecolor=c, facecolor='none'))

        c='#990000'
        axx.scatter( [2]*fgens.shape[0] + dx, anaens[:,iobs], color=c, alpha=alphasym, s=symsz, edgecolors='none' )
        axx.plot( (1.5,2.5), [anamean[iobs]]*2, color=c, linewidth=lw, solid_capstyle='butt' )
        sprd = anaens[:,iobs].std()
        axx.add_patch( Rectangle((2-0.4, anamean[iobs]-sprd/2), 0.8, sprd, edgecolor=c, facecolor='none'))

        c=col_a
        axx.scatter( [3]*fgens.shape[0] + dx, anaens_real[:,iobs], color=c, alpha=alphasym, s=symsz, edgecolors='none' )
        axx.plot( (2.5,3.5), [anamean_real[iobs]]*2, color=c, linewidth=lw, solid_capstyle='butt' )
        sprd = anaens_real[:,iobs].std()
        axx.add_patch( Rectangle((3-0.4, anamean_real[iobs]-sprd/2), 0.8, sprd, edgecolor=c, facecolor='none'))

        axx.set_ylim((0,1.0))
        axx.set_ylabel('reflectance')
        axx.set_xlim((0.5,3.5))
        axx.set_xticks([1,2,3])
        axx.set_xticklabels(['B','A','A*'])

    Rfig.savefig( "%s/refldist_paper_%s_obs%d.%s" % (obsdir,time,iobs,args.file_type), bbox_inches='tight' )
    plt.close(Rfig)

    shading = ''
    print_values = False

    # plot model state variables
    for i, mvar in enumerate(mvars) :
        axx = maxs[mvar]

        # plot spatially averaged ensemble spread
        if mvar != 'T' :
            if False :
                axx.plot( fgspread_so[mvar] /varscl[mvar],  fgmean_so['P']/100, color=col_b, linewidth=lwn, alpha=0.5 )
                axx.plot( anaspread_so[mvar]/varscl[mvar], anamean_so['P']/100, color=col_a, linewidth=lwn, alpha=0.5 )
            else :
                fgm = fgmean_so[mvar]/varscl[mvar]
                fgs = fgspread_so[mvar]/varscl[mvar]
                fgl = fgm - fgs/2
                #fgl[where(fgl<0)] = 0
                fgr = fgm + fgs/2
                if shading == 'alpha' :
                    # semitransparent solid
                    axx.fill_betweenx( anamean_so['P']/100, fgl, fgr, color=col_b, alpha=0.05, zorder=-10 )
                elif shading == 'hatch' :
                    # hatched
                    mpl.rcParams['hatch.linewidth'] = 0.3
                    axx.fill_betweenx( anamean_so['P']/100, fgl, fgr, color='none', edgecolor=col_b, hatch='////', linewidth=0.3, zorder=-10 )

                agm = anamean_so[mvar]/varscl[mvar]
                ags = anaspread_so[mvar]/varscl[mvar]
                agl = agm - ags/2
                #agl[where(agl<0)] = 0
                agr = agm + ags/2
                if shading == 'alpha' :
                    # semitransparent solid
                    axx.fill_betweenx( anamean_so['P']/100, agl, agr, color=col_a, alpha=0.1, zorder=-5 )
                elif shading == 'hatch' :
                    # hatched
                    mpl.rcParams['hatch.linewidth'] = 0.3
                    axx.fill_betweenx( anamean_so['P']/100, agl, agr, color='none', edgecolor=col_a, hatch='\\\\\\\\', linewidth=0.3, zorder=-5 )

        # plot spatially averaged ensemble means
        if mvar == 'T' :
            axx.plot( (anamean_so[mvar]-fgmean_so[mvar])  /varscl[mvar], anamean_so['P']/100, color=col_i, label='A-B', linewidth=lw )
        else :
            axx.plot( fgmean_so[mvar]   /varscl[mvar],  fgmean_so['P']/100, color=col_b, label='B', linewidth=lw )
            axx.plot( anamean_so[mvar]  /varscl[mvar], anamean_so['P']/100, color=col_a, label='A', linewidth=lw )

        if mvar in ['QC','QI'] :
            axx.set_xlim((0,13.9))
        elif mvar in ['RELHUM','CLC'] :
            axx.set_xlim((0,100))
        elif mvar in ['U','V'] :
            axx.set_xlim((-10,10))
        elif mvar == 'T' :
            axx.set_xlim((-1,1))
        if mvar in ['U','V','T'] :
            axx.plot( (0,0), (1000,pmin), '--k', linewidth=0.5, alpha=0.5 )
        axx.set_ylim((1000,pmin))

        if print_values :
            xlm = axx.get_xlim()
            if short :
                if mvar == 'QC' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\tau^B_{\rm w}=%.1f$' % (
                        tauw_mean_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\tau^A_{\rm w}=%.1f$' % (
                        tauw_mean_ana[iobs]), fontsize=10, ha='right' )
                if mvar == 'QI' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\tau^B_{\rm i}=%.1f$' % (
                        taui_mean_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\tau^A_{\rm i}=%.1f$' % (
                        taui_mean_ana[iobs]), fontsize=10, ha='right' )
                if mvar == 'CLC' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\gamma^B_{\rm tot}=%.1f$' % (
                        clc_tot_mean_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\gamma^A_{\rm tot}=%.1f$' % (
                        clc_tot_mean_ana[iobs]), fontsize=10, ha='right' )

            else :
                if mvar == 'QC' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\tau^B_{\rm w}=%.1f\,(%.1f)$' % (
                        tauw_mean_fg[iobs], tauw_spread_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\tau^A_{\rm w}=%.1f\,(%.1f)$' % (
                        tauw_mean_ana[iobs], tauw_spread_ana[iobs]), fontsize=10, ha='right' )
                if mvar == 'QI' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\tau^B_{\rm i}=%.1f\,(%.1f)$' % (
                        taui_mean_fg[iobs], taui_spread_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\tau^A_{\rm i}=%.1f\,(%.1f)$' % (
                        taui_mean_ana[iobs], taui_spread_ana[iobs]), fontsize=10, ha='right' )
                if mvar == 'CLC' :
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 930, r'$\gamma^B_{\rm tot}=%.1f\,(%.1f)$' % (
                        clc_tot_mean_fg[iobs], clc_tot_spread_fg[iobs]), fontsize=10, ha='right' )
                    axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\gamma^A_{\rm tot}=%.1f\,(%.1f)$' % (
                        clc_tot_mean_ana[iobs], clc_tot_spread_ana[iobs]), fontsize=10, ha='right' )

        axx.set_xlabel({'QC':r'$10^{-5}$ kg/kg','QI':r'$10^{-5}$ kg/kg','CLC':'%','RELHUM':'%','T':'K'}[mvar])

        if i < len(mvars) - 1 :
            axx.set_xticks( list(array(axx.get_xticks())[:-1]) )
        if i == 0 :
            axx.set_ylabel('pressure [hPa]')
        axx.legend(frameon=False, title=mvar.replace('RELHUM','RH'), fontsize=10, loc='upper left')

    # plot radio sonde variables
    for i, rvar in enumerate(rvars) :
        axx = raxs[rvar]
        irep = temp_rep_close_to_obs[iobs]
        ekf_temp.replace_filter(filter='state=all varname=%s report=%d'%(rvar,irep))
        p_temp = ekf_temp.obs(param='plevel')/100

        scl = 1.0
        if rvar == 'RH' :
            axx.set_xlim((0,100))
            scl = 100.0
        elif rvar in ['U','V'] :
            axx.set_xlim((-13,13))

        if rvar == 'T' : # plot departures instead of means
            axx.set_xlim((-2.3,2.3))
            axx.plot( ekf_temp.fgmean() -ekf_temp.obs(), p_temp, col_temp_b, linewidth=lw, label='B-S')
            axx.plot( ekf_temp.anamean()-ekf_temp.obs(), p_temp, col_temp_a, linewidth=lw, label='A-S')
        else :
            axx.plot( scl*ekf_temp.obs(),     p_temp, col_temp_s, linewidth=lw, label='S')
            axx.plot( scl*ekf_temp.fgmean(),  p_temp, col_temp_b, linewidth=lw, label='B')
            axx.plot( scl*ekf_temp.anamean(), p_temp, col_temp_a, linewidth=lw, label='A')

        if rvar in ['U','V','T'] :
            axx.plot( (0,0), (1000,pmin), '--k', linewidth=0.5, alpha=0.5 )
        axx.set_ylim((1000,pmin))

        axx.set_xlabel({'U':'m/s','V':'m/s','T':'K','RH':'%'}[rvar])

        axx.legend(frameon=False, title=rvar, fontsize=10, loc='upper left')
        if print_values :
            if False :
                xlm = axx.get_xlim()
                axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 980, r'$\Delta\epsilon_{%s}=%.4f$' % (rvar,temp_errors[iobs][rvar+'_delta_rmse']),
                          ha='right', fontsize=10 )
            else :
                axx.set_title( r'$\Delta\epsilon_{%s}=%.3f$' % (rvar,temp_errors[iobs][rvar+'_delta_rmse']) )
        if i < len(rvars) - 1 :
            axx.set_xticks( list(array(axx.get_xticks())[1:-1]) )
        axx.set_yticklabels('')

    # save figure ......................................................................................................
    fig.savefig( "%s/profiles_paper_%s_obs%d.%s" % (obsdir,time,iobs,args.file_type), bbox_inches='tight' )
    plt.close(fig)



#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate VISOP plots for KENDA experiment')

    parser.add_argument(       '--bmo-max',    dest='bmo_max',    help='max. |B-O| to be plotted', type=float, default=0.0 )

    parser.add_argument( '-E', '--evolution',   dest='error_evolution',   help='generate error evolution plots', action='store_true' )
    parser.add_argument(       '--no-ekf',      dest='no_ekf',            help='do not use ekf data in error evolution plots', action='store_true' )

    parser.add_argument(       '--no-tauw',      dest='no_tauw',          help='do not use additional visop output like tauw', action='store_true' )

    parser.add_argument( '-o', '--obsloc',      dest='plot_obsloc',      help='plot observation locations',         action='store_true' )
    parser.add_argument( '-S', '--stamps',      dest='stamps',           help='generate stamp collection plot for each observation', action='store_true' )
    parser.add_argument( '-R', '--radio',       dest='radio',            help='include radio sond information', action='store_true' )

    parser.add_argument( '-i', '--increments',  dest='plot_increments',  help='plot fg + ana state and increments', action='store_true' )
    parser.add_argument( '-P', '--points',      dest='points',           help='indices of single obs. points for which increments are to be plotted', default='all' )
    parser.add_argument( '-L', '--location',    dest='location',         help='plot results for observation closest to the location specified as <lon>,<lat>', default='' )

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

 #   parser.add_argument( '-V', '--variables',   dest='variables',   help='comma-separated list of variables  to be considered (default: all)', default='' )
 #   parser.add_argument( '-O', '--obs-types',   dest='obs_types',   help='comma-separated list of obs. types to be considered (default: all)', default='' )
    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default='' )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default='' )

    parser.add_argument(       '--recompute',   dest='recompute',   help='write but do not read cache file', action='store_true' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='auto' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    # process all log files

    xps = {}
    rss = {}
    for logfile in args.logfile :

        print()
        print(("processing file %s ..." % logfile))

        xp = Experiment(logfile) #, verbose=True)
        xps[logfile] = xp
        print(('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
               xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] )))
        print()

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
            start_time = xp.veri_times[0]

        if args.end_time != '' :
            end_time = args.end_time
        else :
            end_time = start_time = xp.veri_times[-1]

        # loop over channels and times, read data and generate plots
        for channel in args.channel.split(',') :
            for itime, time in enumerate(xp.veri_times) :

                if Time14(time) < Time14(args.start_time) or Time14(time) > Time14(args.end_time) :
                    continue
                print()
                print('PROCESSING ekfRAD for t=%s ..........................................................' % time)
                ekf = xp.get_ekf( time, 'RAD', time_filter='0' )
                obs_times = ekf.obs(param='time')
                print('time %s : found %d observations...' % (time, len(obs_times)))
                print('observation times : ', time, ' - ', sorted(set(obs_times)))
                prefix = time + '_'
                
                if False :
                    idcs = where(obs_times == 0)
                    print('idcs', idcs)
                    n_obs = len(idcs[0])
                    print('ignoring times < 0 (before analysis) --> %d observations left.' % n_obs)
                    lat, lon = ekf.obs(param='lat')[idcs], ekf.obs(param='lon')[idcs]
                    obs = ekf.obs()[idcs]
                    fgmean   = ekf.fgmean()[idcs]
                    fgspread = ekf.fgspread()[idcs]
                    anamean   = ekf.anamean()[idcs]
                else :
                    n_obs = len(obs_times)
                    lat, lon = ekf.obs(param='lat'), ekf.obs(param='lon')
                    obs      = ekf.obs()
                    fgmean   = ekf.fgmean()
                    fgspread = ekf.fgspread()
                    anamean  = ekf.anamean()
                    anaens  = ekf.anaens()
                    fgens  = ekf.fgens()
                    print('anens.shape ', anaens.shape, fgens.shape, obs.shape, fgmean.shape, fgspread.shape)
                ered = abs(obs-fgmean) - abs(obs-anamean)

                if n_obs < 100 :
                    print()
                    print('--> RAD/REFL observations: ----------------------------------------------------------------------')
                    for i in range(n_obs) :
                        if i==0 :
                            print("--- #idx :   lat |   lon |  obs | fgmean | obs-fgm | fgsprd | anamean | obs-anm |  ered. |")
                        print("--- #%03d : %5.1f | %5.1f | %4.2f |   %4.2f |   %5.2f |   %4.2f |    %4.2f |   %5.2f | %6.3f |" % \
                            ( i, lat[i], lon[i], obs[i], fgmean[i], obs[i]-fgmean[i], fgspread[i], anamean[i], obs[i]-anamean[i], ered[i] ), end=' ')
                        print( obs_times[i], end=' ')
                        if obs[i]-fgmean[i] >  0.2 : print('missed', end=' ')
                        if obs[i]-fgmean[i] < -0.2 : print('false_alarm', end=' ')
                        if abs(obs[i]-fgmean[i]) < 0.05 :  print('  low |B-O|', end=' ')
                        if abs(obs[i]-fgmean[i]) > fgspread[i] :
                            print('  |O-B| > spread...')
                        else :
                            print()
                    print()
                else :
                    print('too many observations, skipping list...')

                hloc = float(xp.settings['VISOP_HLOC'])
                rcutoff = hloc*2*sqrt(10/3.0)
                print(('horizontal localization length scale : %.0fkm --> cutoff radius %.0fkm' % ( hloc, rcutoff )))
                soscale = float(xp.settings['VISOP_SUPEROBB'].split(',')[0])*3.0 # km
                print('superobbing scale : %.0fkm' % soscale)
                if not 'latlon' in xp.settings['VISOP_THINNING'] :
                    if ',' in  xp.settings['VISOP_THINNING'] :
                        thinx, thiny, thinxo, thinyo = list(map( int, xp.settings['VISOP_THINNING'].split(',') ))
                    else :
                        thinx, thiny, thinxo, thinyo = [int(xp.settings['VISOP_THINNING'])] * 4
                    print('thinning -> obs distance %.0fkm' % (thinx*soscale))
                    print('observation impacts do not overlap if r_cutoff = %.0fkm < %.0fkm = obsdistance/2' % (rcutoff,thinx*soscale/2))
                print()

                if True :
                    if args.no_tauw :
                        addvars = None
                    else :
                        tauw_mean_fg, tauw_spread_fg = zeros(n_obs), zeros(n_obs)
                        taui_mean_fg, taui_spread_fg = zeros(n_obs), zeros(n_obs)
                        clc_tot_mean_fg, clc_tot_spread_fg = zeros(n_obs), zeros(n_obs)
                        addvars = ['tauw_mean','taui_mean','clc_tot']

                    print('comparing fg ekfRAD values with visop_ensemble.i2o for fcst start time '+xp.fcst_start_times[itime]+', last output time')
                    print('i2o settings : '+xp.settings['VISOP_I2O_SETTINGS'])
                    vens_fg = xp.get_visop( xp.fcst_start_times[itime], channel=channel, preload=True, addvars=addvars )
                    #tauw_mean, tauw_spread = vens_fg.get_mean_and_spread_of_variable('tauw')
                    elat, elon, eobs, efgmean =  vens_fg.i2o('lat'), vens_fg.i2o('lon'), vens_fg.i2o('obs'), vens_fg.i2o('ensmean')

                    if not args.no_tauw :
                        for i in range(n_obs) :
                            j = argmin( abs(lat[i]-elat) + abs(lon[i]-elon) )
                            tauw_mean_fg[i], tauw_spread_fg[i] = vens_fg.i2o('tauw_mean_mean')[j], vens_fg.i2o('tauw_mean_spread')[j]
                            taui_mean_fg[i], taui_spread_fg[i] = vens_fg.i2o('taui_mean_mean')[j], vens_fg.i2o('taui_mean_spread')[j]
                            clc_tot_mean_fg[i], clc_tot_spread_fg[i]  = vens_fg.i2o('clc_tot_mean')[j],  vens_fg.i2o('clc_tot_spread')[j]
                            #print('--', i, j, abs(obs[i]-eobs[j])<1e-3, abs(fgmean[i]-efgmean[j])<1e-3, tauw_mean_fg[i], tauw_spread_fg[i])
                            print
                            if i==0 :
                                print("--- #idx i/j :   lat |   lon |  obs | fgmean | tau_w  |  tau_i |  clc_tot |")
                            print("--- #%03d/%03d : %5.1f | %5.1f | %4.2f |   %4.2f |  %5.2f |  %5.2f |    %5.2f |" % \
                                ( i, j, elat[i], elon[i], eobs[i], efgmean[i], tauw_mean_fg[i], taui_mean_fg[i], clc_tot_mean_fg[i] ), end=' ')
                            if abs(obs[i]-eobs[j])>1e-3 or abs(fgmean[i]-efgmean[j])>1e-3 or abs(elat[i]-lat[i])>0.05 or abs(elon[i]-lon[i])>0.05 :
                                print('<-- DOES NOT MATCH')
                            else :
                                print(' -- ok.')

                    fig, ax = plt.subplots()
                    ax.scatter( lon, lat, color='r', s=15 )
                    ax.scatter( vens_fg.i2o('lon'), vens_fg.i2o('lat'), color='b', s=5 )
                    fig.savefig(prefix+'obloc_ekf_vs_visopens.'+args.file_type)
                    plt.close(fig)
                    print()

                print('computing actual RAD/REFL analysis values .......................................................')
                print('from results of nonlinear operator applied to fcst started at analysis time = ', xp.veri_times[itime])
                tauw_mean_ana, tauw_spread_ana = zeros(n_obs), zeros(n_obs)
                taui_mean_ana, taui_spread_ana = zeros(n_obs), zeros(n_obs)
                clc_tot_mean_ana, clc_tot_spread_ana = zeros(n_obs), zeros(n_obs)
                vens_ana = xp.get_visop( xp.veri_times[itime], fcsttime=0, channel=channel, preload=True, addvars=addvars )
                print('meq shape : ', vens_ana.i2o('meq').shape)
                elat, elon, eobs, eanamean =  vens_ana.i2o('lat'), vens_ana.i2o('lon'), vens_ana.i2o('obs'), vens_ana.i2o('ensmean')
                anamean_real = zeros(anamean.shape)
                anaens_real = []
                for i in range(n_obs) :
                    j = argmin( abs(lat[i]-elat) + abs(lon[i]-elon) )
                    if not args.no_tauw :
                        tauw_mean_ana[i], tauw_spread_ana[i] = vens_ana.i2o('tauw_mean_mean')[j], vens_ana.i2o('tauw_mean_spread')[j]
                        taui_mean_ana[i], taui_spread_ana[i] = vens_ana.i2o('taui_mean_mean')[j], vens_ana.i2o('taui_mean_spread')[j]
                        clc_tot_mean_ana[i], clc_tot_spread_ana[i]  = vens_ana.i2o('clc_tot_mean')[j],  vens_ana.i2o('clc_tot_spread')[j]
                        #print(('-- %3d %3d obs = %5.3f / %5.3f,  ana = %5.3f / %5.3f' % ( i, j, obs[i], eobs[j], anamean[i], eanamean[j] )), tauw_mean_ana[i], tauw_spread_ana[i])
                        print
                        if i==0 :
                            print("--- #idx i/j :   lat |   lon |  obs | anamean | tau_w  |  tau_i |  clc_tot |")
                        print("--- #%03d/%03d : %5.1f | %5.1f | %4.2f |   %5.2f |  %5.2f |  %5.2f |    %5.2f |" % \
                            ( i, j, elat[i], elon[i], eobs[i], eanamean[i], tauw_mean_ana[i], taui_mean_ana[i], clc_tot_mean_ana[i] ), end=' ')
                        if abs(obs[i]-eobs[j])>1e-3  or abs(elat[i]-lat[i])>0.05 or abs(elon[i]-lon[i])>0.05 :
                            print('<-- DOES NOT MATCH')
                        else :
                            print(' -- ok.')
                    else :
                        print
                        if i==0 :
                            print("--- #idx i/j :   lat |   lon |  obs | anamean |")
                        print("--- #%03d/%03d : %5.1f | %5.1f | %4.2f |   %5.2f  |" % \
                            ( i, j, elat[i], elon[i], eobs[i], eanamean[i] ), end=' ')
                        if abs(obs[i]-eobs[j])>1e-3  or abs(elat[i]-lat[i])>0.05 or abs(elon[i]-lon[i])>0.05 :
                            print('<-- DOES NOT MATCH')
                        else :
                            print(' -- ok.')


                    anamean_real[i] = eanamean[j]
                    anaens_real.append(vens_ana.i2o('meq')[:,j])
                anaens_real = transpose(array(anaens_real))

                if True :

                    col_lin    = '#0066dd'
                    col_nonlin = '#ee6600'
                    col_diff   = '#ee0066'

                    danamean = abs(anamean_real-anamean)
                    refl_small = percentile( anamean, 10 )
                    idcs = where(anamean > refl_small)
                    anamean_err_rel_mean = (danamean/anamean)[idcs].mean()

                    if anamean.size > 100 :
                        alpha_ens = 0.02
                        alpha_mean = 0.1
                    else :
                        alpha_ens = 0.2
                        alpha_mean = 0.5

                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.scatter( anaens.ravel(), anaens_real.ravel(), color='#003399', s=15, alpha=alpha_ens, label='ensemble' )
                    ax.scatter( anamean, anamean_real, color='#ff0000', s=15, alpha=alpha_mean, label='mean' )
                    ax.plot((0,1.0),(0,1.0),'k',alpha=0.5)
                    ax.set_xlim((0,1))
                    ax.set_ylim((0,1))
                    ax.text( 0.98, 0.02, 'average relative change (for R>%4.2f) : %f'%(refl_small,anamean_err_rel_mean), ha='right' )
                    ax.set_xlabel('linear estimate for model equivalent from analysis')
                    ax.set_ylabel('model equivalent computed with nonlinear operator')
                    ax.legend(loc='upper left', frameon=False)
                    fig.savefig(prefix+'anaens_linear_vs_real.'+args.file_type)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8,8))
                    #ax.scatter( abs(fgens-obs)-fgspread, abs(anaens-anaens_real), color='#003399', s=15, alpha=0.01 )
                    #ax.scatter( obs+0*fgens, abs(anaens-anaens_real), color='#003399', s=15, alpha=0.01 )
                    #ax.scatter( fgens-obs, abs(anaens-anaens_real), color='#003399', s=15, alpha=0.01 )
                    #ax.scatter( fgens - obs, fgspread +0*fgens, s=50*minimum(abs(anaens-anaens_real)/0.2,1), color='#003399', alpha=0.01 )
                    #ax.scatter( fgens - obs, anaens-fgens, s=50*minimum(abs(anaens-anaens_real)/0.2,1), color='#003399', alpha=0.01 )
                    #ax.scatter( anaens-fgens, anaens-anaens_real, color='#003399', s=15, alpha=0.01 )
                    #ax.scatter( fgens, obs+0*fgens, s=100*minimum(abs(anaens-anaens_real)/0.2,1), color='#003399', alpha=0.01 )
                    ax.scatter( fgmean-obs, fgspread, s=100*minimum(abs(anamean-anamean_real)/0.2,1), color='#003399', alpha=0.3 )
                    ax.set_xlabel('B-O')
                    ax.set_ylabel('spread')
                    ax.set_title('size = nonlinearity error')
                    ax.axvspan( -0.01, 0.01, facecolor='#cc0000', alpha=0.1)
                    #ax.scatter( anamean, anamean_real, color='#ff0000', s=15, alpha=0.1 )
                    #ax.plot((0,1.0),(0,1.0),'k',alpha=0.5)
                    #ax.set_xlim((0,1))
                    #ax.set_ylim((0,1))
                    #ax.text( 0.98, 0.02, 'average relative change (for R>%4.2f) : %f'%(refl_small,anamean_err_rel_mean), ha='right' )
                    #ax.set_xlabel('linear estimate for model equivalent from analysis')
                    #ax.set_ylabel('model equivalent computed with nonlinear operator')
                    fig.savefig(prefix+'nlerror_vs_fgdep_vs_spread.'+args.file_type)
                    plt.close(fig)
                    print( 'spread min/mean/max : ', fgspread.min(), fgspread.mean(), fgspread.max() )

                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.scatter( fgmean-obs, abs(anamean-anamean_real)/abs(fgmean-obs), s=100*fgspread, color='#003399', alpha=0.3 )
                    ax.set_xlabel('B-O')
                    ax.set_ylabel('|A-A*| / |B-O|')
                    ax.set_title('size = spread')
                    ax.set_ylim((0,2))
                    ax.axvspan( -0.01, 0.01, facecolor='#cc0000', alpha=0.1)
                    fig.savefig(prefix+'nlerror_vs_fgdep_vs_spread2.'+args.file_type)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.scatter( fgmean+obs, abs(anamean-anamean_real)/abs(fgmean-obs), s=100*fgspread, color='#003399', alpha=0.3 )
                    ax.set_xlabel('B+O')
                    ax.set_ylabel('|A-A*|')
                    ax.set_title('size = spread')
                    fig.savefig(prefix+'nlerror_vs_fgdep_vs_spread3.'+args.file_type)
                    plt.close(fig)


                    dref = 0.1
                    n_tot = len(obs)
                    n_too_cloudy = len(where( fgmean - obs > dref )[0])
                    n_not_cloudy_enough = len(where( fgmean - obs < -dref )[0])
                    n_about_right = len(obs) - n_too_cloudy - n_not_cloudy_enough

                    dref2 = 0.03
                    n_better = len(where( abs(anamean-obs) < abs(fgmean-obs) - dref2 )[0])
                    n_worse  = len(where( abs(anamean-obs) >= abs(fgmean-obs) + dref2 )[0])
                    n_same   = len(where( abs( abs(anamean-obs) - abs(fgmean-obs) ) < dref2 )[0])
                    n_real_better = len(where( abs(anamean_real-obs) < abs(fgmean-obs) - dref2 )[0])
                    n_real_worse  = len(where( abs(anamean_real-obs) >= abs(fgmean-obs) + dref2 )[0])
                    n_real_same   = len(where( abs( abs(anamean_real-obs) - abs(fgmean-obs) ) < dref2 )[0])

                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.scatter( fgmean - obs, anamean-obs, color=col_lin, s=15, alpha=0.05, label='linear' )
                    ax.scatter( fgmean - obs, anamean_real-obs, color=col_nonlin, s=15, alpha=0.05, label='nonlinear' )
                    ax.plot((-1,1),(-1,1),'k',alpha=0.5)
                    ax.plot((-1,1),(0,0),'k',alpha=0.5)
                    ax.plot((-dref,-dref),(-1,1),'k',alpha=0.2)
                    ax.plot(( dref, dref),(-1,1),'k',alpha=0.2)
                    ax.plot( (-1,1), [dref2]*2, 'k', alpha=0.2 )
                    ax.plot( (-1,1), [-dref2]*2, 'k', alpha=0.2 )
                    ax.text( -dref*1.1, 0.9, '%4.2f'%(n_not_cloudy_enough/float(n_tot)), fontsize=8, ha='right', va='top')
                    ax.text(  dref*1.1, 0.9, '%4.2f'%(n_too_cloudy/float(n_tot)), fontsize=8, ha='left', va='top')
                    ax.text(         0, 0.9, '%4.2f'%(n_about_right/float(n_tot)), fontsize=8, ha='center', va='top')
                    ax.text( -0.95, -0.90, 'lin. estimate: better: %4.2f, worse: %4.2f, about the same %4.2f' % (n_better/float(n_tot),n_worse/float(n_tot),n_same/float(n_tot)) )
                    ax.text( -0.95, -0.95, 'nonlinear:     better: %4.2f, worse: %4.2f, about the same %4.2f' % (n_real_better/float(n_tot),n_real_worse/float(n_tot),n_real_same/float(n_tot)) )
                    ax.set_xlim((-1,1))
                    ax.set_ylim((-1,1))
                    ax.grid()
                    ax.set_xlabel('fgmean - obs')
                    ax.set_ylabel('anamean - obs')
                    ax.legend(loc='upper left', frameon=False)
                    fig.savefig(prefix+'error_reduction.'+args.file_type)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8,8))
                    idcs = where(abs(fgmean-obs)<0.05)
                    ax.scatter( obs[idcs], fgmean[idcs], color='b', alpha=0.1,  label='B')
                    ax.scatter( obs[idcs], anamean[idcs], color='g', alpha=0.1, label='A' )
                    ax.scatter( obs[idcs], anamean_real[idcs], color='r', alpha=0.1, label='A*')
                    ax.legend(loc='upper left', frameon=False)
                    ax.text( 0.05, 0.05, '<|A-A*|> = {}'.format(abs(anamean[idcs]-anamean_real[idcs]).mean()))
                    ax.text( 0.05, 0.10, '<|O-A*|> = {}'.format(abs(obs[idcs]-anamean_real[idcs]).mean()))
                    ax.text( 0.05, 0.15, '<|O-A|> = {}'.format(abs(obs[idcs]-anamean[idcs]).mean()))
                    ax.text( 0.05, 0.20, '<|O-B|> = {}'.format(abs(obs[idcs]-fgmean[idcs]).mean()))
                    fig.savefig(prefix+'OBA.'+args.file_type)
                    plt.close(fig)
                    
                                                            
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.scatter( fgmean - obs, abs(fgmean-obs)-abs(anamean-obs), color=col_lin, s=15, alpha=0.05 )
                    ax.scatter( fgmean - obs, abs(fgmean-obs)-abs(anamean_real-obs), color=col_nonlin, s=15, alpha=0.1 )
                    #ax.plot((-1,1),(-1,1),'k',alpha=0.5)
                    #ax.plot((-1,1),(0,0),'k',alpha=0.5)
                    ax.plot((-dref,-dref),ax.get_ylim(),'k',alpha=0.2)
                    ax.plot(( dref, dref),ax.get_ylim(),'k',alpha=0.2)
                    ax.plot( (-1,1), [dref2]*2, 'k', alpha=0.2 )
                    ax.plot( (-1,1), [-dref2]*2, 'k', alpha=0.2 )
                    #ax.text( -dref*1.1, 0.9, '%4.2f'%(n_not_cloudy_enough/float(n_tot)), fontsize=8, ha='right', va='top')
                    #ax.text(  dref*1.1, 0.9, '%4.2f'%(n_too_cloudy/float(n_tot)), fontsize=8, ha='left', va='top')
                    #ax.text(         0, 0.9, '%4.2f'%(n_about_right/float(n_tot)), fontsize=8, ha='center', va='top')
                    #ax.text( -0.95, -0.90, 'lin. estimate: better: %4.2f, worse: %4.2f, about the same %4.2f' % (n_better/float(n_tot),n_worse/float(n_tot),n_same/float(n_tot)) )
                    #ax.text( -0.95, -0.95, 'nonlinear:     better: %4.2f, worse: %4.2f, about the same %4.2f' % (n_real_better/float(n_tot),n_real_worse/float(n_tot),n_real_same/float(n_tot)) )
                    #ax.set_xlim((-1,1))
                    #ax.set_ylim((-1,1))
                    ax.grid()
                    ax.set_xlabel('fgmean - obs')
                    ax.set_ylabel('abs(fgmean-obs)-abs(anamean - obs)')
                    fig.savefig(prefix+'error_reduction2.'+args.file_type)
                    plt.close(fig)


                    if anamean.size > 100 :
                        alpha_dots = 0.3
                    else :
                        alpha_dots = 0.7
                    er3lin    = (anamean-obs)/(fgmean - obs)
                    er3nonlin = (anamean_real-obs)/(fgmean - obs)
                    fig, ax = plt.subplots(figsize=(5,5))
                    for i in range(obs.size) :
                        ax.plot( (fgmean[i] - obs[i], fgmean[i] - obs[i]), (er3lin[i],er3nonlin[i]), 'k', alpha=alpha_dots/3 )
                    ax.scatter( fgmean - obs, er3lin, color=col_lin, s=15, alpha=alpha_dots, label='linear' )
                    ax.scatter( fgmean - obs, er3nonlin, color=col_nonlin, s=15, alpha=alpha_dots, label='nonlinear' )
                    ax.axhspan( -1, 1, facecolor='#0066cc', alpha=0.1, label='reduction')
                    ax.set_xlim((-0.6,0.6))
                    ax.set_ylim((-2,2))
                    ax.grid()
                    ax.set_xlabel('fgmean - obs')
                    ax.set_ylabel('(anamean - obs)/(fgmean - obs)')
                    ax.legend(loc='lower right', frameon=False)
                    fig.savefig(prefix+'error_reduction3.'+args.file_type)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(5,5))
                    for i in range(obs.size) :
                        ax.plot( (fgmean[i] + obs[i], fgmean[i] + obs[i]), (er3lin[i],er3nonlin[i]), 'k', alpha=alpha_dots/3 )
                    ax.scatter( fgmean + obs, er3lin, color=col_lin, s=15, alpha=alpha_dots, label='linear' )
                    ax.scatter( fgmean + obs, er3nonlin, color=col_nonlin, s=15, alpha=alpha_dots, label='nonlinear' )
                    ax.axhspan( -1, 1, facecolor='#0066cc', alpha=0.1, label='reduction')
                    ax.set_xlim((0,2))
                    ax.set_ylim((-2,2))
                    ax.grid()
                    ax.set_xlabel('fgmean + obs')
                    ax.set_ylabel('(anamean - obs)/(fgmean - obs)')
                    ax.legend(loc='lower right', frameon=False)
                    fig.savefig(prefix+'error_reduction3b.'+args.file_type)
                    plt.close(fig)


                    aesum_fg        = abs(fgmean-obs).sum()
                    aesum_analin    = abs(anamean-obs).sum()
                    aesum_ananonlin = abs(anamean_real-obs).sum()
                    aesum_diff      = abs(anamean_real-anamean).sum()

                    if obs.size < 100 :
                        fmo_bins = arange(-0.7,0.701,0.2)
                    else :
                        fmo_bins = arange(-0.7,0.701,0.05)

                    maerlin    = zeros(fmo_bins.size-1)
                    maernonlin = zeros(fmo_bins.size-1)
                    maerdiff   = zeros(fmo_bins.size-1)
                    maercases  = zeros(fmo_bins.size-1,dtype=int)
                    for i in range(fmo_bins.size-1) :
                        fmo_min = fmo_bins[i]
                        fmo_max = fmo_bins[i+1]
                        idcs = where( ((fgmean - obs)>=fmo_min) & ((fgmean - obs)<=fmo_max) )

                        # abs(anamean-obs)-abs(fgmean-obs) < 0 -> error reduced
                        if len(idcs[0]) > 0 :
                            maerlin[i]    = ( abs(fgmean[idcs]-obs[idcs]) - abs(anamean[idcs]-obs[idcs]) ).mean()
                            maernonlin[i] = ( abs(fgmean[idcs]-obs[idcs]) - abs(anamean_real[idcs]-obs[idcs]) ).mean()
                            maerdiff[i]   = abs( anamean_real[idcs] - anamean[idcs] ).mean()
                            maercases[i]  = len(idcs[0])
                        else :
                            maerlin[i]    = 0
                            maernonlin[i] = 0
                            maerdiff[i]   = 0
                            maercases[i]  = 0

                    #for i in range(fmo_bins.size-1) :
                    #    ax.plot( (fmo_min,fmo_max), (maerlin[i],maerlin[i]), color=col_lin )
                    #    ax.plot( (fmo_min,fmo_max), (maernonlin[i],maernonlin[i]), color=col_nonlin )
                    #    ax.text( fmo_min, 0.05, str(maercases), fontsize=8, rotation=90)
                    fmo_bincenters = 0.5*(fmo_bins[1:] + fmo_bins[:-1])
                    maercasefrac = maercases/float(maercases.sum())

                    fig, ax = plt.subplots( 2, figsize=(5,3.5), sharex=True, gridspec_kw = {'height_ratios':[5, 1]} )
                    fig.subplots_adjust(hspace=0)

                    maerval = maerlin
                    maerval[where(maercases<3)] = NaN
                    ax[0].plot( fmo_bincenters, maerval, color=col_lin, label='<|B-O|-|A-O|>', linewidth=2 )
                    maerval = maernonlin
                    maerval[where(maercases<3)] = NaN
                    ax[0].plot( fmo_bincenters, maerval, color=col_nonlin, label='<|B-O|-|A*-O|>', linewidth=2 )

                    maerval = maerdiff
                    maerval[where(maercases<3)] = NaN
                    ax[0].plot( fmo_bincenters, maerval, color=col_diff, label='<|A-A*|>', linewidth=2 )

                    #ax[0].plot( fmo_bincenters, 0.3*maercasefrac/maercasefrac.max(), color='#999999', label='~ #obs', linewidth=2  )
                    ax[0].set_xlim((fmo_bins[0],fmo_bins[-1]))
                    if args.bmo_max > 1e-3 :
                        ax[0].set_xlim((-args.bmo_max,args.bmo_max))
                        ax[1].set_xlim((-args.bmo_max,args.bmo_max))
                        ylim=(-0.05,0.35*args.bmo_max/0.7)
                    else :
                        ax[0].set_xlim((fmo_bins[0],fmo_bins[-1]))
                        ax[1].set_xlim((fmo_bins[0],fmo_bins[-1]))
                        ylim=(-0.05,0.35)
                    ax[0].set_ylim(ylim)
                    #ax.plot((0,0),(-0.35,0.1),'k',linewidth=0.5)
                    ax[0].plot((-0.35,0,0.35),(0.35,0,0.35),'--k',linewidth=0.5)
                    ax[0].axhspan( ylim[0], 0, facecolor='#cc0000', alpha=0.1)
                    ax[0].plot( (fmo_bins[0],fmo_bins[-1]),(0,0), color='#cc0000', linewidth=0.5)
                    #ax[0].grid()
                    #ax.set_title('mean error reduction in B-O bins')
                    ax[0].set_title('{} / {} UTC ({} obs.)'.format( xp.settings['exp'], time[8:-4], obs.size), fontsize=10)

                    ax[0].set_ylabel('mean error reduction')
                    ax[0].legend(loc='upper center', frameon=False) #, title=time[:-4])
                    ax[1].plot( fmo_bincenters, maercases, color='k', linewidth=1 )
                    ax[1].set_ylabel('#obs')
                    ax[1].set_yticklabels('')
                    ax[1].set_xlabel('B-O')
                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.02)
                    fig.savefig(prefix+'error_reduction4.'+args.file_type, bbox_inches='tight')
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(6,4))
                    vmin = minimum( (maerlin*maercasefrac).min(), (maernonlin*maercasefrac).min() )*1.1
                    vmax = maximum( (maerlin*maercasefrac).max(), (maernonlin*maercasefrac).max() )*1.1
                    #if obs.size < 200 :
                    vmin = -0.005 #-0.012
                    vmax =  0.012 #0.005
                    #else :
                    #    vmin = -0.005 #-0.008
                    #    vmax = 0.008 #0.005
                    print('vmin={}, vmax={}'.format(vmin,vmax))
                    ax.plot( fmo_bincenters, maerlin*maercasefrac, color=col_lin, label='N(B-O) <|B-O|-|A-O|>', linewidth=2 )
                    ax.plot( fmo_bincenters, maernonlin*maercasefrac, color=col_nonlin, label='N(B-O)<|B-O|-|A*-O|>', linewidth=2 )
                    ax.plot( fmo_bincenters, maerdiff*maercasefrac, color=col_diff, label='N(B-O) <|A-A*|>', linewidth=2 )
                    #ax.plot( fmo_bincenters, 0.1*maercasefrac/maercasefrac.max(), color='#999999' )
                    ax.axhspan( vmin, 0, facecolor='#cc0000', alpha=0.1 )
                    ax.set_title('{} / {} UTC : relative error changes: lin. {:.0f}%, nonlin. {:.0f}%, diff. {:.0f}%'.format( \
                        xp.settings['exp'], time[8:-4],
                        100*(aesum_analin-aesum_fg)/aesum_fg, 100*(aesum_ananonlin-aesum_fg)/aesum_fg, 100*aesum_diff/aesum_fg ),
                        fontsize=10)
                    ax.plot( (fmo_bins[0],fmo_bins[-1]),(0,0), color='#cc0000', linewidth=0.5)
                    ax.plot( (0,0), (vmin,vmax), color='k', linewidth=0.5)
                    
                    if args.bmo_max > 1e-3 :
                        ax.set_xlim((-args.bmo_max,args.bmo_max))
                    else :
                        ax.set_xlim((fmo_bins[0],fmo_bins[-1]))
                    #ax.set_xlim((fmo_bins[0],fmo_bins[-1]))
                    ax.set_ylim((vmin,vmax))
                    #ax.grid()
                    ax.set_xlabel('B-O')
                    ax.set_ylabel('mean error reduction x number of cases') #'$\Delta$e_i')
                    ax.legend(loc='upper left', frameon=False, fontsize=11) #, title=time[:-4])
                    fig.savefig(prefix+'error_reduction4b.'+args.file_type, bbox_inches='tight')
                    plt.close(fig)

                    # nonlinearity error as function of spread and O+B
                    #fpo_bins = arange(0,2,0.3/2)
                    #spread_bins = arange(0,1.01,0.1)
                    #for p in range(fpo_bins.size-1) :
                    #    for s in range(spread_bins.size-1) :
                    #        fpo_min = fpo_bins[p]
                    #        fpo_max = fpo_bins[p+1]
                    #        spread_min = spread_bins[s]
                    #        spread_max = spread_bins[s+1]
                    #        idcs = where( ( fgspread>=spread_min) & (fgspread<=spread_max) \
                    #                    & ((fgmean + obs)>=fpo_min) & ((fgmean + obs)<=fpo_max) )
                    #        (abs(anamean_real[idcs]-obs[idcs])-abs(anamean[idcs]-obs[idcs]) / (fgmean[idcs]+obs[idcs])



                    # error reduction as function of B+O and B-O
                    fmo_bins = arange(-0.7,0.701,0.2/2)
                    fpo_bins = arange(0,2,0.3/2)
                    maerlin    = zeros((fmo_bins.size-1,fpo_bins.size-1))
                    maernonlin = zeros((fmo_bins.size-1,fpo_bins.size-1))
                    maercases  = zeros((fmo_bins.size-1,fpo_bins.size-1),dtype=int)
                    for m in range(fmo_bins.size-1) :
                        for p in range(fpo_bins.size-1) :
                            fmo_min = fmo_bins[m]
                            fmo_max = fmo_bins[m+1]
                            fpo_min = fpo_bins[p]
                            fpo_max = fpo_bins[p+1]
                            idcs = where( ((fgmean - obs)>=fmo_min) & ((fgmean - obs)<=fmo_max) \
                                        & ((fgmean + obs)>=fpo_min) & ((fgmean + obs)<=fpo_max) )
                            if len(idcs[0]) > 0 :
                                maerlin[m,p]    = (abs(anamean[idcs]-obs[idcs])-abs(fgmean[idcs]-obs[idcs])).mean()
                                maernonlin[m,p] = (abs(anamean_real[idcs]-obs[idcs])-abs(fgmean[idcs]-obs[idcs])).mean()
                                maercases[m,p]  = len(idcs[0])
                            else :
                                maerlin[m,p]    = 0
                                maernonlin[m,p] = 0
                                maercases[m,p]  = 0


                    fig, ax = plt.subplots(figsize=(5,5))
                    mw = maerlin*maercases
                    mw_mx = maximum( abs(mw.min()), abs(mw.max()) )
                    ax.imshow( mw, extent=(fpo_bins[0], fpo_bins[-1], fmo_bins[0], fmo_bins[-1]),
                               vmin=-mw_mx, vmax=mw_mx, origin='lower', cmap='RdBu_r' )
                    fmo_bc = 0.5*(fmo_bins[1:] + fmo_bins[:-1])
                    fpo_bc = 0.5*(fpo_bins[1:] + fpo_bins[:-1])
                    ax.contour( fpo_bc, fmo_bc, mw, levels=[0] )
                    for m in range(fmo_bins.size-1) :
                        for p in range(fpo_bins.size-1) :
                            ax.text( fpo_bc[p], fmo_bc[m], '{:.3f}'.format(maerlin[m,p]), fontsize=8, color='r' if maerlin[m,p] > 0 else 'k' )
                    ax.set_xlim((fpo_bins[0],fpo_bins[-1]))
                    ax.set_ylim((fmo_bins[0],fmo_bins[-1]))
                    ax.set_title('{} {}'.format(mw.sum(), (aesum_analin-aesum_fg)/aesum_fg))
                    fig.savefig(prefix+'error_reduction5_maerlin2d.'+args.file_type)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(5,5))
                    mw = maernonlin*maercases
                    mw_mx = maximum( abs(mw.min()), abs(mw.max()) )
                    ax.imshow( mw, extent=(fpo_bins[0], fpo_bins[-1], fmo_bins[0], fmo_bins[-1]),
                               vmin=-mw_mx, vmax=mw_mx, origin='lower', cmap='RdBu_r' )
                    fmo_bc = 0.5*(fmo_bins[1:] + fmo_bins[:-1])
                    fpo_bc = 0.5*(fpo_bins[1:] + fpo_bins[:-1])
                    ax.contour( fpo_bc, fmo_bc, mw, levels=[0] )
                    for m in range(fmo_bins.size-1) :
                        for p in range(fpo_bins.size-1) :
                            ax.text( fpo_bc[p], fmo_bc[m], '{:.3f}'.format(maernonlin[m,p]), fontsize=8, color='r' if maernonlin[m,p] > 0 else 'k' )
                    ax.set_xlim((fpo_bins[0],fpo_bins[-1]))
                    ax.set_ylim((fmo_bins[0],fmo_bins[-1]))
                    #ax.set_title('{}'.format(mw.sum()))
                    ax.set_title('{} {}'.format(mw.sum(), (aesum_ananonlin-aesum_fg)/aesum_fg))
                    fig.savefig(prefix+'error_reduction5_maernonlin2d.'+args.file_type)
                    plt.close(fig)





                    fig, ax = plt.subplots(figsize=(8,8))
                    if n_obs < 100 :
                        alpha = 0.5
                    elif n_obs < 1000 :
                        alpha = 0.1
                    else :
                        alpha = 0.05
                    ax.scatter( abs(abs(fgmean-obs)-abs(anamean-obs)), abs(abs(fgmean-obs)-abs(anamean_real-obs)), color=col_lin, s=15, alpha=alpha )
                    #ax.plot((-1,1),(-1,1),'k',alpha=0.5)
                    #ax.plot((-1,1),(0,0),'k',alpha=0.5)
                    ax.plot( (0,0.3), (0,0.3), 'k', alpha=0.5 )
                    ax.grid()
                    ax.set_xlabel('||FG-O|-|ANA-O|| linear')
                    ax.set_ylabel('||FG-O|-|ANA-O|| non-linear')
                    fig.savefig('error_reduction_nl.'+args.file_type)
                    plt.close(fig)


                # generate stamps plot
                if args.stamps :
                    print()
                    print('Generating stamp plots...')
                    # get reflectance ensembles
                    #vens_fg   = xp.get_visop( xp.fcst_start_times[itime], fcsttime=-1, channel=channel, preload=True )
                    vens_temp = xp.get_visop( xp.fcst_start_times[itime], fcsttime=45, channel=channel, preload=False )
                    #vens_ana  = xp.get_visop( xp.veri_times[itime], fcsttime=0, channel=channel, preload=True )

                    cwd = os.getcwd()

                    for ipt in range(obs.size) :
                        print(ipt, end=' ')
                        obsdir = cwd+'/obs%03d'%ipt
                        if not os.path.exists(obsdir) :
                            os.mkdir(obsdir)

                        # determine superobs center pixel position
                        ijpx = vens_fg.coordinates_to_indices( r_[lon[ipt]], r_[lat[ipt]] )
                        ipx, jpx = ijpx[0][0], ijpx[1][0]
                        dipx, djpx = list(map( int, xp.settings['VISOP_SUPEROBB'].split(',')[:2] ))
                        ipxl = ipx - dipx//2
                        ipxh = ipxl + dipx
                        jpxl = jpx - djpx//2
                        jpxh = jpxl + djpx

                        fig = plt.figure( figsize=(5,5) )

                        nline = 4
                        nrow = int(ceil(xp.n_ens/float(nline)))
                        w = 1.0 / minimum(nline,nrow+1)

                        ax_obs = fig.add_axes(( 0.0, 1.0-w, w, w ))
                        ax_obs.imshow( vens_fg[-1][ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower', cmap='gray',
                                       aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                        ax_obs.text( 0.05, 0.05, "OBS : %4.2f"%(vens_fg[-1][ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                        ax_obs.get_xaxis().set_visible(False)
                        ax_obs.get_yaxis().set_visible(False)

                        ax_temp = fig.add_axes(( 0.0, 1.0-2*w, w, w ))
                        ax_temp.imshow( vens_temp[-1][ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower', cmap='gray',
                                       aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                        ax_temp.text( 0.05, 0.05, "OBS(-15min) : %4.2f"%(vens_temp[-1][ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                        ax_temp.get_xaxis().set_visible(False)
                        ax_temp.get_yaxis().set_visible(False)

                        ax_mean = fig.add_axes(( 0.0, 1.0-3*w, w, w ))
                        ax_mean.imshow( vens_fg.mean2d()[ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower', cmap='gray',
                                        aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                        ax_mean.text( 0.05, 0.05, "FG mean : %4.2f"%(vens_fg.mean2d()[ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                        ax_mean.get_xaxis().set_visible(False)
                        ax_mean.get_yaxis().set_visible(False)

                        ax_anamean = fig.add_axes(( 0.0, 1.0-4*w, w, w ))
                        ax_anamean.imshow( vens_ana.mean2d()[ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower', cmap='gray',
                                        aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                        ax_anamean.text( 0.05, 0.05, "ANA mean : %4.2f"%(vens_ana.mean2d()[ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                        ax_anamean.get_xaxis().set_visible(False)
                        ax_anamean.get_yaxis().set_visible(False)

                        for i in range(nline) :
                            for j in range(nrow) :
                                k = 1 + i + j*nline
                                if k > xp.n_ens : break
                                ax_mem = fig.add_axes(( w + j*w, 1.0 - (i+1)*w, w, w ))
                                ax_mem.imshow( vens_fg[k][ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower',
                                               cmap='gray', aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                                ax_mem.text( 0.05, 0.05, "%d : %4.2f"%(k,vens_fg[k][ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                                ax_mem.get_xaxis().set_visible(False)
                                ax_mem.get_yaxis().set_visible(False)

                        for i in range(nline) :
                            for j in range(nrow) :
                                k = 1 + i + j*nline
                                if k > xp.n_ens : break
                                ax_mem = fig.add_axes(( w + j*w, 1.0 - (nline+0.1+i+1)*w, w, w ))
                                ax_mem.imshow( vens_ana[k][ ipxl:ipxh, jpxl:jpxh ], vmin=0, vmax=1, origin='lower',
                                               cmap='gray', aspect=1.0, extent=(0,1,0,1), interpolation='nearest' )
                                ax_mem.text( 0.05, 0.05, "%d : %4.2f"%(k,vens_ana[k][ ipxl:ipxh, jpxl:jpxh ].mean()), fontsize=9, color='r' )
                                ax_mem.get_xaxis().set_visible(False)
                                ax_mem.get_yaxis().set_visible(False)

                        fig.savefig('obs%03d/stamps.%s'%(ipt,args.file_type), bbox_inches='tight', dpi=100)
                        plt.close(fig)
                    print()

                temp_rep_close_to_obs = {}
                temp_errors = {}

                if not args.radio :
                    ekf_temp = None
                else :
                    # check if radiosonde data is available in the next assimilation cycle
                    print()
                    print('Searching for radiosondes in ekfTEMP for analysis at %s...' % time)
                    try :
                        ekf_temp = xp.get_ekf( time, 'TEMP', state_filter='all' ) #, area_filter=xp.settings['OBS_EVAL_AREA'] )
                    except :
                        print('found no radio sonde observations...')
                        ekf_temp = None

                    if not ekf_temp is None :

                        import kendapy.plot_ekf as plot_ekf

                        temp_lat = []
                        temp_lon = []
                        temp_time = []

                        temp_reps = ekf_temp.find_reports_containing( 'T' )
                        n_temp = len(temp_reps)

                        filter = ekf_temp.get_filter()
                        for irep in temp_reps :
                            ekf_temp.add_filter(filter='report=%d'%irep)
                            temp_lat.append(  ekf_temp.obs(param='lat')[0] )
                            temp_lon.append(  ekf_temp.obs(param='lon')[0] )
                            temp_time.append( ekf_temp.obs(param='time')[0] )
                            ekf_temp.replace_filter(filter=filter)

                        print('TEMP available for UTC', ekf_temp.attr['verification_ref_time'])
                        for k in range(n_temp) :
                            print("--- #%04d : time=%3d | lat=%5.2f | lon=%5.2f |" % (k, temp_time[k], temp_lat[k], temp_lon[k]), end=' ')

                            dist = sqrt( (lat-temp_lat[k])**2 + ((lon-temp_lon[k])*cos(lon*pi/180.0))**2 )
                            i = argmin( dist )
                            print(' --> closest superobb : #', i, dist[i], end=' ')
                            temp_rep_close_to_obs[i] = temp_reps[k]

                            mindist = dist.min()
                            if mindist < 0.35 :
                                print(' --> HIT!')
                                '    --> plotting radio sonde profiles...'
                                cwd = os.getcwd()
                                obsdir = cwd+'/obs%03d'%i
                                if not os.path.exists(obsdir) :
                                    os.mkdir(obsdir)
                                temp_errors[i] = plot_ekf.plot_radiosonde( ekf_temp, temp_reps[k], path=obsdir, nsmooth=1, return_errors=True )
                                #print 'TEMP ERRORS : ', temp_errors[i]
                            else :
                                print()
                        print()
                    # save all observation information known so far...
                    allobs={}
                    for i in arange(n_obs) : #qwer
                        obsinfo = {}
                        obsinfo.update(temp_errors[i])
                        if not args.no_tauw :
                            obsinfo.update({ 'tauw_mean_fg':tauw_mean_fg[i],   'tauw_spread_fg':tauw_spread_fg[i],
                                             'taui_mean_fg':taui_mean_fg[i],   'taui_spread_fg':taui_spread_fg[i],
                                             'tauw_mean_ana':tauw_mean_ana[i], 'tauw_spread_ana':tauw_spread_ana[i],
                                             'taui_mean_ana':taui_mean_ana[i], 'taui_spread_ana':taui_spread_ana[i] })
                        obsinfo.update({ 'expid':xp.expid, 'iobs':i, 'lat':lat[i], 'lon':lon[i], 'obs':obs[i],
                                         'fgmean':fgmean[i], 'fgspread':fgspread[i], 'anamean':anamean[i],
                                         'anamean_real':anamean_real[i]})
                        allobs[xp.expid+('_%d'%i)] = obsinfo
                    with open( 'observations.pickle','w') as f :
                        pickle.dump( allobs, f, pickle.HIGHEST_PROTOCOL )


                if args.plot_obsloc :
                    print('plotting observation locations...')

                    # scatter plots
                    scplots = [{'lat':lat,'lon':lon,'text':list(map(str,arange(lat.size))),'color':'#ff00ff'}]
                    if not ekf_temp is None :
                        scplots.append({'lat':array(temp_lat),'lon':array(temp_lon),'text':list(map(str,temp_time)),'color':'#ffff00'})

                    #vens_fg = xp.get_visop( xp.fcst_start_times[itime], channel=channel, preload=True )
                    plot_reflectance( vens_fg[-1],         'observation', time+'_fg_obs.'+args.file_type,      lat=vens_fg.lat, lon=vens_fg.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_fg.mean2d(),    'ens. mean',   time+'_fg_mean.'+args.file_type,     lat=vens_fg.lat, lon=vens_fg.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_fg.spread2d(),  'ens. spread', time+'_fg_spread.'+args.file_type,   lat=vens_fg.lat, lon=vens_fg.lon, vmin=0, vmax=0.3,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_fg.bias2d(),    'ens. bias',   time+'_fg_bias.'+args.file_type,     lat=vens_fg.lat, lon=vens_fg.lon, vmin=-0.3, vmax=0.3,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_fg.probex(0.3), 'probex03',    time+'_fg_probex03.'+args.file_type, lat=vens_fg.lat, lon=vens_fg.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_fg.probex(0.5), 'probex05',    time+'_fg_probex05.'+args.file_type, lat=vens_fg.lat, lon=vens_fg.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )

                    plot_reflectance( vens_ana[-1],         'observation', time+'_ana_obs.'+args.file_type,      lat=vens_ana.lat, lon=vens_ana.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_ana.mean2d(),    'ens. mean',   time+'_ana_mean.'+args.file_type,     lat=vens_ana.lat, lon=vens_ana.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_ana.spread2d(),  'ens. spread', time+'_ana_spread.'+args.file_type,   lat=vens_ana.lat, lon=vens_ana.lon, vmin=0, vmax=0.3,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_ana.bias2d(),    'ens. bias',   time+'_ana_bias.'+args.file_type,     lat=vens_ana.lat, lon=vens_ana.lon, vmin=-0.3, vmax=0.3,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_ana.probex(0.3), 'probex03',    time+'_ana_probex03.'+args.file_type, lat=vens_ana.lat, lon=vens_ana.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )
                    plot_reflectance( vens_ana.probex(0.5), 'probex05',    time+'_ana_probex05.'+args.file_type, lat=vens_ana.lat, lon=vens_ana.lon, vmin=0, vmax=1.0,
                                              infos=True, notext=False, grid=True, scatter=scplots )

                if args.plot_increments :
                    print()
                    print('GENERATING INCREMENT PLOTS...................................................................')
                    print('reading mean model states and computing increments... ')

                    maxinc = { 'QC':1e-4, 'QI':1e-4, 'QS':1e-5, 'QV':1e-3, 'T':1.0, 'U':1.5, 'V':1.5, 'W':0.2, 'P':25, 'RELHUM':100 } #, 'CLC':1
                    varscl = { 'QC':1e-5, 'QI':1e-5, 'QS':1e-5, 'QV':1e-3, 'T':1.0, 'U':1.0, 'V':1.0, 'W':1.0, 'P':100, 'RELHUM':1,  'CLC':1,
                               'TQC':1, 'TQI':1, 'TQV':1e3 }

                    # get model state
                    #cs_anamean = xp.get_cosmo( time, prefix='laf', suffix='mean' )
                    #cs_fgmean  = xp.get_cosmo( xp.fcst_start_times[itime], prefix='lff', suffix='mean' )

                    preload_vars = list(maxinc.keys()) + ['RLAT','RLON'] #['QC','QI','QS','QV','T','U','V','W','P']
                    translation = 'pre2019'

                    print('--> loading ana mean cosmo file '+ xp.cosmo_filename( time, prefix='laf', suffix='mean' ))
                    cs_anamean_file = xp.cosmo_filename( time, prefix='laf', suffix='mean' )
                    cs_anamean = CosmoState( cs_anamean_file, preload=preload_vars, translation=translation )
                    #print 'ANAMEAN FILE CONTENTS........................................................................'
                    #cs_anamean.list_variables()
                    #print

                    print('--> loading ana spread cosmo file '+ xp.cosmo_filename( time, prefix='laf', suffix='spread' ))
                    cs_anaspread_file = xp.cosmo_filename( time, prefix='laf', suffix='spread' )
                    cs_anaspread = CosmoState(cs_anaspread_file,preload=preload_vars, translation=translation )
                    #print 'ANASPREAD FILE CONTENTS........................................................................'
                    #cs_anaspread.list_variables()
                    #print

                    cs_fgmean_file = xp.cosmo_filename( xp.fcst_start_times[itime], prefix='lff', suffix='mean' )
                    print('--> loading fg  mean cosmo file '+ cs_fgmean_file)
                    cs_fgmean = CosmoState( cs_fgmean_file, constfile=cs_fgmean_file,preload=preload_vars, translation=translation )
                    #cs_fgmean = CosmoState( cs_fgmean_file, constfile=cs_fgmean_file.replace('.mean','.det'),preload=preload_vars) # AAAAARGH!!!
                    #print 'FGMEAN FILE CONTENTS........................................................................'
                    #cs_fgmean.list_variables()
                    #print

                    cs_fgspread_file = xp.cosmo_filename( xp.fcst_start_times[itime], prefix='lff', suffix='spread' )
                    print('--> loading fg spread cosmo file '+ cs_fgspread_file)
                    cs_fgspread = CosmoState(cs_fgspread_file,preload=preload_vars, translation=translation )
                    #print 'FGSPREAD FILE CONTENTS........................................................................'
                    #cs_fgspread.list_variables()
                    #print

                    if True : # ........................................................................................

                        anaens_vars = ['CLC','RELHUM']

                        print('reading full ANA ensemble for variables '+ (' '.join(anaens_vars)) + '...')

                        cachedir =  xp.settings['PLOT_DIR']+'/cache'
                        if not os.path.exists(cachedir) :
                            os.makedirs(cachedir)

                        cachefname = cachedir + '/singleobs_anaens_'+time+'.pickle'

                        if not os.path.exists(cachefname) or args.recompute :

                            starttime = tttime.clock()
                            print()
                            print('reading analysis model state ensemble (lffs) for time %s ...'%time)
                            cs_anaens = []
                            for m in range(xp.n_ens) :
                                cs_anaens_file = xp.cosmo_filename( time, prefix='lfff', member=m+1, output_time=0, suffix='' )
                                print(' -- reading %s ...' % cs_anaens_file)
                                cs_anaens.append( CosmoState( cs_anaens_file, preload=anaens_vars, translation=translation ) )
                            print()

                            # compute mean
                            print('computing mean of ana model state ensemble for '+(' '.join(anaens_vars)))
                            cs_anaens_mean = {}
                            for m in range(xp.n_ens) :
                                for v in anaens_vars :
                                    if not v in cs_anaens_mean :
                                        cs_anaens_mean[v] = cs_anaens[m][v] / xp.n_ens
                                    else :
                                        cs_anaens_mean[v] += cs_anaens[m][v] / xp.n_ens
                            for v in anaens_vars :
                                print('mean ana mean ', v, cs_anaens_mean[v].mean())

                            # compute spread
                            print('computing spread of ana model state ensemble for '+(' '.join(anaens_vars)))
                            cs_anaens_spread = {}
                            for m in range(xp.n_ens) :
                                for v in anaens_vars :
                                    if not v in cs_anaens_spread :
                                        cs_anaens_spread[v] =  (cs_anaens[m][v] - cs_anaens_mean[v])**2
                                    else :
                                        cs_anaens_spread[v] += (cs_anaens[m][v] - cs_anaens_mean[v])**2
                            for v in anaens_vars :
                                cs_anaens_spread[v] = sqrt( cs_anaens_spread[v] / (1 + xp.n_ens) )
                                print('mean ana spread ', v, cs_anaens_spread[v].mean())

                            print('saving %s ...' % cachefname)
                            with open(cachefname,'w') as f :
                                pickle.dump( (cs_anaens,cs_anaens_mean,cs_anaens_spread), f, pickle.HIGHEST_PROTOCOL )
                            print('done (took %f sec).' % (tttime.clock() - starttime))

                        else :
                            starttime = tttime.clock()
                            print('loading %s ...' % cachefname)
                            with open(cachefname,'r') as f :
                                cs_anaens, cs_anaens_mean, cs_anaens_spread = pickle.load(f)
                            print('done (took %f sec).' % (tttime.clock() - starttime))
                        print()


                    if True : #.........................................................................................

                        read_qci_ens = False
                        if read_qci_ens :
                            fgens_vars_2d = ['RELHUM','CLC','QC','QI']
                        else :
                            fgens_vars_2d = ['RELHUM','CLC']
                        fgens_vars_1d = ['TQC','TQI','TQV']
                        fgens_vars = fgens_vars_1d + fgens_vars_2d

                        print('reading full FG ensemble for variables '+ (' '.join(fgens_vars)) + '...')

                        cachedir =  xp.settings['PLOT_DIR']+'/cache'
                        if not os.path.exists(cachedir) :
                            os.makedirs(cachedir)


                        if read_qci_ens :
                            cachefname = cachedir + '/singleobs_fgens_'+('_'.join(fgens_vars))+'_time'+time+'.pickle'
                        else :
                            cachefname = cachedir + '/singleobs_fgens_'+time+'.pickle'

                        #print 'FIXMEEEEEEEEEE SETTING N_ENS TO 4!'
                        #xp.n_ens = 4

                        if not os.path.exists(cachefname) or args.recompute :

                            starttime = tttime.clock()
                            print()
                            print('reading first guess model state ensemble...')
                            cs_fgens = []
                            for m in range(xp.n_ens) :
                                cs_fgens_file = xp.cosmo_filename( xp.fcst_start_times[itime], prefix='lff', member=m+1 )
                                print(' -- reading %s ...' % cs_fgens_file)
                                cs_fgens.append( CosmoState( cs_fgens_file, preload=fgens_vars, translation=translation ) )
                            print('...done')
                            print()

                            # compute mean
                            print('computing mean of first guess model state ensemble for '+(' '.join(fgens_vars)))
                            cs_fgens_mean = {}
                            for m in range(xp.n_ens) :
                                for v in fgens_vars :
                                    if not v in cs_fgens_mean :
                                        cs_fgens_mean[v] = cs_fgens[m][v] / xp.n_ens
                                    else :
                                        cs_fgens_mean[v] += cs_fgens[m][v] / xp.n_ens
                            for v in fgens_vars :
                                print('mean fg mean ', v, cs_fgens_mean[v].mean())

                            # compute spread
                            print('computing spread of first guess model state ensemble for '+(' '.join(fgens_vars)))
                            cs_fgens_spread = {}
                            for m in range(xp.n_ens) :
                                for v in fgens_vars :
                                    if not v in cs_fgens_spread :
                                        cs_fgens_spread[v] =  (cs_fgens[m][v] - cs_fgens_mean[v])**2
                                    else :
                                        cs_fgens_spread[v] += (cs_fgens[m][v] - cs_fgens_mean[v])**2
                            for v in fgens_vars :
                                cs_fgens_spread[v] = sqrt( cs_fgens_spread[v] / (1 + xp.n_ens) )
                                print('mean fg spread ', v, cs_fgens_spread[v].mean())

                            print('done. Reading first guess model state ensemble took %f sec...' % (tttime.clock() - starttime))
                            print()

                            starttime = tttime.clock()
                            print('saving %s ...' % cachefname)
                            with open(cachefname,'w') as f :
                                pickle.dump( (cs_fgens,cs_fgens_mean,cs_fgens_spread), f, pickle.HIGHEST_PROTOCOL )
                            print('done (took %f sec).' % (tttime.clock() - starttime))

                        else :
                            starttime = tttime.clock()
                            print('loading %s ...' % cachefname)
                            with open(cachefname,'r') as f :
                                cs_fgens, cs_fgens_mean, cs_fgens_spread = pickle.load(f)
                            print('done (took %f sec).' % (tttime.clock() - starttime))
                        print()
                    #...................................................................................................

                    # compute increments
                    mvars = list(maxinc.keys())
                    increments = {}
                    for mvar in mvars :
                        increments[mvar] = cs_anamean[mvar] - cs_fgmean[mvar]
                        print('--', mvar, increments[mvar].min(), increments[mvar].max(), end=' ')
                        if 'validityTime' in cs_anamean.meta[mvar] : print(cs_anamean.meta[mvar]['validityTime'], end=' ')
                        if 'validityTime' in cs_fgmean.meta[mvar]  : print(cs_fgmean.meta[mvar]['validityTime'], end=' ')
                        print()
                    print()
                    nlat, nlon, nz = cs_anamean[mvars[0]].shape

                    print('ANA MEAN validityTime = ', cs_anamean.meta['QC']['validityTime'], cs_anamean.filename)
                    print('FG  MEAN validityTime = ', cs_fgmean.meta[ 'QC']['validityTime'], cs_fgmean.filename)

                    print()
                    print('generating horizontal overview plots...')

                    # check orientation
                    #fig = plt.figure(figsize=(7,7))
                    #ax  = fig.add_subplot(1, 1, 1)
                    #img = ax.imshow( cs_fgmean['ALB_DIF'], origin='lower' )
                    #plt.colorbar(img,shrink=0.5)
                    #ax.set_title('albedo')
                    #fig.savefig( "%s_albedo_latlon.png" % time, bbox_inches='tight' )

                    imdls, jmdls = cs_fgmean.cosmo_indices( lat, lon )
                    for mvar in mvars :
                        mainc = abs(increments[mvar]).max(axis=2)
                        fig = plt.figure(figsize=(7,7))
                        ax  = fig.add_subplot(1, 1, 1)
                        img = ax.imshow( mainc, origin='lower' )
                        #ax.scatter( jmdls, imdls, color=None, edgecolor='w', s=1000 )
                        for i in range(n_obs) :
                            ax.text( jmdls[i], imdls[i], str(i), ha='center', va='center', fontsize=10, color='w')
                        ax.contour( cs_fgmean['RLAT'], levels=list(range(40,60,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        ax.contour( cs_fgmean['RLON'], levels=list(range(0,30,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        plt.colorbar(img,shrink=0.5)
                        ax.set_title(mvar+' vert.max. increments of ens. mean')
                        fig.savefig( "%s_vertmax_absincr_latlon_%s.%s" % (time,mvar,args.file_type), bbox_inches='tight' )
                        plt.close(fig)

                        fig = plt.figure(figsize=(7,7))
                        ax  = fig.add_subplot(1, 1, 1)
                        img = ax.imshow( abs(cs_fgspread[mvar]).max(axis=2), origin='lower' )
                        #ax.scatter( jmdls, imdls, color=None, edgecolor='w', s=1000 )
                        ax.contour( cs_fgmean['RLAT'], levels=list(range(40,60,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        ax.contour( cs_fgmean['RLON'], levels=list(range(0,30,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        plt.colorbar(img,shrink=0.5)
                        ax.set_title(mvar+' vert.max. fg spread')
                        fig.savefig( "%s_vertmax_fgspread_latlon_%s.%s" % (time,mvar,args.file_type), bbox_inches='tight' )
                        plt.close(fig)

                        fig = plt.figure(figsize=(7,7))
                        ax  = fig.add_subplot(1, 1, 1)
                        img = ax.imshow( abs(cs_anaspread[mvar]).max(axis=2), origin='lower' )
                        #ax.scatter( jmdls, imdls, color=None, edgecolor='w', s=1000 )
                        ax.contour( cs_anamean['RLAT'], levels=list(range(40,60,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        ax.contour( cs_anamean['RLON'], levels=list(range(0,30,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        plt.colorbar(img,shrink=0.5)
                        ax.set_title(mvar+' vert.max. ana spread')
                        fig.savefig( "%s_vertmax_anaspread_latlon_%s.%s" % (time,mvar,args.file_type), bbox_inches='tight' )
                        plt.close(fig)

                        fig = plt.figure(figsize=(7,7))
                        ax  = fig.add_subplot(1, 1, 1)
                        img = ax.imshow( abs(cs_fgmean[mvar]).max(axis=2), origin='lower' )
                        #ax.scatter( jmdls, imdls, color=None, edgecolor='w', s=1000 )
                        ax.contour( cs_fgmean['RLAT'], levels=list(range(40,60,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        ax.contour( cs_fgmean['RLON'], levels=list(range(0,30,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        plt.colorbar(img,shrink=0.5)
                        ax.set_title(mvar+' vert.max. fg mean')
                        fig.savefig( "%s_vertmax_fgmean_latlon_%s.%s" % (time,mvar,args.file_type), bbox_inches='tight' )
                        plt.close(fig)

                        fig = plt.figure(figsize=(7,7))
                        ax  = fig.add_subplot(1, 1, 1)
                        img = ax.imshow( abs(cs_anamean[mvar]).max(axis=2), origin='lower' )
                        #ax.scatter( jmdls, imdls, color=None, edgecolor='w', s=1000 )
                        ax.contour( cs_anamean['RLAT'], levels=list(range(40,60,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        ax.contour( cs_anamean['RLON'], levels=list(range(0,30,1)), colors='w', linewidths=(2,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5), alpha=0.3 )
                        plt.colorbar(img,shrink=0.5)
                        ax.set_title(mvar+' vert.max. ana mean')
                        fig.savefig( "%s_vertmax_anamean_latlon_%s.%s" % (time,mvar,args.file_type), bbox_inches='tight' )
                        plt.close(fig)
                        #lvl =

                    print()
                    print('generating plots for individual observation locations...')

                    if args.points == 'all' and args.location == '' :
                        points = arange(lat.size)
                    else :
                        if args.location != '' :
                            lonobs, latobs = list(map( float, args.location.split(',') ))
                            print('observation closest to lon=%f, lat=%f : ' % (lonobs,latobs), end=' ')
                            points = [ argmin(abs(lat-latobs)+abs(lon-lonobs)) ]
                            print(points)
                        else :
                            points = array( list(map( int, args.points.split(',') )) )

                    cwd = os.getcwd()
                    for iobs in points : # range(n_obs)

                        obsdir = cwd+'/obs%03d'%(iobs)
                        if not os.path.exists(obsdir) :
                            os.mkdir(obsdir)

                        print()
                        print('.'*80)
                        print('plotting increments for observation #%d at lat=%f, lon=%f...' % (iobs, lat[iobs], lon[iobs]))
                        imdl, jmdl = cs_fgmean.cosmo_indices( lat[iobs], lon[iobs] )
                        print('...corresponding to COSMO grid indices %d,%d' % (imdl,jmdl))
                        print('...at lat=%f, lon=%f ' % ( cs_fgmean['RLAT'][imdl,jmdl], cs_fgmean['RLON'][imdl,jmdl] ))

                        # plot vertical cuts
                        n_pts = 100 # -> 280km
                        nlev = 21

                        # first index <-> lat
                        imin = imdl-n_pts//2
                        imax = imdl+n_pts//2

                        # second index <-> lon
                        jmin = jmdl-n_pts//2
                        jmax = jmdl+n_pts//2

                        #print 'variation of lat / lon with first  index : ', cs_fgmean['RLAT'][imin:imax+1, jmdl], cs_fgmean['RLON'][imin:imax+1, jmdl]
                        #print 'variation of lat / lon with second index : ', cs_fgmean['RLAT'][imdl, jmin:jmax+1], cs_fgmean['RLON'][imdl, jmin:jmax+1]

                        #lonax = cs_fgmean['RLON'][imdl, jmin:jmax+1]
                        #lonax_label = 'longitude [deg]'
                        lonax = arange( -n_pts//2, n_pts//2+0.5 )*2.8
                        lonax_label = 'zonal distance from observation position [km]'
                        #print 'shape lonax ', lonax.shape

                        print('>>> ', imin, imax, jmin, jmax)


                        # model grid range corresponding to superobb size
                        sx, sy = list(map( int, xp.settings['VISOP_SUPEROBB'].split(',')[:2] ))
                        sy *= 2
                        sil = -sx//2
                        sih = sil + sx
                        sjl = -sy//2
                        sjh = sjl + sy
                        print('SUPEROBBING region size in model indices : ', sx, sy, ' --> relative index ranges ', sil, sih, sjl, sjh)

                        # extract model var profiles at observation location
                        fgmean_so, anamean_so, fgspread_so, anaspread_so = {}, {}, {}, {}
                        for mvar in mvars :
                            if True : # superobbing area
                                fgmean_so[   mvar] = cs_fgmean[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                                anamean_so[  mvar] = cs_anamean[mvar][imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                                fgspread_so[ mvar] = cs_fgspread[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                                anaspread_so[mvar] = cs_anaspread[mvar][imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                            else : # single column
                                fgmean_so[   mvar] = cs_fgmean[mvar][  imdl,jmdl,: ]
                                anamean_so[  mvar] = cs_anamean[mvar][ imdl,jmdl,: ]
                                fgspread_so[ mvar] = cs_fgspread[mvar][  imdl,jmdl,: ]
                                anaspread_so[mvar] = cs_anaspread[mvar][ imdl,jmdl,: ]

                        # RELHUM, CLC : not in l?f*mean, l?f*.spread -> use values computed from the member files
                        for mvar in ['RELHUM','CLC'] :
                            fgmean_so[mvar] = cs_fgens_mean[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                            fgspread_so[mvar] = cs_fgens_spread[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                            anamean_so[mvar] = cs_anaens_mean[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))
                            anaspread_so[mvar] = cs_anaens_spread[mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh,:].mean(axis=(0,1))

                        # average vertically integrated variables over superobb region
                        fgens_intvar = {}
                        for mvar in ['TQC','TQI','TQV'] :
                            fgens_intvar[mvar] = zeros(xp.n_ens)
                            for m in range(xp.n_ens) :
                                fgens_intvar[mvar][m] = cs_fgens[m][mvar][ imdl+sil:imdl+sih,jmdl+sjl:jmdl+sjh].mean()

                        # compute correlation coefficients between reflectance and vertically integrated variables
                        print()
                        print('Computing correlation coefficients...')
                        fgens_corcoef = {}
                        for mvar in ['TQC','TQI','TQV'] :
                            #print 'cor shape ', fgens[:,iobs].ravel().shape, fgens_intvar[mvar].shape
                            #print '          ', stack(fgens[:,iobs].ravel(),fgens_intvar[mvar]).shape
                            #fgens_corcoef[mvar] = corrcoef( stack(fgens[:,iobs].ravel(),fgens_intvar[mvar]))
                            fgens_corcoef[mvar] = stats.linregress( fgens_intvar[mvar], fgens[:,iobs].ravel() ).rvalue
                            print(' -- %s : %f' % (mvar,fgens_corcoef[mvar]))
                        print()

                        #


                        # summary profile plot .........................................................................

                        summary_profile_plot( iobs, xp, lat, lon, obs, fgens, fgmean, anaens, anaens_real, anamean, anamean_real,
                                              fgmean_so, anamean_so, fgspread_so, anaspread_so,
                                              tauw_mean_fg, tauw_spread_fg, tauw_mean_ana, tauw_spread_ana,
                                              taui_mean_fg, taui_spread_fg, taui_mean_ana, taui_spread_ana,
                                              clc_tot_mean_fg, clc_tot_spread_fg, clc_tot_mean_ana, clc_tot_spread_ana,
                                              ekf_temp, temp_rep_close_to_obs, temp_errors,
                                              obsdir, time, args )

                        col = {'QC':'r', 'QI':'b', 'U':'#ff9900', 'V':'#009999', 'T':'#cc00cc', 'QV':'#009900', 'RELHUM':'#009900', 'CLC':'#006699',
                               'TQC':'r', 'TQI':'b', 'TQV':'#009999'}
                        lw=3.0
                        lwn=2.0
                        fig, ax = plt.subplots( 1, 6, figsize=(22,7)) #sharey=True,

                        iaxx=0
                        axx = ax[iaxx]
                        if False : # reflectance histograms
                            refl_bins = linspace(0,1.0,11)
                            fghist,       edges = histogram( fgens[:,iobs],       bins=refl_bins )
                            anahist,      edges = histogram( anaens[:,iobs],      bins=refl_bins )
                            anahist_real, edges = histogram( anaens_real[:,iobs], bins=refl_bins )
                            print('edges=', edges)
                            #ax[3].plot( edges[:-1], fghist, 'b', label='FG' )
                            #ax[3].plot( edges[:-1], anahist, '#990000', label='ANA (linear)' )
                            #ax[3].plot( edges[:-1], anahist_real, '#ff0000', label='ANA (nonlin.)' )
                            binplot( edges-0.005, fghist, color='b', label='FG', ax=axx, linewidth=lw )
                            binplot( edges, anahist, color='#990000', label='ANA (linear)', ax=axx, linewidth=lw )
                            binplot( edges+0.005, anahist_real, color='#ff0000', label='ANA (nonlin.)', ax=axx, linewidth=lw )
                            axx.set_ylim(( 0, ax[3].get_ylim()[1]*1.5 ))
                            axx.plot( [obs[iobs]]*2, ax[3].get_ylim(), '#009900', label='OBS', linewidth=lw )
                            axx.set_xlim(( edges[0], edges[-1] ))
                            axx.grid()
                            axx.legend()
                            axx.set_title('reflectance histograms')
                            axx.set_xlabel('reflectance')
                            axx.set_ylabel('#members')
                        else : # reflectance scatter plots

                            c='#009900'
                            axx.barh( obs[iobs], 3, float(xp.settings['VISOP_ERROR']), 0.5, align='center', color=c,
                                      alpha=0.1, edgecolor='' )
                            axx.plot( (0.5,3.5), [obs[iobs]]*2, color=c, linewidth=lw )
                            axx.text( 0.6, obs[iobs]+0.01, 'OBS', color=c )

                            c='b'
                            axx.scatter( [1]*fgens.shape[0], fgens[:,iobs], color=c, alpha=0.3, s=40 )
                            axx.plot( (0.5,1.5), [fgmean[iobs]]*2, color=c, linewidth=lw )

                            c='#990000'
                            axx.scatter( [2]*fgens.shape[0], anaens[:,iobs], color=c, alpha=0.3, s=40 )
                            axx.plot( (1.5,2.5), [anamean[iobs]]*2, color=c, linewidth=lw )

                            c='#ff0000'
                            axx.scatter( [3]*fgens.shape[0], anaens_real[:,iobs], color=c, alpha=0.3, s=40 )
                            axx.plot( (2.5,3.5), [anamean_real[iobs]]*2, color=c, linewidth=lw )


                            axx.set_ylim((0,1.0))
                            axx.set_ylabel('reflectance')

                            axx.set_xlim((0.5,3.5))
                            axx.set_xticks([1,2,3])
                            axx.set_xticklabels(['FG','ANA(lin.)','ANA'])

                            axx.set_title('reflectance ensembles')

                        if True :
                            iaxx+=1
                            axx = ax[iaxx]
                            for imvar, mvar in enumerate(['TQC','TQI']) : #,'TQV'

                                # correlation
                                cr = stats.linregress( fgens_intvar[mvar], fgens[:,iobs].ravel() )
                                req_change = cr.slope*(obs[iobs]-fgmean[iobs]) / fgens_intvar[mvar].mean()
                                req_change_abs = cr.slope*(obs[iobs]-fgmean[iobs])
                                print('required reflectance change : %f, dR/d%s = %f ' % (obs[iobs]-fgmean[iobs], mvar, cr.slope))
                                print('mean / std %s : %f / %f' % (mvar, fgens_intvar[mvar].mean(), fgens_intvar[mvar].std()))
                                print('required relative change in %s to compensate for reflectance error %f : %f' % (mvar,fgmean[iobs]-obs[iobs],req_change))

                                mvar_fit = linspace(fgens_intvar[mvar].min(),fgens_intvar[mvar].max(),10)
                                refl_fit    = cr.intercept + cr.slope*mvar_fit
                                axx.plot( mvar_fit/varscl[mvar], refl_fit, color=col[mvar], alpha=0.5 )

                                axx.scatter( fgens_intvar[mvar]/varscl[mvar], fgens[:,iobs].ravel(), color=col[mvar], label='%s %6f %6f %6f' % (mvar,fgens_intvar[mvar].mean(),req_change_abs,req_change) )

                            axx.plot( (0,0.1), [obs[iobs]]*2, '#009900', label='OBS', linewidth=lw ) #axx.get_xlim()
                            axx.plot( (0,0.1), [fgmean[iobs]]*2, '--', color='#999999', label='FG', linewidth=lw )
                            axx.plot( (0,0.1), [anamean_real[iobs]]*2, color='#999999', label='ANA', linewidth=lw )
                            axx.set_ylim((0,1))
                            axx.set_xlim(left=0)
                            axx.legend(frameon=False,fontsize=6)
                            axx.set_title('correlations')
                            axx.set_ylabel('reflectance')
                            axx.set_xlabel('TQC, TQI, TQV')
                            axx.grid()

                        iaxx+=1
                        axx = ax[iaxx]
                        for imvar, mvar in enumerate(['QC','QI']) :
                            # spread shading
                            fgm = fgmean_so[mvar]/varscl[mvar]
                            fgs = fgspread_so[mvar]/varscl[mvar]
                            fgl = fgm - fgs/2
                            fgl[where(fgl<0)] = 0
                            fgr = fgm + fgs/2
                            axx.fill_betweenx( anamean_so['P']/100, fgl, fgr, color=col[mvar], alpha=0.05, zorder=imvar-10 )
                            #axx.plot( fgl, fgmean_so['P']/100,  color=col[mvar], linewidth=1, alpha=0.5  )
                            #axx.plot( fgr, fgmean_so['P']/100,  color=col[mvar], linewidth=1, alpha=0.5  )

                            agm = anamean_so[mvar]/varscl[mvar]
                            ags = anaspread_so[mvar]/varscl[mvar]
                            agl = agm - ags/2
                            agl[where(agl<0)] = 0
                            agr = agm + ags/2
                            axx.fill_betweenx( anamean_so['P']/100, agl, agr, color=col[mvar], alpha=0.1, zorder=imvar-5 )

                            axx.plot( anamean_so[mvar]/varscl[mvar], anamean_so['P']/100, color=col[mvar], label=mvar, linewidth=lw )
                            axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  color=col[mvar], linewidth=lwn )
                            axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  '--', color='w', linewidth=lwn*0.8 )

                        axx.set_ylim((1000,100))

                        xlm = axx.get_xlim()
                        axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 950, 'W %.1f (%.1f) -> %.1f (%.1f)' % (
                                  tauw_mean_fg[iobs], tauw_spread_fg[iobs], tauw_mean_ana[iobs], tauw_spread_ana[iobs]),
                                  fontsize=10, ha='right' )
                        axx.text( xlm[1]-0.05*(xlm[1]-xlm[0]), 900, 'I %.1f (%.1f) -> %.1f (%.1f)' % (
                                  taui_mean_fg[iobs], taui_spread_fg[iobs], taui_mean_ana[iobs], taui_spread_ana[iobs]),
                                  fontsize=10, ha='right' )

                        axx.legend(frameon=False, fontsize=10)

                        #axx.set_title('fg = dashed, ana = solid')
                        corrs = 'corr: '
                        for mvar in ['TQC','TQI','TQV'] :
                            corrs += ('%s:%.2f ' % (mvar,fgens_corcoef[mvar]))
                        axx.set_title(corrs)
                        
                        axx.set_xlabel('QC, QI [1e-5]')
                        #axx.set_ylabel('P [hPa]')
                        axx.grid()

                        if True :
                            iaxx+=1
                            axx = ax[iaxx]
                            for imvar, mvar in enumerate(['CLC','RELHUM']) :
                                # spread shading
                                fgm = fgmean_so[mvar]/varscl[mvar]
                                fgs = fgspread_so[mvar]/varscl[mvar]
                                fgl = fgm - fgs/2
                                fgl[where(fgl<0)] = 0
                                fgr = fgm + fgs/2
                                axx.fill_betweenx( anamean_so['P']/100, fgl, fgr, color=col[mvar], alpha=0.05, zorder=imvar-10 )
                                #axx.plot( fgl, fgmean_so['P']/100,  color=col[mvar], linewidth=1, alpha=0.5  )
                                #axx.plot( fgr, fgmean_so['P']/100,  color=col[mvar], linewidth=1, alpha=0.5  )

                                #if mvar != 'CLC' :
                                agm = anamean_so[mvar]/varscl[mvar]
                                ags = anaspread_so[mvar]/varscl[mvar]
                                agl = agm - ags/2
                                agl[where(agl<0)] = 0
                                agr = agm + ags/2
                                axx.fill_betweenx( anamean_so['P']/100, agl, agr, color=col[mvar], alpha=0.1, zorder=imvar-5 )

                                axx.plot( anamean_so[mvar]/varscl[mvar], anamean_so['P']/100, color=col[mvar], linewidth=lw )

                                axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  color=col[mvar], label=mvar, linewidth=lwn )
                                axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  '--', color='w', linewidth=lwn*0.8 )

                            if iobs in temp_rep_close_to_obs :
                                print('FOUND RADIOSONDE! ', iobs, temp_rep_close_to_obs[iobs])
                                irep = temp_rep_close_to_obs[iobs]
                                vname = 'RH'
                                ekf_temp.replace_filter(filter='state=all varname=%s report=%d'%(vname,irep))
                                p_temp = ekf_temp.obs(param='plevel')/100
                                axx.plot( 100*ekf_temp.obs(),     p_temp, 'k', label='TEMP OBS')
                                axx.plot( 100*ekf_temp.fgmean(),  p_temp, 'b', label='TEMP FG')
                                axx.plot( 100*ekf_temp.anamean(), p_temp, 'r', label='TEMP ANA')

                            axx.set_ylim((1000,100))
                            axx.set_xlim((0,100))
                            axx.legend(frameon=False,fontsize=10)
                            axx.set_title('fg = dashed, ana = solid')
                            axx.set_xlabel('CLC, RH [%]')
                            #axx.set_ylabel('P [hPa]')
                            axx.grid()


                        iaxx+=1
                        axx = ax[iaxx]
                        for mvar in ['U','V'] :
                            axx.plot( anamean_so[mvar]/varscl[mvar], anamean_so['P']/100, color=col[mvar], label=mvar, linewidth=lw )
                            axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  color=col[mvar], linewidth=lwn )
                            axx.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  '--', color='w', linewidth=lwn*0.8 )
                        axx.set_ylim((1000,100))
                        axx.plot( (0,0), (1000,100), 'k', alpha=0.5, zorder=0 )
                        axx.legend(frameon=False,fontsize=10)
                        axx.set_title('fg = dashed, ana = solid')
                        axx.set_xlabel('U, V [m/s]')
                        axx.grid()

                        iaxx+=1
                        axx = ax[iaxx]
                        for mvar in ['T','QV'] :
                            axx.plot( increments[mvar][imdl,jmdl,:]/varscl[mvar], fgmean_so['P']/100, color=col[mvar], label=mvar, linewidth=lw )
                        axx.set_ylim((1000,100))
                        axx.plot( (0,0), (1000,100), 'k', alpha=0.5, zorder=0 )
                        axx.set_xlim((-1.5,1.5))

                        axx.text( 1.4, 800, 'TEMP RH %.5f' % temp_errors[iobs]['RH_delta_rmse'], ha='right', fontsize=10 )
                        axx.text( 1.4, 850, 'TEMP T  %.5f' % temp_errors[iobs]['T_delta_rmse'], ha='right', fontsize=10 )
                        axx.text( 1.4, 900, 'TEMP U  %.5f' % temp_errors[iobs]['U_delta_rmse'], ha='right', fontsize=10 )
                        axx.text( 1.4, 950, 'TEMP V  %.5f' % temp_errors[iobs]['V_delta_rmse'], ha='right', fontsize=10 )

                        axx.legend(frameon=False,fontsize=10)
                        axx.set_title('increments')
                        axx.set_xlabel('T [K], QV [1e-3]')
                        axx.grid()

                        fig.suptitle("%s : obs #%03d (lon=%5.1f, lat=%5.1f) : obs=%4.2f, fgmean=%4.2f, fgspread=%4.2f, anamean=%4.2f (%4.2f)" % \
                                     ( xp.expid, iobs, lat[iobs], lon[iobs], obs[iobs], fgmean[iobs], fgspread[iobs], anamean[iobs], anamean_real[iobs] ) )

                        fig.savefig( "%s/profiles_%s_obs%d.%s" % (obsdir,time,iobs,args.file_type), bbox_inches='tight' )
                        plt.close(fig)

                        # reflectance -- model state correlations plots
                        #fig, ax = plt.subplots( 1, 3, figsize=(16,7)) #sharey=True,
                        #axx = ax[0]
                        #axx.plot( fgens[iobs], cs_fgmean['TQC'][imdl,jmdl] )
                        #axx = ax[1]
                        #axx.plot( fgens[iobs], cs_fgmean['TQI'][imdl,jmdl] )
                        #axx = ax[2]
                        #axx.plot( fgens[iobs], cs_fgmean['TQV'][imdl,jmdl] )
                        #fig.savefig( "%s/reflcorr_%s_obs%d.png" % (obsdir,time,iobs), bbox_inches='tight' )
                        #plt.close(fig)

                        for mvar in increments :

                            # increments
                            if jmin > 0 and jmax < increments[mvar].shape[1] :

                                fig = plt.figure(figsize=(7,7))
                                ax  = fig.add_subplot(1, 1, 1)
                                #print 'sh-apes ', lonax.shape, nz+1, transpose(increments[mvar][ imdl, jmin:jmax+1, ::-1 ])[:nz,:].shape, increments[mvar].shape
                                img = ax.contourf( lonax, arange(nz)+1, transpose(increments[mvar][ imdl, jmin:jmax+1, ::-1 ])[:nz,:],
                                                   levels=linspace(-maxinc[mvar],maxinc[mvar],nlev), cmap='RdBu_r' )
                                cs  = ax.contour(  lonax, arange(nz)+1, transpose(cs_fgmean['P'][ imdl, jmin:jmax+1, ::-1 ])[:nz,:]/1e2, levels=arange(0,1e3+1,1e2), colors='#999999' )
                                plt.clabel(cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10)
                                plt.colorbar(img,shrink=0.5)
                                ax.set_xlabel(lonax_label)
                                ax.set_ylabel('model level')
                                ax.set_title(mvar+' increments')
                                fig.savefig( "%s/%s_incr_latconst_%s_obs%d.%s" % (obsdir,time,mvar,iobs,args.file_type) )
                                plt.close(fig)

                                # fg and ana mean
                                for iveri,veri in enumerate((cs_fgmean,cs_anamean,cs_fgspread)) :
                                    fig = plt.figure(figsize=(7,7))
                                    ax  = fig.add_subplot(1, 1, 1)
                                    img = ax.contourf( lonax, arange(nz)+1, transpose(veri[mvar][ imdl, jmin:jmax+1, ::-1 ])[:nz,:], cmap='gist_ncar' )
                                    cs = ax.contour( lonax, arange(nz)+1, transpose(veri['P'][ imdl, jmin:jmax+1, ::-1 ])[:nz,:]/1e2, levels=arange(0,1e3+1,1e2), colors='#999999' )
                                    plt.clabel(cs, cs.levels, inline=True, fontsize=10)
                                    plt.colorbar(img,shrink=0.5)
                                    ax.set_xlabel(lonax_label)
                                    ax.set_ylabel('model level')
                                    ax.set_title(['FG mean ','ANA mean','FG spread'][iveri]+mvar)
                                    fig.savefig( "%s/%s_%s_latconst_%s_obs%d.%s" % (obsdir,time,['fgmean','anamean','fgspread'][iveri],mvar,iobs,args.file_type) )
                                    plt.close(fig)

                            # increment profile
                            if increments[mvar].shape[2] == cs_fgmean['P'].shape[2] :
                                fig, ax = plt.subplots(figsize=(5,8))
                                ax.plot( increments[mvar][imdl,jmdl,:], fgmean_so['P']/100 )
                                ax.set_xlim((-maxinc[mvar],maxinc[mvar]))
                                ax.set_ylim((1000,100))
                                ax.set_title(mvar)
                                ax.grid()
                                fig.savefig( "%s/%s_incr_prof_%s_obs%d.%s" % (obsdir,time,mvar,iobs,args.file_type) )
                                plt.close(fig)

                                fig, ax = plt.subplots(figsize=(5,8))
                                ax.plot( anamean_so[mvar]/varscl[mvar], anamean_so['P']/100, 'r' )
                                ax.plot( fgmean_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  'b' )
                                ax.plot( anaspread_so[mvar]/varscl[mvar], anamean_so['P']/100, '--r' )
                                ax.plot( fgspread_so[mvar]/varscl[mvar],  fgmean_so['P']/100,  '--b' )
                                #ax.set_xlim((-maxinc[mvar],maxinc[mvar]))
                                ax.set_ylim((1000,100))
                                ax.set_title('%s [%f]'%(mvar,varscl[mvar]))
                                ax.grid()
                                fig.savefig( "%s/%s_anafg_prof_%s_obs%d.%s" % (obsdir,time,mvar,iobs,args.file_type) )
                                plt.close(fig)



