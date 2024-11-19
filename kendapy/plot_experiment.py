#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P L O T _ E X P E R I M E N T
#  postprocess KENDA experiments
#
#  2016.10 L.Scheck 

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
import pylab as plt
#from mpl_toolkits.basemap import Basemap, cm
import matplotlib.gridspec as gridspec

import sys, os, argparse
from numpy import *

from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.binplot import binplot
import kendapy.ekf
from kendapy.colseq import colseq

#--------------------------------------------------------------------------------
def plot_error_overview( experiments, names=None, colors=None, times=[], obs_types=[], variables=[],
                         n_obs_min=100, output_path='', file_type='png', verbose=False, **filter_kw ) :

    import colorsys

    if not type(experiments) == list :
        xps = [experiments]
        compare = False
        if colors is None : colors = ['k']
    else :
        xps = experiments
        compare = True
        #colors = ['r','#009900','#0066ff','#ff9900','#ff00ff','#009966']
        #hues = arange(len(experiments))/float(len(experiments))
        #if colors is None : colors = [ colorsys.hls_to_rgb( hue, 0.6-0.2*exp(-((hue-0.35)/0.1)**2), 0.7 ) for hue in hues ]

        if colors is None :
            colors = [ colseq(i,len(experiments)) for i in range(len(experiments)) ]

    if verbose :
        print(('>>> plot_error_overview : experiments : ', [xp.expid for xp in xps]))
        print(('>>>                       filters     : ', filter_kw))

    if compare :
        print(('>>> plot_error_overview : comparing', [ xp.settings['exp'] for xp in xps ]))

    n_exp = len(xps)

    if names is None :
        names = [ xp.settings['exp'] for xp in xps ]

    # collect data
    eevo = {}
    xptimes = {}
    obs_type_var_occurs = {}
    for xp in xps :
        if len(times) == 0 :
            xptimes[xp] = xp.veri_times
        else :
            xptimes[xp] = times

        eevo[xp] = xp.compute_error_evolution( xptimes[xp], obs_types=obs_types, variables=variables, verbose=True,
                                               **filter_kw )

        for obs_type in list(eevo[xp].keys()) :
            if not obs_type in list(obs_type_var_occurs.keys()) :
                obs_type_var_occurs[obs_type] = {}
            for vname in list(eevo[xp][obs_type].keys()) :
                obs_type_var_occurs[obs_type][vname] = True
        if verbose :
            print(('       -- ', xp.expid, obs_type_var_occurs))

    erel_fg = {}
    erel_ana = {}
    nobs_tot = []
    ov = []
    for obs_type in sorted( list(obs_type_var_occurs.keys()), reverse=True ) :
        for vname in sorted(list(obs_type_var_occurs[obs_type].keys()), reverse=True ) :

            print((obs_type, vname))

            fgmean_rmse = {}
            anamean_rmse = {}
            nobs = {}
            nobs_min = None
            for xp in xps :
                times = sorted(eevo[xp][obs_type][vname].keys())
                fgmean_rmse[xp]  = sqrt(array([ eevo[xp][obs_type][vname][t]['fgmean' ]['rmse']**2 for t in times ]).mean())
                anamean_rmse[xp] = sqrt(array([ eevo[xp][obs_type][vname][t]['anamean']['rmse']**2 for t in times ]).mean())
                nobs[xp]         =      array([ eevo[xp][obs_type][vname][t]['n_obs']              for t in times ]).sum()
                nobs_min = minimum( nobs_min, nobs[xp] ) if not nobs_min is None else nobs[xp]

            if nobs_min < n_obs_min :
                continue

            if compare : # copmute relative change in rmse wrt. the reference run xps[0]
                for i, xp in enumerate(xps[1:]) :
                    if not xp in erel_fg : erel_fg[xp] = []
                    erel_fg[xp].append( fgmean_rmse[xp]/fgmean_rmse[xps[0]] - 1.0 )
                    print((erel_fg[xp][-1]))
                    if not xp in erel_ana : erel_ana[xp] = []
                    erel_ana[xp].append( anamean_rmse[xp]/anamean_rmse[xps[0]] - 1.0 )

            nobs_tot.append(nobs)
            ov.append( obs_type+'/'+vname)

    #plt.figure(48,figsize=(8,16))
    width = 0.6 / n_exp       # the width of the bars

    if compare : # .....................................................................................................
        if verbose : print('plotting FG mean rmse change...')
        fig, ax = plt.subplots(figsize=(9,9))
        rects = {}
        for i, xp in enumerate(xps) :
            #for i, xp in enumerate(erel_fg) :
            if not xp in erel_fg : continue
            ind = arange(len(ov))  # the x locations for the groups
            rects[xp] = ax.barh( ind+i*width, erel_fg[xp], width, color=colors[i], edgecolor='k', linewidth=0,
                                 label=names[i] ) #xp.settings['exp'] ) #, orientation='horizontal' )
            for j in range(len(ov)) :
                if i == 1 :
                    ax.text( 0, ind[j]+i*width, " %d "%nobs_tot[j][xp], fontsize=7, va='bottom', ha='right' if erel_fg[xp][j] > 0 else 'left')
        ax.set_xlabel('relative change in mean rmse wrt. '+xps[0].settings['exp'])
        ax.set_xlim((-0.4,0.4))
        ax.set_yticks(ind + width*n_exp/2.0)
        ax.set_yticklabels(ov)
        #plt.legend(frameon=False,loc='upper left',title='FG',fontsize=8)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title='FG',fontsize=10,frameon=False)
        plt.grid()
        plt.savefig(output_path+'error_overview_fgmean.'+file_type, bbox_inches='tight')
        plt.close(fig)

        #plt.clf()
        #fig, ax = plt.subplots()
        if verbose : print('plotting ANA mean rmse change...')
        fig, ax = plt.subplots(figsize=(9,9))
        rects = {}
        for i, xp in enumerate(xps) :
            if not xp in erel_ana : continue
            ind = arange(len(ov))  # the x locations for the groups
            rects[xp] = ax.barh( ind+i*width, erel_ana[xp], width, color=colors[i], edgecolor='k', linewidth=0,
                                 label=names[i] ) #xp.settings['exp'] ) #, orientation='horizontal' )
        ax.set_xlabel('relative change in mean rmse wrt. '+xps[0].settings['exp'])
        ax.set_xlim((-0.4,0.4))
        ax.set_yticks(ind + width*n_exp/2.0)
        ax.set_yticklabels(ov)
        #plt.legend(frameon=False,loc='upper left',title='ANA',fontsize=8)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title='ANA',fontsize=10,frameon=False)
        plt.grid()
        plt.savefig(output_path+'error_overview_anamean.'+file_type, bbox_inches='tight')
        plt.close(fig)

    else :
        print('absolute error overview : implement me!')

#--------------------------------------------------------------------------------
def plot_fofens_overview( experiments, names=None, colors=None, times=None, obs_types=[], variables=[],
                          time_range='151-180', n_obs_min=100, output_path='', file_type='png', recompute=False,
                          verbose=True, **filter_kw ) :
    import colorsys

    if not type(experiments) == list :
        xps = [experiments]
        compare = False
        if colors is None : colors = ['k']
    else :
        xps = experiments
        compare = True
        if colors is None :
            colors = [ colseq(i,len(experiments)) for i in range(len(experiments)) ]

    #expids = [ xp.expid for xp in xps ]

    if compare :
        print(('comparing', [ xp.settings['exp'] for xp in xps ]))

    n_exp = len(xps)

    if names is None :
        names = [ xp.settings['exp'] for xp in xps ]

    # gather data ......................................................................................................
    fofstats = {}
    obs_type_var_occurs = {}
    for xp in xps :
        # compute statistics from fof file ensemble for the given time range
        fofstats[xp.expid] = xp.get_fofens_statistics( times=times, time_range=time_range, obs_types=obs_types,
                                                       variables=variables, lfcst=True, recompute=recompute )
        # --> e.g. fofstats[xp.expid][obs_type][vname][start_time]['rmse']
        if False : #verbose :
            print(('EXPERIMENT %s / t = %s min : ' %(xp.expid,time_range)))
            for obs_type in fofstats[xp.expid] :
                for vname in fofstats[xp.expid][obs_type] :
                    for start_time in fofstats[xp.expid][obs_type][vname] :
                        print(('    %15s / %15s / %s : %5d' % ( obs_type, vname, start_time, fofstats[xp.expid][obs_type][vname][start_time]['n_obs'] )))

        # remember which obstype/varname combinations we have encountered
        for obs_type in fofstats[xp.expid] :
            if not obs_type in obs_type_var_occurs :
                obs_type_var_occurs[obs_type] = {}
            for vname in fofstats[xp.expid][obs_type] :
                obs_type_var_occurs[obs_type][vname] = True

        if verbose :
            print((xp.expid, ' : found ', obs_type_var_occurs))


    # plot .............................................................................................................
    erel = {}
    nobs_tot = []
    ov = []
    for obs_type in sorted(list(obs_type_var_occurs.keys()),reverse=True) :
        for vname in sorted(list(obs_type_var_occurs[obs_type].keys()),reverse=True) :

            print(('*'*50, obs_type, vname))

            mean_rmse = {}
            nobs = {}
            nobs_min = None
            for xp in xps :
                if obs_type in fofstats[xp.expid] and vname in fofstats[xp.expid][obs_type] :
                    times =                     sorted(fofstats[xp.expid][obs_type][vname].keys())
                    nobs[xp.expid] =           array([ fofstats[xp.expid][obs_type][vname][t]['n_obs'] for t in times ]).sum()
                    mean_rmse[xp.expid] = sqrt(array([ fofstats[xp.expid][obs_type][vname][t]['n_obs'] * \
                                                       fofstats[xp.expid][obs_type][vname][t]['rmse']**2 for t in times ]).sum())
                    mean_rmse[xp.expid] = sqrt( mean_rmse[xp.expid] / nobs[xp.expid] )
                    nobs_min = minimum( nobs_min, nobs[xp.expid] ) if not nobs_min is None else nobs[xp.expid]
                else :
                    print(('%s : %s/%s not available -> will be ignored for all experiments...' % (xp.expid,obs_type,vname)))
                    #nobs_min = 0

            if nobs_min < n_obs_min :
                continue

            for i, xp in enumerate(xps[1:]) :
                if not xp.expid in erel : erel[xp.expid] = []
                erel[xp.expid].append( mean_rmse[xp.expid]/mean_rmse[xps[0].expid] - 1.0 )

            nobs_tot.append(nobs)
            ov.append( obs_type+'/'+vname)

    plt.figure(48,figsize=(8,16))
    width = 0.6 / n_exp       # the width of the bars

    plt.clf()
    fig, ax = plt.subplots()
    rects = {}
    for i, xp in enumerate(xps) :
        #for i, xp in enumerate(erel_fg) :
        if not xp.expid in erel : continue
        ind = arange(len(ov))  # the x locations for the groups
        rects[xp.expid] = ax.barh( ind+i*width, erel[xp.expid], width, color=colors[i], edgecolor='k', linewidth=0,
                             label=names[i] ) #xp.settings['exp'] ) #, orientation='horizontal' )
        for j in range(len(ov)) :
            ax.text( 0, ind[j]+i*width, " %d "%nobs_tot[j][xp.expid], fontsize=7, va='bottom', ha='right' if erel[xp.expid][j] > 0 else 'left')
    ax.set_xlabel('relative change in mean rmse wrt. '+xps[0].settings['exp'])
    ax.set_yticks(ind + width*n_exp/2.0)
    ax.set_yticklabels(ov)
    plt.legend(frameon=False,loc='upper left',title='FG')
    plt.grid()
    plt.savefig(output_path+'error_overview_fofens_t'+time_range+'.'+file_type, bbox_inches='tight')

#--------------------------------------------------------------------------------
def plot_fof_error_evolution( xp, times=None, variables=None, obs_types=None, area_filter=None, state_filter=None,
                                      output_path='', file_type='png', lfcst=False, timeresmin=60, **filter_kw ) :
    if times is None :
        times_ = xp.lfcst_start_times
    else :
        times_ = []
        for t in times :
            if t in xp.lfcst_start_times :
                times_.append(t)

    fcint = int(xp.lfcst_settings['FCINT']) # [sec]

    time_range_starts = list(range( 1, fcint//60, timeresmin)) # [min]
    time_ranges = ['%d,%d'%(t,t+timeresmin-1) for t in time_range_starts]
    print(('time_ranges : ', time_ranges))

    fofstat = {}
    for time_range in time_ranges :
        fofstat[time_range] = xp.get_fofens_statistics( time_range=time_range, times=times_, obs_types=obs_types,
                                                        variables=variables, area_filter=area_filter,
                                                        state_filter=state_filter, lfcst=True, recompute=True )
        # --> e.g. fofstat[time_range][obs_type][vname][start_time]['rmse']

    tmn_abs = times_[0]
    tmx_abs = (Time14(times_[-1])+Time14(fcint)).string14()
    tmn = 0
    tmx = (Time14(tmx_abs) - Time14(tmn_abs)).dayhour()

    fig, ax = plt.subplots()
    for time_range in time_ranges :
        for ot in fofstat[time_range] :
            for vn in fofstat[time_range][ot] :
                for start_time in fofstat[time_range][ot][vn] :
                    print(('+++ ', time_range, ot, vn, start_time))
                    t1 = (Time14(start_time)+Time14(int(time_range.split(',')[0])*60)-Time14(tmn_abs)).dayhour()
                    t2 = (Time14(start_time)+Time14(int(time_range.split(',')[1])*60)-Time14(tmn_abs)).dayhour()
                    rmse = fofstat[time_range][ot][vn][start_time]['rmse']
                    n_obs = fofstat[time_range][ot][vn][start_time]['n_obs']
                    ax.plot( (t1,t2), [rmse]*2, 'b' )
                    ax.text( t2, rmse, '%d'%n_obs, color='b' )
                    if start_time in xp.fcst_start_times and time_range == '1,60' :
                        ekf = xp.get_ekf( (Time14(start_time)+Time14(3600)).string14(), ot )
                        rmse_fg = ekf.statistics(varname=vn,recompute=True,desroz=False,rankhist=False)['fgmean']['rmse']
                        n_obs_fg = ekf.statistics(varname=vn,recompute=True,desroz=False,rankhist=False)['n_obs']
                        ax.plot( (t1,t2), [rmse_fg]*2, 'r' )
                        ax.text( t1, rmse_fg, '%d'%n_obs_fg, color='r')


    tickvalues, ticklabels = time_axis_ticks( tmn, tmx, tmn_abs, tmx_abs )
    ax.set_xticks(tickvalues)
    ax.set_xticklabels(ticklabels)
    plt.grid()
    plt.savefig(output_path+'error_evolution_fof.'+file_type, bbox_inches='tight')

#--------------------------------------------------------------------------------
def plot_error_evolution( experiments, names=None, colors=None, times=[], obs_types=[], variables=[],
                          output_path='', file_type='png', lfcst=False, **filter_kw ) :

    if not type(experiments) == list :
        xps = [experiments]
        compare = False
        if colors is None : colors = ['k']
    else :
        xps = experiments
        compare = True
        #if colors is None : colors = ['r','#009900','#0066ff','#ff9900','#ff00ff','#009966']
        if colors is None :
            colors = [ colseq(i,len(experiments)) for i in range(len(experiments)) ]

    if compare :
        print(('comparing', [ xp.settings['exp'] for xp in xps ]))

    if names is None :
        names = [ xp.settings['exp'] for xp in xps ]

    # collect data
    eevo = {}
    xptimes = {}
    obs_type_var_occurs = {}
    for xp in xps :
        if len(times) == 0 :
            xptimes[xp] = xp.veri_times
        else :
            xptimes[xp] = times

        eevo[xp] = xp.compute_error_evolution( xptimes[xp], obs_types=obs_types, variables=variables, **filter_kw )

        for obs_type in list(eevo[xp].keys()) :
            if not obs_type in list(obs_type_var_occurs.keys()) :
                obs_type_var_occurs[obs_type] = {}
            for vname in list(eevo[xp][obs_type].keys()) :
                obs_type_var_occurs[obs_type][vname] = True


    plt.figure(44,figsize=(10,6))

    for obs_type in list(obs_type_var_occurs.keys()) :
        print('>>> plotting error evolution for observation type %s ' % obs_type, end=' ')

        for vname in list(obs_type_var_occurs[obs_type].keys()) :
            print(vname, end=' ')

            # rmse, spread & bias sawtooth plot
            plt.clf()
            plt.grid(zorder=-10)

            # determine reference time
            for ixp, xp in enumerate(xps) :
                if ixp == 0 :
                    ref_time = xptimes[xp][0]
                else :
                    if Time14(xptimes[xp][0]) < Time14(ref_time) :
                        ref_time = xptimes[xp][0]

            tmn = 0; tmn_abs = ref_time
            tmx = 0; tmx_abs = ref_time

            nolabel = False
            for ixp, xp in enumerate(xps) :
                if (obs_type in list(eevo[xp].keys())) and (vname in list(eevo[xp][obs_type].keys())) :

                    plot_sawtooth(       xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp], color=colors[ixp], nolabel=nolabel, quan='rmse' )
                    plot_sawtooth(       xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp], color=colors[ixp], nolabel=nolabel, quan='bias', style='--' )
                    trel, tabs = plot_sawtooth( xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp], color=colors[ixp], nolabel=nolabel, veri='ens', quan='spread', style=':' )
                    nolabel = True
                    if trel[0]  < tmn :
                        tmn     = trel[0]
                        tmn_abs = tabs[0]
                    if trel[-1] > tmx :
                        tmx     = trel[-1]
                        tmx_abs = tabs[-1]

                    if compare :
                        #plt.figtext( 0.01+ixp*0.1, 0.01, xp.settings['exp'], color=colors[ixp])
                        plt.figtext( 0.01+ixp*0.1, 0.01, names[ixp], color=colors[ixp], fontsize=8)

            #print '+++++* ', tmn, tmn_abs, tmx, tmx_abs
            xrng =  (tmn - 0.05*(tmx-tmn), tmx + 0.05*(tmx-tmn))
            plt.xlim( xrng )

            tickvalues, ticklabels = time_axis_ticks( tmn, tmx, tmn_abs, tmx_abs )
            ax = plt.gca()
            ax.set_xticks(tickvalues)
            ax.set_xticklabels(ticklabels)

            plt.plot( xrng, (0,0), '#999999', zorder=10, linewidth=1.5 )
            plt.legend(frameon=False,loc='upper right')
            plt.xlabel('hour [UTC] (start time %s)'% ref_time)
            #plt.xlabel('t - %s [h]' % ref_time )
            plt.ylabel('bias / spread / rmse')
            if compare :
                plt.title( obs_type+' '+vname)
            else :
                #plt.title(xps[0].settings['exp']+' '+obs_type+' '+vname)
                plt.title( names[0]+' '+obs_type+' '+vname)
            plt.savefig(output_path+'error_evo_'+obs_type+'__'+vname+'.'+file_type, bbox_inches='tight')


            # spread/skill relation plot
            plt.clf()
            plt.grid(zorder=-10)
            nolabel = False
            for ixp, xp in enumerate(xps) :
                if (obs_type in list(eevo[xp].keys())) and (vname in list(eevo[xp][obs_type].keys())) :
                    plot_sawtooth( xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp],
                                   color=colors[ixp], nolabel=nolabel, veri='ens', quan='spreadskill' )
                    plot_sawtooth( xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp],
                                   color=colors[ixp], nolabel=nolabel, veri='desroz', ana='', fg='',
                                   quan='spreadskill_mean', name='desroziers spread-skill', style='--' )

                    nolabel = True
                    if compare :
                        #plt.figtext( 0.01+ixp*0.1, 0.01, xp.settings['exp'], color=colors[ixp])
                        plt.figtext( 0.01+ixp*0.1, 0.01, names[ixp], color=colors[ixp])
            plt.plot( xrng, (1,1), '#999999', zorder=10, linewidth=1.5 )
            plt.xlim( xrng )
            ax = plt.gca()
            ax.set_xticks(tickvalues)
            ax.set_xticklabels(ticklabels)
            plt.ylim( (0,2) )
            plt.legend(frameon=False,loc='upper right')
            #plt.xlabel('hour [UTC]')
            plt.xlabel('hour [UTC] (start time %s)'% ref_time)
            plt.ylabel('spread/skill')
            if compare :
                plt.title( obs_type+' '+vname)
            else :
                #plt.title(xps[0].settings['exp']+' '+obs_type+' '+vname)
                plt.title( names[0]+' '+obs_type+' '+vname)
            plt.savefig(output_path+'spreadskill_evo_'+obs_type+'__'+vname+'.'+file_type, bbox_inches='tight')


            # Desroziers plot
            plt.clf()
            plt.grid(zorder=-10)
            nolabel = False
            for ixp, xp in enumerate(xps) :
                if (obs_type in list(eevo[xp].keys())) and (vname in list(eevo[xp][obs_type].keys())) :
                    
                    plot_sawtooth( xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp], color=colors[ixp], nolabel=nolabel,
                                   veri='desroz', ana='', fg='', quan='e_o_mean', name='mean desroziers estimate' )
                    plot_sawtooth( xp, eevo[xp][obs_type][vname], ref_time=ref_time, times=xptimes[xp], color=colors[ixp], nolabel=nolabel,
                                   veri='obs', ana='', fg='', quan='meanerr', style='--', name='mean assumed error' )
                    nolabel = True

                    if compare :
                        #plt.figtext( 0.01+ixp*0.1, 0.01, xp.settings['exp'], color=colors[ixp])
                        plt.figtext( 0.01+ixp*0.1, 0.01, names[ixp], color=colors[ixp])
            plt.xlim( xrng )
            ax = plt.gca()
            ax.set_xticks(tickvalues)
            ax.set_xticklabels(ticklabels)
            plt.gca().set_ylim(bottom=0)
            plt.legend(frameon=False,loc='upper right',fontsize=10)
            #plt.xlabel('hour [UTC]')
            plt.xlabel('hour [UTC] (start time %s)'% ref_time)
            plt.ylabel('obs. error')
            if compare :
                plt.title( obs_type+' '+vname)
            else :
                #plt.title(xps[0].settings['exp']+' '+obs_type+' '+vname)
                plt.title( names[0]+' '+obs_type+' '+vname)
            plt.savefig(output_path+'err_desroz_evo_'+obs_type+'__'+vname+'.'+file_type, bbox_inches='tight')

            print()

#--------------------------------------------------------------------------------
def time_axis_ticks( tmn, tmx, tmn_abs, tmx_abs ) :

    # determine tick interval
    duration = tmx-tmn
    if duration <= 24 :
        dt_maj = 3
    elif duration <= 48 :
        dt_maj = 6
    elif duration <= 96 :
        dt_maj = 12
    else :
        dt_maj = 24

    dt = Time14(dt_maj*3600)
    #print 'dt = ', dt, dt.is_delta()

    tfirst = Time14(tmn_abs).divisible( dt, ge=True )
    tlast  = Time14(tmx_abs).divisible( dt, ge=False )

    #print 'tfirst = ', tfirst
    #print 'tlast  = ', tlast

    t_abs = time_range( tfirst, tlast, dt )
    t_rel = []
    t_label = []
    tmn_es = Time14(tmn_abs).epoch_sec()
    tmx_es = Time14(tmx_abs).epoch_sec()
    for t in t_abs :
        #print '<<<', t
        t_rel.append( tmn + (tmx-tmn)*(Time14(t).epoch_sec()-tmn_es)/(tmx_es-tmn_es) )
        t_label.append( '%d' % Time14(t).dayhour() )

    return t_rel, t_label


#--------------------------------------------------------------------------------
def plot_sawtooth( xp, data, ref_time=None, times=[], quan='rmse', veri='mean', ana='ana', fg='fg',
                   time_unit='hour', color='k', style='-', linewidth=2, nolabel=False, name=None ) :

    if len(times) == 0 :
        times = xp.veri_times

    if ref_time is None :
        ref_time = times[0]

    es0 = Time14(ref_time).epoch_sec()
    hour0 = Time14(ref_time).dayhour()
    t_rel = []
    t_abs = []

    prev = False
    for time in times :
        if time in list(data.keys()) :
            if time_unit == 'minute' :
                t = (Time14(time).epoch_sec() - es0)//60
            elif time_unit == 'hour' :
                t = (Time14(time).epoch_sec() - es0)//3600
            elif time_unit == 'day' :
                t = (Time14(time).epoch_sec() - es0)//(3600*24)

            #    #t = Time14(time).dayhour()
            #    t = hour0 + (Time14(time).epoch_sec() - es0)/3600
            #print '--- time', time, t
            t_rel.append(t)
            t_abs.append(time)

            #print 'KEYS OF data ['+time+']['+ana+veri+'] ', data[time][ana+veri].keys()
            ana_rmse = data[time][ana+veri][quan]
            fg_rmse  = data[time][fg+veri][quan]
            if (prev == False) and not nolabel:
                if name is None :
                    label = quan
                else :
                    label = name
            else :
                label = None
            plt.plot( (t,t), (ana_rmse,fg_rmse), style, color=color, linewidth=linewidth, zorder=10, label=label )
            if prev == False :
                prev = True
            else :
                plt.plot( (old_t,t), (old_ana_rmse,fg_rmse), style, color=color, linewidth=linewidth, zorder=10 )

            plt.scatter( t, ana_rmse, s=40, facecolors=color, edgecolors=color, zorder=13 )
            plt.scatter( t, fg_rmse,  s=40, facecolors='w',   edgecolors=color, zorder=12 )

            old_t = t
            old_ana_rmse = ana_rmse

    return t_rel, t_abs

#--------------------------------------------------------------------------------
def plot_maps( xp, times=[], variables=[], obs_types=[], output_path='', file_type='png', **filter_kw ) :

    from kendapy.plot_ekf import plot_obsloc

    if len(times) == 0 :
        times = xp.veri_times

    for obs_type in obs_types :
        print(('>>> plotting maps for observation type %s...' % obs_type))

        for t in times :
            fname = xp.ekf_filename( t, obs_type )
            if not os.path.exists(fname) : continue
            ekf = xp.get_ekf( t, obs_type, **filter_kw )
            #ekf = kendapy.ekf.read_ekf( fname ) #, members=False )
            #kendapy.plot_ekf.plot_map( ekf, vars=variables, output_path=output_path, file_type=file_type )
            if xp.exp_type == 'sofia' :
                file_name = fname.split('/')[-1].replace('.nc','')+'_'+t
            else :
                file_name = fname.split('/')[-1].replace('.nc','')
            plot_obsloc( ekf, vars=[], proj='geos', output_path=output_path, file_name=file_name, file_type='png' )

#--------------------------------------------------------------------------------
def plot_cumulative_statistics( xp, times=None, variables=None, obs_types=None, output_path='', file_type='png',
                                fit=True, **filter_kw ) :

    if times     is None : times = []
    if variables is None : variables = []
    if obs_types is None : obs_types = []

    if len(times) == 0 :
        times = xp.veri_times

    cstat = xp.compute_cumulative_statistics( times, obs_types=obs_types, variables=variables, **filter_kw )

    for obs_type in list(cstat.keys()) :
        for vname in list(cstat[obs_type].keys()) :

            fig = plt.figure(figsize=(8,5))
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            binplot( cstat[obs_type][vname]['anaens_dep_hist_edges'], cstat[obs_type][vname]['anaens_dep_hist']+0.5, color='r',  semilogy=True, label='ANA' )
            binplot( cstat[obs_type][vname]['fgens_dep_hist_edges'],  cstat[obs_type][vname]['fgens_dep_hist']+0.5,  color='b' , semilogy=True, label='FG' )
            nmin = maximum( 1, cstat[obs_type][vname]['anaens_dep_hist'].max()/1e4 )
            if fit :
                anafit = gauss_fit( cstat[obs_type][vname]['anaens_dep_hist'], cstat[obs_type][vname]['anaens_dep_hist_edges'], nmin=-100 )
                plt.semilogy( cstat[obs_type][vname]['anaens_dep_hist_edges'], anafit+0.5, '--', linewidth=0.5, color='#990000' )
                fgfit = gauss_fit( cstat[obs_type][vname]['fgens_dep_hist'], cstat[obs_type][vname]['fgens_dep_hist_edges'], nmin=-100 )
                plt.semilogy( cstat[obs_type][vname]['fgens_dep_hist_edges'], fgfit+0.5, '--', linewidth=0.5, color='#000099' )

            plt.gca().set_ylim( bottom = nmin )
            plt.grid()
            plt.legend(frameon=False,loc='upper right',title=xp.settings['exp'])
            plt.title('normalised departures for %s/%s between %s and %s' % (obs_type, vname, times[0], times[-1]), fontsize=10)
            plt.xlabel('(model equivalent - observation) / observation error')
            figfname = output_path+'cumdepstat_'+obs_type+'_'+vname+'.'+file_type
            print(('SAVING ', figfname))
            plt.savefig( figfname, bbox_inches='tight' )

            plt.clf()
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            binplot( cstat[obs_type][vname]['obs_val_hist_edges'],    cstat[obs_type][vname]['obs_val_hist'],    color='k', label='OBS' )
            binplot( cstat[obs_type][vname]['anaens_val_hist_edges'], cstat[obs_type][vname]['anaens_val_hist'], color='r', label='ANA' )
            binplot( cstat[obs_type][vname]['fgens_val_hist_edges'],  cstat[obs_type][vname]['fgens_val_hist'],  color='b', label='FG'  )
            plt.grid()
            plt.legend(frameon=False,loc='upper right',title=xp.settings['exp'])
            plt.title('obs. & model equiv. for %s/%s between %s and %s' % (obs_type, vname, times[0], times[-1]), fontsize=10)
            figfname = output_path+'cumvalstat_'+obs_type+'_'+vname+'.'+file_type
            print(('SAVING ', figfname))
            plt.savefig( figfname, bbox_inches='tight' )

def gauss_fit( dep_hist, dep_bin_centers, nmin = 100 ) :
    # compute gaussian fit to departure histogram

    from scipy.optimize import curve_fit

    if nmin < 0 :
        nmin_ = maximum( dep_hist.max()/(-nmin), 10 )
    else :
        nmin_ = nmin

    imin = 0
    for i in range(dep_hist.size) :
        if dep_hist[i] > nmin_ :
            imin = i
            break
    imax = dep_hist.size-1
    for i in arange(dep_hist.size)[::-1] :
        if dep_hist[i] > nmin_ :
            imax = i
            break
    print(('considering %d < i < %d for fit...' % (imin,imax)))
    popt, pcov = curve_fit( gaussian, dep_bin_centers[imin:imax+1], dep_hist[imin:imax+1], p0=(dep_hist.max(),1.0) ) #, bounds=([dep_hist.max()/100,-10],[dep_hist.max()*100,10]) )
    dep_hist_fit = gaussian( dep_bin_centers, *popt)
    #idcs = where(dep_hist_fit > nmin_)
    print(('gaussian fit parameters : ', popt))
    return dep_hist_fit

def gaussian( x, a, b ) :
    return a*exp( -(x/b)**2 )

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate plots for KENDA experiment')

    parser.add_argument( '-E', '--evolution',   dest='evolution',   help='generate error evolution plots', action='store_true' )
    parser.add_argument( '-o', '--overview',    dest='overview',    help='generate error overview plot', action='store_true' )
    parser.add_argument( '-F', '--fofens-overview', dest='fofens_overview', help='generate error overview plot based on fof ensemble', action='store_true' )
    parser.add_argument(       '--fofevo',      dest='fofevo',      help='generate error evlution plots based on fof ensemble', action='store_true' )
    parser.add_argument( '-c', '--cumulative',  dest='cumulative',  help='plot cumulative statistics', action='store_true' )
    parser.add_argument(       '--maxnormdep',  dest='maxnormdep',  help='maximum normalised departure for departure histogram', type=float, default=5.0 )
    parser.add_argument(       '--ndepbins',    dest='ndepbins',    help='number of bins for departure statistics',              type=int,   default=100 )
    parser.add_argument(       '--valmax',      dest='valmax',      help='maximum value for value histogram',  default='auto' )
    parser.add_argument(       '--valmin',      dest='valmin',      help='minimum value for value histogram',  default='auto' )
    parser.add_argument(       '--nvalbins',    dest='nvalbins',    help='number of bins for value histogram', type=int, default=100 )
    parser.add_argument( '-m', '--maps',        dest='maps',        help='plot maps', action='store_true' )

    parser.add_argument( '-C', '--compare',     dest='compare',    help='generate comparison plots with data from all specified experiments', action='store_true' )
    parser.add_argument( '-l', '--lfcst',       dest='lfcst',      help='take also long forecasts (not only cycling results) into account',   action='store_true' )
    parser.add_argument(       '--recompute',   dest='recompute',  help='do not read cache files', action='store_true' )

    parser.add_argument(       '--colors',      dest='colors',     help='comma-separated list of colors for the experiments', default='' )
    parser.add_argument(       '--names',       dest='names',      help='comma-separated list of names (use ,, for commas within the names)', default='' )

    parser.add_argument( '-A', '--area-filter',  dest='area_filter',  help='area filter for observations', default='auto' )
    parser.add_argument( '-S', '--state-filter', dest='state_filter', help='state filter for observations', default='active' )
    parser.add_argument( '-V', '--variables',    dest='variables',    help='comma-separated list of variables  to be considered (default: all)', default='' )
    parser.add_argument( '-O', '--obs-types',    dest='obs_types',    help='comma-separated list of obs. types to be considered (default: all)', default='' )
    parser.add_argument( '-s', '--start-time',   dest='start_time',   help='start time',    default='' )
    parser.add_argument( '-e', '--end-time',     dest='end_time',     help='end time',      default='' )
    parser.add_argument( '-d', '--delta-time',   dest='delta_time',   help='time interval', default='' )

    parser.add_argument(       '--dwd2lmu',     dest='dwd2lmu',     help='convert settings',  action='store_true' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument( '-I', '--input-path',  dest='input_path',  help='path to the directory containing the log files', default='' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()


    # set some default values

    obs_types = args.obs_types.split(',')
    if len(obs_types) == 0 or (len(obs_types) == 1 and obs_types[0]==''):
        obs_types = list(kendapy.ekf.tables['obstypes'].values())
        if args.verbose : print(('setting default value for obs_types : ', obs_types))

    variables = args.variables.split(',')
    if len(variables) == 0 or (len(variables) == 1 and variables[0]==''):
        variables = list(kendapy.ekf.tables['varnames'].values())
        if args.verbose : print(('setting default value for variables : ', variables))

    if args.output_path != '' :
        output_path = args.output_path+'/'
    else :
        output_path = ''


    # process all log files

    logfiles = args.logfile
    for i, lfn in enumerate(logfiles) :
        if not lfn.endswith('.log') and args.input_path != '' : # asssume it is an experiment id and not a log file
            logfiles[i] += '/run_cycle_'+logfiles[i]+'.log'
        if args.input_path != '' : # add input path
            logfiles[i] = os.path.join( args.input_path, logfiles[i] )
    if args.names != '' :
        xpnames = args.names.split(',')
    else :
        xpnames = [ logfile.split('/')[-1].replace('run_cycle_','').replace('.log','.') for logfile in logfiles ]

    if args.compare and (len(logfiles) > 1) : # generate comparison plots experiments

        xps = []
        for i, logfile in enumerate(logfiles) :
            xp = Experiment(logfile)
            print(('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
                   xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] )))
            xps.append(xp)
            if args.names != '' :
                xpnames[i] = xp.description(xpnames[i])

        print()
        print('popular names and real names :')
        for i,xp in enumerate(xps) :
            print(("#%d : %s : %s = %s" % (i, logfiles[i], xp.expid, xpnames[i])))
        print()

        if (args.start_time != '') and (args.end_time != '') and (args.delta_time != '') :
            times = time_range( args.start_time, args.end_time, args.delta_time  )
        else :
            times = []

        if args.evolution :
            plot_error_evolution( xps,
                                  names = xpnames,
                                  colors=args.colors.split(',') if args.colors != '' else None,
                                  times=times, variables=variables, obs_types=obs_types,
                                  area_filter=args.area_filter, state_filter=args.state_filter,
                                  output_path=output_path, file_type=args.file_type )

        if args.overview :
            plot_error_overview( xps,
                                 names = xpnames,
                                 colors=args.colors.split(',') if args.colors != '' else None,
                                 times=times, variables=variables, obs_types=obs_types,
                                 area_filter=args.area_filter,  state_filter=args.state_filter,
                                 output_path=output_path, file_type=args.file_type )

        if args.fofens_overview :
            for time_range in ('1,60','61,120','121,180') : #('151,180','91,120','31,60') :
                print(('generating fofens comparison for time range t%smin...' % time_range))
                plot_fofens_overview( xps, time_range=time_range, names = xpnames,
                                      colors=args.colors.split(',') if args.colors != '' else None,
                                      times=times, variables=variables, obs_types=obs_types,
                                      area_filter=args.area_filter,  state_filter=args.state_filter,
                                      output_path=output_path, file_type=args.file_type, recompute=args.recompute )

    else : # generate individual plots for each experiment

        for logfile in logfiles :

            print(("processing file %s ..." % logfile))

            xp = Experiment(logfile, dwd2lmu=args.dwd2lmu)
            print(('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
                   xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] )))

            # determine time range

            if args.start_time != '' :
                start_time = args.start_time
            else :
                start_time = xp.veri_times[0]

            if args.end_time != '' :
                end_time = args.end_time
            else :
                end_time =  xp.veri_times[-1]

            if args.delta_time != '' :
                delta_time = args.delta_time
            else :
                delta_time = xp.settings['ASSINT']
            times = time_range( start_time, end_time, delta_time  )

            print(('times: ', times))

            if args.maps :
                plot_maps( xp, times=times, variables=variables, obs_types=obs_types,
                           area_filter=args.area_filter, state_filter=args.state_filter,
                           output_path=output_path, file_type=args.file_type )

            if args.evolution :
                plot_error_evolution( xp, times=times, variables=variables, obs_types=obs_types,
                                      area_filter=args.area_filter, state_filter=args.state_filter,
                                      output_path=output_path, file_type=args.file_type, lfcst=args.lfcst )

            if args.fofevo :
                plot_fof_error_evolution( xp, times=times, variables=variables, obs_types=obs_types,
                                      area_filter=args.area_filter, state_filter=args.state_filter,
                                      output_path=output_path, file_type=args.file_type, lfcst=args.lfcst )

            if args.overview :
                plot_error_overview( xp, times=times, variables=variables, obs_types=obs_types,
                                     area_filter=args.area_filter, state_filter=args.state_filter,
                                     output_path=output_path, file_type=args.file_type )

            if args.cumulative :
                valmin = None if args.valmin == 'auto' else float(args.valmin)
                valmax = None if args.valmax == 'auto' else float(args.valmax)
                plot_cumulative_statistics( xp, times=times, variables=variables, obs_types=obs_types,
                                            area_filter=args.area_filter,  state_filter=args.state_filter,
                                            output_path=output_path, file_type=args.file_type,
                                            depmin=-args.maxnormdep, depmax=args.maxnormdep, ndepbins=args.ndepbins,
                                            valmin=valmin, valmax=valmax, nvalbins=args.nvalbins )
