#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ R A D I A T I O N
#
#  2020.6 L.Scheck 

from __future__ import absolute_import, division, print_function

import os, sys, subprocess, glob, tempfile, re, argparse, pickle, hashlib, netCDF4
from datetime import datetime, timedelta
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from kendapy.ekf        import Ekf
from kendapy.cosmo_state import CosmoState
from kendapy.bacy_utils import t2str, str2t, add_branch, common_subset, default_color_sequence, expand_color_name, to_timedelta
from kendapy.bacy_exp   import BacyExp


#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="Generate basic radiation verification plots")

    # plot type and parameters

    parser.add_argument(       '--lead-time-plots',  dest='lead_time_plots',  action='store_true', help='generate plots for each lead time and fcst time' )
    parser.add_argument(       '--fcst-plots',       dest='fcst_plots',       action='store_true', help='generate plots for each forecast' )

    parser.add_argument( '-C', '--compare',          dest='compare',          action='store_true', help='generate comparison plot instead of individual plots for all experiments' )
    parser.add_argument(       '--plot-mean-evo',    dest='plot_mean_evo',    action='store_true', help='in comparison plots, show also mean values, nit only rmse and bias' )

    parser.add_argument( '-F', '--filter',           dest='filter',           default='state=all', help='filter string (default: "state=active")' )

    parser.add_argument(       '--time-range',       dest='time_range',       default=None,        help='time range, e.g. 20190629170000,20190630120000' )
    parser.add_argument(       '--start-hour',       dest='start_hour',       default=None,        help='use only forecasts started at this hour' )
    
    parser.add_argument(       '--names',            dest='names',            default=None,        help='experiment names (e.g. EXP1,EXP2)' )

    # output options
    parser.add_argument( '-o', '--output-path', dest='output_path',        default=None,              help='output path' )
    parser.add_argument( '-i', '--image-type',  dest='image_type',         default='png',             help='[ png | eps | pdf ... ]' )
    parser.add_argument(       '--dpi',         dest='dpi',                default=100, type=int,     help='dots per inch for pixel graphics (default: 100)' )
    parser.add_argument(       '--figsize',     dest='figsize',            default='5,4',             help='<figure width>,<figure height> [inch]' )
    parser.add_argument(       '--colors',      dest='colors',             default=None,              help='comma-separated list of colors (e.g. "r,#ff0000,pink")' )

    parser.add_argument( '-v', '--verbose',     dest='verbose', action='store_true',  help='be extremely verbose' )

    parser.add_argument( 'experiments', metavar='<experiment path(s)>', help='path(s) to experiment(s)', nargs='*' )

    return parser

#-------------------------------------------------------------------------------
def get_radiation_fcst( xp, fcst, filter, prefix=None ) :

    # determine cache file name
    hash_object = hashlib.md5(filter.encode()) # create has value from filter string
    fn_cache = '{}/bacy_radiation_cache_{}_{}_{}.pickle'.format( fcst, xp.exp_dir, fcst, hash_object.hexdigest() )
    if not prefix is None :
        fn_cache = os.path.join( prefix, fn_cache )
        cache_dir = os.path.join( prefix, fcst )
    else :
        cache_dir = fcst
    print('  cache file name generated using filter "{}" is {}'.format(filter,fn_cache))

    # create output directory for current forecast
    if not os.path.exists(cache_dir) :
        os.makedirs(cache_dir)

    # load cached results, if available
    if os.path.exists(fn_cache) :
        print('  reading cache file {} ...'.format(fn_cache) )
        with open( fn_cache, 'rb' ) as f :
            res = pickle.load( f )

    else : # compute results for forecast started at fcst

        print('  could not find cache file {} -> computing results for forecast started at {}...'.format(fn_cache,fcst) )

        res = {} ### bad idea (removed): include { 'filter':filter }

        # load grid (for deterministic run, may differ from grid for ensemble runs)
        grid = xp.get_grid(mem='det')
        clon = grid.variables['clon'][...]
        clat = grid.variables['clat'][...]

        # load lead time 0 file
        fn_mdl_0 = xp.get_filename( 'fc', time_start=fcst, lead_time=0, mem='det')
        mdl_prev = CosmoState(fn_mdl_0, preload=['ASWDIFD_S','ASWDIR_S'])
        tv_0 = str2t(fcst)
        tv_prev = tv_0

        # loop over lead times
        for ldt in xp.fc_lead_times[1:] : # .......................................................................
            print('    processing lead time ', ldt)

            # load ekf file
            tv = str2t(fcst) + to_timedelta(ldt, units='h') # valid time = fcst start time + lead time
            fn_ekf = xp.get_filename( 'ekf', obs_type='SYNOP', time_valid=tv)
            if not os.path.exists(fn_ekf) :
                print('    {} does not exist --> not processing rest of forecast.'.format(fn_ekf))
                break
            print('    opening '+fn_ekf)
            ekf = Ekf( fn_ekf, filter=filter )
            print('    found {} RAD_GL and {} RAD_DF observations...'.format(
                ekf.n_obs(filter='varname=RAD_GL'), ekf.n_obs(filter='varname=RAD_DF')))

            # identify cases where we have global and diffuse radiation
            #for varname in ['RAD_GL','RAD_DF'] :
            #    for iobs in range(ekf.n_obs(filter='varname='+varname)) :
            #        oid = "%d,%09.5f,%09.5f,%f" % ( self.data['time'][j]+time_shift, self.data['lat'][j], self.data['lon'][j], self.data['level'][i] )

            # load model output
            fn_mdl = xp.get_filename( 'fc', time_start=fcst, lead_time=ldt, mem='det')
            print('    opening '+fn_mdl)
            mdl = CosmoState(fn_mdl, preload=['ASWDIFD_S','ASWDIR_S'])

            # initialize output dictionary
            res[ldt] = {}

            # compute model equivalents
            t_now  = (tv-tv_0).total_seconds()
            t_prev = (tv_prev-tv_0).total_seconds()
            meq = {}
            for varname in ['RAD_GL','RAD_DF'] : # ................................................................

                obs = ekf.obs(filter='varname='+varname) / 3600.0 # FIXME generalize!
                lat = ekf.obs(filter='varname='+varname,param='lat')
                lon = ekf.obs(filter='varname='+varname,param='lon')
                meq[varname] = obs*0
                idx = np.zeros( obs.size, dtype=int )

                # loop over observations
                for i in range(lat.size) :

                    # find grid cell index best matching observation location
                    icell = np.argmin( np.abs( grid.variables['clon'][...] - lon[i]*np.pi/180 )
                                    + np.abs( grid.variables['clat'][...] - lat[i]*np.pi/180 ) )
                    idx[i] = icell

                    # compute (deaccumulate) model radiation quantities
                    rad_df  =  (t_now * mdl['ASWDIFD_S'][icell] - t_prev * mdl_prev['ASWDIFD_S'][icell]) / ( t_now - t_prev)
                    if varname == 'RAD_GL' :
                        rad_dir =  (t_now * mdl['ASWDIR_S'][icell]  - t_prev * mdl_prev['ASWDIR_S'][icell] ) / ( t_now - t_prev)
                        meq[varname][i] = rad_df + rad_dir
                    elif varname == 'RAD_DF' :
                        meq[varname][i] = rad_df

                    # location match check
                    if (np.abs(lat[i]-grid.variables['clat'][icell]*180/np.pi) > 0.1) or (lon[i]-grid.variables['clon'][icell]*180/np.pi > 0.1) :
                        print('WARNING: Could not find matching grid cell')
                        #print( i, obs[i], meq[varname][i], lat[i]-grid.variables['clat'][icell]*180/np.pi, lon[i]-grid.variables['clon'][icell]*180/np.pi )
                        print( i, obs[i], rad_df+rad_dir, rad_dir, rad_df, lat[i], grid.variables['clat'][icell]*180/np.pi, lon[i], grid.variables['clon'][icell]*180/np.pi )

                res[ldt][varname] = { 'obs':obs, 'meq':meq[varname], 'lat':lat, 'lon':lon, 'idx':idx }

                print( '    -- ', varname, np.sqrt(( (obs-meq[varname])**2 ).mean()), (obs-meq[varname]).mean() )

                # end of varname loop .................................................................................

            # update mdl_prev
            mdl_prev = mdl
            tv_prev  = tv

            # end of lead time loop ...................................................................................

        # save results to cache file
        print('  saving results for forecast started at {} to {} ...'.format( fcst, fn_cache ))
        with open( fn_cache, 'wb' ) as f :                    
            pickle.dump( res, f, pickle.HIGHEST_PROTOCOL )

    return res


#-------------------------------------------------------------------------------
def compare_radiation_fcst( xps, vargs ) :

    print('COMPARING EXPERIMENTS ', [xp.exp_dir for xp in xps], '...' )

    # define experiment names
    if vargs['names'] is None :
        exp_names = [ xp.exp_dir for xp in xps ]
    else :
        exp_names = vargs['names'].split(',')

    # create output directory, if necessary
    if not args.output_path is None :
        if not os.path.exists( args.output_path ) :
            os.makedirs( args.output_path )

    # determine forecast start times
    print('fcst start times of each experiment:')
    for xp in xps :
        print( '  - ', xp.exp_dir, xp.fc_start_times)

    fc_start_times = common_subset([ xp.fc_start_times for xp in xps ])
    print('all      fcst start times : ', fc_start_times)

    if not vargs['time_range'] is None :
        t0, t1 = vargs['time_range'].split(',')
        fc_start_times = [ t for t in fc_start_times if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]
    if not vargs['start_hour'] is None :
        fc_start_times = [ t for t in fc_start_times if str2t(t).hour == int(vargs['start_hour']) ]
    print('selected fcst start times : ', fc_start_times)

    metrics = {}

    # loop over forecast start times ..............................................................................
    for fcst in fc_start_times :
        print()
        print('  processing forecast started at ', fcst)

        res = [ get_radiation_fcst( xp, fcst, vargs['filter'], prefix=xp.exp_dir ) for xp in xps ]

        # determine lead times
        lead_times = sorted(list(common_subset([ r.keys() for r in res ])))[:-1]
        # this did not work any more, because there was an element 'filter' in res.keys() (now removed)        
        # lead_times = sorted(list(common_subset( [ [ rr for rr in r.keys() if rr != 'filter' ] for r in res ] )))[:-1]
        print('  common lead times : ', lead_times )

        for varname in [ 'RAD_GL', 'RAD_DF' ] :

            for ldt in lead_times :

                # generate obs IDs
                obsids = [ generate_obsids(r[ldt][varname]) for r in res ]

                # find common subset of IDs
                common_obsids = obsids[0].keys()
                for i in range(1,len(xps)) :
                    common_obsids = common_subset(( common_obsids, obsids[i].keys() ))
                print('    lead time {} : Found {} common observations...'.format(ldt,len(common_obsids)), [ r[ldt][varname]['obs'].size for r in res] )

                # compute metrics for common observations

                for ixp, xp in enumerate(xps) :

                    # convert IDs to indices
                    obsidcs = [ obsids[ixp][oid] for oid in common_obsids ]

                    r = res[ixp][ldt][varname]

                    r['meqmean'] =          ( r['meq'][obsidcs]                        ).mean()
                    r['obsmean'] =          (                     r['obs'][obsidcs]    ).mean()
                    r['bias']    =          ( r['meq'][obsidcs] - r['obs'][obsidcs]    ).mean()
                    r['rmse']    = np.sqrt( ((r['meq'][obsidcs] - r['obs'][obsidcs])**2).mean() )
                    r['n_common'] = len(obsidcs)

        metrics[fcst] = res

    # generate plots ..................................................................................................

    # RMSE & BIAS time evolution
    for varname in [ 'RAD_GL', 'RAD_DF' ] :
        fig, ax = plt.subplots(figsize=(10,5))
        xticks = []
        xticklabels = []
        for fcst in fc_start_times :
            for ixp, xp in enumerate(xps) :
                lead_times = sorted(metrics[fcst][ixp].keys())[:-1]
                #print('LDTs', lead_times)

                obsmean = [ metrics[fcst][ixp][ldt][varname]['obsmean'] for ldt in lead_times ]
                meqmean = [ metrics[fcst][ixp][ldt][varname]['meqmean'] for ldt in lead_times ]
                rmse    = [ metrics[fcst][ixp][ldt][varname]['rmse']    for ldt in lead_times ]
                bias    = [ metrics[fcst][ixp][ldt][varname]['bias']    for ldt in lead_times ]

                t0 = str2t(fc_start_times[0])
                times = [ ((str2t(fcst) + to_timedelta(ldt,units='h')) - t0).total_seconds()/3600.0 for ldt in lead_times ]                

                if vargs['plot_mean_evo'] :

                    ax.plot( times, obsmean, ':', color='k',
                             label='obs. mean' if (fcst == fc_start_times[0] and ixp == 0) else None )
                    ax.plot( times, meqmean, ':', color=default_color_sequence()[ixp],
                             label='model mean '+exp_names[ixp] if fcst == fc_start_times[0] else None )

                ax.plot( times, rmse, color=default_color_sequence()[ixp],
                         label='rmse '+exp_names[ixp] if fcst == fc_start_times[0] else None )

                ax.plot( times, bias, '--', color=default_color_sequence()[ixp],
                         label='bias '+exp_names[ixp] if fcst == fc_start_times[0] else None )

            
            start_time = str2t(fcst)
            start_hour = (start_time - t0).total_seconds()/3600.0
            xticks.append(start_hour)
            xticklabels.append(fcst[-8:-6]+'/'+fcst[-6:-4])
            ax.axvline( x=start_hour, color='k', linestyle='--', alpha=0.3, zorder=0)
        
        ax.set_xticks(xticks) #,rotation=90)
        ax.set_xticklabels(xticklabels)
        ax.axhline(color='r',zorder=0, alpha=0.5,linewidth=0.5)
        ax.set_ylabel(r'{} $[W/m^2]$'.format(varname))
        if not vargs['start_hour'] is None :
            hstart = '__start{}UTC'.format(vargs['start_hour'])
        else :
            hstart = ''
        xps_str = '_'.join([xp.exp_dir for xp in xps])
        meanstr = '_incl_mean' if vargs['plot_mean_evo'] else ''
        fn_plot = 'comparison_{}{}__{}{}.{}'.format(xps_str,hstart,varname,meanstr,vargs['image_type'])
        if not vargs['output_path'] is None :
            fn_plot = os.path.join( vargs['output_path'], fn_plot )
        print('    saving {} ...'.format(fn_plot))
        ax.legend(frameon=False)
        fig.savefig( fn_plot, bbox_inches='tight' )
        plt.close(fig)


#-------------------------------------------------------------------------------
def generate_obsids( res ) :
    """Generate unique observation IDs"""
    #print('generate_obsids : got ', res.keys() )

    obsids = {}
    for i in range(res['obs'].size) :
        obsid = "%09.5f,%09.5f" % ( res['lat'][i], res['lon'][i] )
        if obsid in obsids :
            print('WARNING: NON-UNIQUE OBS ID!')
        else :
            obsids[obsid] = i

    return obsids

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # parse command line arguments
    parser = define_parser()
    args = parser.parse_args()
    
    # convert argparse object to dictionary
    vargs = vars(args)

    # remember current working directory
    cwd = os.getcwd()

    if args.compare : # compare several experiments ===================================================================
        
        compare_radiation_fcst( [ BacyExp(p) for p in args.experiments ], vargs )

    else : # generate individual plots for each experiment ============================================================

        # loop over experiments
        for xp_path in args.experiments : # ...............................................................................
            print()
            print('processing experiment {}...'.format(xp_path))

            xp = BacyExp( xp_path )

            # by default, generate plots in subdir named like the experiment dir
            if args.output_path is None :
                if not os.path.exists(xp.exp_dir) :
                    os.mkdir(xp.exp_dir)
                os.chdir( xp.exp_dir)
            else :
                os.chdir( args.output_path )

            # determine start times
            print('available fcst start times : ', xp.fc_start_times)
            fc_start_times = xp.fc_start_times
            if not vargs['time_range'] is None :
                t0, t1 = vargs['time_range'].split(',')
                fc_start_times = [ t for t in fc_start_times if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]
                print('selected fcst start times : ', fc_start_times)

            # loop over forecast start times ..............................................................................
            for fcst in fc_start_times :
                print()
                print('  processing forecast started at ', fcst)

                res = get_radiation_fcst( xp, fcst, args.filter )

                # generate some plots
                for varname in [ 'RAD_GL', 'RAD_DF' ] :

                    if args.fcst_plots :
                        # plot obs/meq distribution as function of lead time
                        fig_fcst, ax_fcst = plt.subplots(figsize=(10,5))

                        for ldt in xp.fc_lead_times[1:] :
                            
                            if not ldt in res :
                                break

                            obs, meq = [ res[ldt][varname][q] for q in ['obs','meq'] ]

                            ax_fcst.scatter( [ldt+0.1]*obs.size, obs, marker='.', color='k', alpha=0.2)
                            ax_fcst.scatter( [ldt-0.1]*obs.size, meq, marker='.', color='r', alpha=0.2)

                            ax_fcst.plot( [ldt-0.4,ldt+0.4], [obs.mean()]*2, color='k' )
                            ax_fcst.plot( [ldt-0.4,ldt+0.4], [meq.mean()]*2, color='r' )

                            ax_fcst.plot( [ldt-0.4,ldt+0.4], [ np.sqrt(( (obs-meq)**2 ).mean()) ]*2, color='b' )
                            #ax_fcst.plot( [ldt-0.4,ldt+0.4], [ np.std(meq) ]*2, color='g' )

                        ax_fcst.set_title(xp.exp_dir + ' fcst started at ' + fcst)
                        ax_fcst.set_xlabel('lead time [h]')
                        ax_fcst.set_ylabel(varname)
                        ax_fcst.grid()
                        ax_fcst.set_xlim((0.5,xp.fc_lead_times[-1]+0.5))
                        fn_plot = '{}/obs_vs_mdl_{}_{}.{}'.format(fcst,varname,fcst,args.image_type)
                        print('  saving {} ...'.format(fn_plot))
                        fig_fcst.savefig( fn_plot, bbox_inches='tight')
                        plt.close(fig_fcst)

                    if args.lead_time_plots :
                        # generate scatter plot for each lead time
                        vmax = 1200
                        for ldt in xp.fc_lead_times[1:] :

                            obs, meq = [ res[ldt][varname][q] for q in ['obs','meq'] ]
                            
                            fig, ax = plt.subplots(figsize=(4,4))
                            ax.plot( (0,vmax), (0,vmax), 'r', alpha=0.5, linewidth=0.5 )
                            ax.scatter( obs, meq, marker='.', alpha=0.3 )
                            ax.set_title(xp.exp_dir)
                            ax.set_xlabel('observed '+varname)
                            ax.set_ylabel('forecast '+varname)
                            ax.text( 10, 10, 'RMSE={:.1f}, BIAS={:.1f}, #obs={}'.format( np.sqrt(( (obs-meq)**2 ).mean()),
                                                                                    (meq-obs).mean(), obs.size ) )
                            ax.text( vmax-10, vmax-10, '{}+{:02d}h'.format(fcst,ldt), ha='right', va='top' )                    
                            ax.set_xlim((0,vmax))
                            ax.set_ylim((0,vmax))
                            ax.grid()
                            fn_plot = '{}/obs_vs_mdl_{}_{}_{:02d}hr.{}'.format(fcst,varname,fcst,ldt,args.image_type)
                            print('  saving {} ...'.format(fn_plot))    
                            fig.savefig( fn_plot, bbox_inches='tight')
                            plt.close(fig)

                # end of fcst start time loop .............................................................................

            os.chdir(cwd)

            # end of experiment loop ......................................................................................
