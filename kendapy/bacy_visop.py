#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ V I S O P
#
#  2020.6 L.Scheck 

from __future__ import absolute_import, division, print_function

import os, sys, subprocess, glob, tempfile, re, argparse
from datetime import datetime, timedelta
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from kendapy.score_fss import fss_dict
from kendapy.bacy_exp   import BacyExp
from kendapy.bacy_utils import common_subset, str2t, t2str, to_datetime, to_timedelta, midnight, default_color_sequence

#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="Generate visible satellite image plots for bacy experiments")

    parser.add_argument(       '--visop',       dest='visop',      action='store_true',  help='use visop-generated images' )
    parser.add_argument(       '--datool',      dest='datool',     action='store_true',  help='use datool-generated images' )
    parser.add_argument(       '--histograms',  dest='histograms', action='store_true',  help='generate histograms' )
    parser.add_argument(       '--calibration', dest='calibration', default=None, type=float, help='divide observations by this calibration factor' )

    # plot type and parameters
    parser.add_argument( '-f', '--plot-files',       dest='plot_files',       action='store_true',  help='plot reflectance maps for the specified files' )
    parser.add_argument( '-e', '--plot-experiments', dest='plot_experiments', action='store_true',  help='plot reflectance maps for the specified experiments' )
    parser.add_argument( '-C', '--compare',          dest='compare',          action='store_true',  help='generate comparison plot instead of individual plots for all experiments' )
    parser.add_argument( '-d', '--diff',             dest='diff',             action='store_true',  help='plot also B-O differences' )
    parser.add_argument( '-F', '--forecasts',        dest='forecasts',        action='store_true',  help='plot forecasts' )
    parser.add_argument( '-b', '--belscl',           dest='belscl',           action='store_true',  help='plot believable scale' )

    ### VISOP vs. DATOOL comparison
    parser.add_argument(       '--visop-vs-datool',  dest='visop_vs_datool',  action='store_true',  help='compare visop and datool generated det fg images' )
    parser.add_argument(       '--cmp-visop-main',   dest='cmp_visop_main',   action='store_true',  help='compare visop images generated for cycle and main forecasts' )
    
    
    parser.add_argument(       '--cmp-visop-types',  dest='cmp_visop_types',  action='store_true',  help='compare cycle results for different visop types' )
    parser.add_argument(       '--skip-datool',      dest='skip_datool',      action='store_true',  help='do not include datool results in comparison of visop types' )
    parser.add_argument(       '--datool-coords',    dest='datool_coords',    default=None, help='file containing lat-lon coordinates of the observations in the seviri_* files' )

    parser.add_argument(       '--visop-type',       dest='visop_type',       default=None,         help='visop settings type, e.g. "uncalib" (default: "std")' )
    parser.add_argument(       '--visop-path',       dest='visop_path',       default=None,         help='if specified, read visop output from this path and not from visop inside experiment directory' )
    parser.add_argument(       '--refl-thres',       dest='refl_thres',       default=0.0, type=float, help='threshold reflectance for histogram error' )
    parser.add_argument(       '--plot-each-fcst',   dest='plot_each_fcst',   action='store_true',  help='generate plots for each fcst, not just summary plots' )
    
    parser.add_argument(       '--latlon',           dest='latlon',           default='45,0,56,17', help='latmin,lonmin,latmax,lonmax' )

    # forcast start times can be limited to this window
    parser.add_argument( '-S', '--start-daily',  dest='start_daily',   default='6:00',         help='daily start hour' )
    parser.add_argument( '-E', '--end-daily',    dest='end_daily',     default='16:00',        help='daily end hour' )

    parser.add_argument(       '--time-range', dest='time_range',     default=None,              help='time range, e.g. 20190629170000,20190630120000' )

    parser.add_argument( '-o', '--output-path', dest='output_path',        default=None,              help='output path' )
    parser.add_argument( '-i', '--image-type',  dest='image_type',         default='png',             help='[ png | eps | pdf ... ]' )

    parser.add_argument( 'experiments', metavar='<experiment path(s)>', help='path(s) to experiment(s) / files', nargs='+' )

    return parser


#----------------------------------------------------------------------------------------------------------------------
def load_seviri_file( fn, lat=None, lon=None, transpose=False, datool_coords=None, verbose=False ) :
    """Load reflectance field (from the specified file name)
       and lat-lon coordinates, unless they are provided as arguments.
       returns reflectance, latitude, longitude as 2d numpy arrays"""

    from kendapy.cosmo_state import CosmoState

    # load coordinates, if they have not been provided in lat and lon
    if lat is None or lon is None :

        if not datool_coords is None :
            cs2 = CosmoState(datool_coords)
        else :
            # open file
            cs2 = None
            local_lonlat = ('/'.join(fn.split('/')[:-1])) + '/seviri_lonlat.grb2'
            if os.path.exists(local_lonlat) :
                print('loading ', local_lonlat)
                cs2 = CosmoState(local_lonlat)
            else : # FIXME preliminary solution...  

                tkns = fn.split('/')
                ndir = len(tkns)-1
                for n in range(ndir, 1, -1) :
                    fnll = '/'.join( tkns[:n ]) + '/seviri_latlon.grb'                
                    if os.path.exists(fnll) :
                        print('found coordinates file: ', fnll)    
                        cs2 = CosmoState(fnll)                    
                        print('coordinates shape / size', cs2['CLAT'].shape, cs2['CLAT'].size)

                if cs2 is None : # still not found...
                    if 'VISOP_ENV' in os.environ :
                        envpath = os.environ['VISOP_ENV']
                    else :
                        envpath = '/lustre2/uwork/lscheck/visop_env'
                    print( local_lonlat, ' does not exist --> loading file from visop environment ', envpath)
                    if ('_402' in fn) or ('_802' in fn) :
                        print(envpath+'/seviri_lonlat_X02.grb2')
                        cs2 = CosmoState(envpath+'/seviri_lonlat_X02.grb2') # 402, 802 experiments  
                    else :
                        print(envpath+'/seviri_lonlat.grb2')
                        cs2 = CosmoState(envpath+'/seviri_lonlat.grb2') # 403, 803

        if verbose :
            print('latitude has size ', cs2['CLAT'].shape )

        # determine 2d shape
        lat1d, lon1d = cs2['CLAT'], cs2['CLON']
        for i in range(lat1d.size) :
            if i>0 and lat1d[i] < lat1d[i-1] :
                nlat = i
                break
        nlon = lat1d.size // nlat
        if nlon * nlat != lat1d.size :
            #print('detecting nlon and nlat failed... (1st try)', nlon, nlat, nlon*nlat, lat1d.size)
            nlat += 1
            nlon = lat1d.size // nlat
            if nlon * nlat != lat1d.size :
                print('detecting nlon and nlat failed... (2nd try)', nlon, nlat, nlon*nlat, lat1d.size)
                nlat += 1
                nlon = lat1d.size // nlat
                if nlon * nlat != lat1d.size :
                    print('detecting nlon and nlat failed... (3nd try)', nlon, nlat, nlon*nlat, lat1d.size)

        if verbose :
            print( 'load_seviri_file : lat-lon grid : nlat={}, nlon={}, nlat*nlon={} (should be {})'.format( nlat, nlon, nlat*nlon, lat1d.size ) )

        # create 2d fields from 1d vectors
        lat = lat1d.reshape((nlon,nlat))
        lon = lon1d.reshape((nlon,nlat))

        if False :
            fig, ax = plt.subplots( 2 ) # lon increases more slowly with index
            ax[0].plot( lat1d )
            ax[1].plot( lon1d )
            fig.savefig('latlon1d.png')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(20,20))
            ax.scatter( lon1d, lat1d, s=1 )
            ax.grid()
            fig.savefig('latlon_scatter.pdf')    
            plt.close(fig)

            fig, ax = plt.subplots( 2 )
            ax[0].imshow( lat )
            ax[1].imshow( lon )
            fig.savefig('latlon2d.png')
            plt.close(fig)
    else :
        nlon, nlat = lat.shape

    # load reflectance field
    cs = CosmoState(fn)
    refl = cs['RFL06'].reshape((nlon,nlat))

    if transpose :
        return np.transpose(refl), np.transpose(lat), np.transpose(lon)
    else :
        return refl, lat, lon


#----------------------------------------------------------------------------------------------------------------------
def refl_diff_colormap() :
    from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list( 'refl_diff', 
    #                                          [(0.00, '#0000ff'),
    #                                           (0.40, '#00ffff'),
    #                                           (0.49, 'w'),
    #                                           (0.51, 'w'),
    #                                           (0.60, '#ffff00'),
    #                                           (1,    '#ff0000')], N=1024)

    cmap = LinearSegmentedColormap.from_list( 'refl_diff', 
                                             [(0.00, '#004275'),
                                              (0.30, '#007EE5'),
                                              (0.40, '#3FC5FE'),
                                              (0.47, '#A3E4FF'),
                                              (0.495, 'w'),
                                              (0.505, 'w'),
                                              (0.53, '#FECB7E'),
                                              (0.60, '#F59301'),
                                              (0.70, '#D52E00'),
                                              (1,    '#7F1B00')], N=1024)
    return cmap


#----------------------------------------------------------------------------------------------------------------------
def plot_reflectance( r, name=None, save_to=None, vmin=0.0, vmax=1.0, infos=True, infostring=None, notext=False, grid=False, grid_alpha=0.4,
                      contour=None, cmap=None, spag=None, cfield=None, cfield_alpha=0.2, cfield_color='#0099dd', thres=0.5,
                      lat=None, lon=None, aspect=None, fig=None, ax=None, colorbar=True, fontsize=10, verbose=False ) :
    """Plot reflectance field"""

    nlon, nlat = r.shape

    if vmax is None : vmax = r.max()
    if vmin is None : vmin = r.min()

    if vmin*vmax >= 0 :
        cmapn='gray'
        ctext='r'
    else :
        cmapn=refl_diff_colormap() #'RdBu'
        ctext='k'
    if not (cmap is None) :
        cmapn = cmap

    # determine aspect ratio
    if aspect is None or aspect == 'model' :
        dlat = (lat.max() - lat.min())/nlat
        dlon = (lon.max() - lon.min())/nlon
        aspect = dlat / ( dlon * np.cos((90-np.abs(lat.mean()))*np.pi/180.0) )
    elif aspect == 'satellite' :
        #print('Implement me!')
        aspect = r.shape[1] / r.shape[0]
    else :
        pass # just use aspect
    
    if verbose :
        print('dimensions   : ', nlon, nlat )
        print('aspect ratio : ', aspect )

    #fig = plt.figure(figsize=(8,8))
    #ax  = fig.add_axes(( 0.1,  0.1, 0.85, 0.85/aspect ))
    #cax = fig.add_axes(( 0.95, 0.1, 0.03, 0.85/aspect ))
    #img = ax.imshow( np.transpose(r[:,::-1]), vmin=vmin, vmax=vmax, extent=[ 0, nlon, 0, nlat ], aspect=aspect*float(nlat)/float(nlon), cmap=cmapn, interpolation='nearest' ) 
    #plt.colorbar( img, cax=cax )


    #fig, ax = plt.subplots( 1, 2, gridspec_kw={'width_ratios':[19,1],'wspace':0.01}, figsize=(8,8))

    if fig is None or ax is None :
        fig, ax = plt.subplots(figsize=(8,8))
        close_fig = True
    else :
        close_fig = False
        
    img = ax.imshow( np.transpose(r[:,::-1]), vmin=vmin, vmax=vmax, extent=[ 0, nlon, 0, nlat ], aspect=aspect, cmap=cmapn, interpolation='nearest' )
    if colorbar :
        plt.colorbar( img, ax=ax, shrink=0.5, pad=0.02 )

    #ax0, ay0, awidth, aheight = ax[0].get_position().bounds
    #print('AX0 x0, y0, width, height = ', ax0, ay0, awidth, aheight )
    #print('AX1 x0, y0, width, height = ', ax[1].get_position().bounds )
    #m_cax = 0.01
    #w_cax = 0.03
    #cax = fig.add_axes( [ax0+awidth+m_cax, ay0, w_cax, aheight] )  # [left, bottom, width, height]
    #plt.colorbar( img, cax=cax )
    

    #ax.transData.transform((5, 0))

    ax.text( 10, 10/aspect, name, color=ctext, size=fontsize )

    if grid :
        levels = np.arange(30,60)
        clat = ax.contour( np.transpose(lat), levels=levels, linewidths=0.3, colors='r', alpha=grid_alpha )
        clat = ax.contour( np.transpose(lat), levels=levels[::5], linewidths=0.3, colors='r', alpha=grid_alpha )
        #clabel( clat, levels[0::5], inline=1, fmt='%2.0f', fontsize=10 )

        levels = np.arange(0,30)
        clon = ax.contour( np.transpose(lon), levels=levels, linewidths=0.3, colors='r', alpha=grid_alpha )
        clon = ax.contour( np.transpose(lon), levels=levels[::5], linewidths=0.3, colors='r', alpha=grid_alpha )
        #clabel( clon, levels[0::5], inline=1, fmt='%2.0f', fontsize=10 )

    if not contour is None : # FIXME more options
        cnt_vals = [contour]
        cnt_cols = ['#00cccc']
        cnt_alpha = 1
        cnt = ax.contour( np.transpose(r), levels=cnt_vals, linewidths=0.3, colors=cnt_cols, alpha=cnt_alpha )

    if False : # FIXME old things from vis_op/plot_reflectance -- to be updated or removed...
        if not notext:
            text( 10, 10, name, color=ctext, size=8 ) #, family='TeX Gyre Adventor'
        if infos :
            if infostring is None :
                text( 10, 20, ('mean abs. value: %f, RMS %f' % ( abs(r).mean(), sqrt((r**2).mean()) ) ), color=ctext, size=8 )
            else :
                text( 10, 20, infostring, color=ctext, size=8 )

        if grid :
            if (lat is None) or (lon is None) :
                lat, lon = read_seviri_reflectance( coordinates=True, silence=True )

            levels = arange(30,60)
            clat = contour( transpose(lat), levels=levels, linewidths=0.3, colors='r', alpha=0.5 )
            clat = contour( transpose(lat), levels=levels[::5], linewidths=0.45, colors='r', alpha=0.5 )
            #clabel( clat, levels[0::5], inline=1, fmt='%2.0f', fontsize=10 )

            levels = arange(0,30)
            clon = contour( transpose(lon), levels=levels, linewidths=0.3, colors='r', alpha=0.5 )
            clon = contour( transpose(lon), levels=levels[::5], linewidths=0.45, colors='r', alpha=0.5 )
            #clabel( clon, levels[0::5], inline=1, fmt='%2.0f', fontsize=10 )

        if type(contours) == type(zeros(2)) :                                    # 0.6       0.65        0.7      0.75      0.8       0.85      0.9       0.95
            cnt = contour( transpose(r), levels=contours, linewidths=0.3, colors=['#ff00ff', '#6600ff', '#0000ff','#0099ff','#00cc00','#cccc00','#ff0000','#ff00ff'] )
            #clabel(cnt, inline=1, fontsize=10)
        else :
            if contours > 0 :
                cnt = contour( transpose(r), levels=linspace(vmin,vmax,contours), linewidths=0.3, colors='r' )
                clabel(cnt, inline=1, fontsize=10)
                
        if not spag is None :
            for i in reversed(sorted(spag.reflectance.keys())) :
                if i == -1 : # obs
                    lw=1
                    col='#0099dd'
                    al=1.0
                elif i == 0 : # det
                    lw = 0.5
                    col = '#990000'
                    al=1.0
                else :
                    lw = 0.3
                    col = 'k'
                    al=0.5
                contour( transpose(spag[i]), levels=[thres], linewidths=lw, colors=col, alpha=al )
            contourf( transpose(spag[-1]), levels=[thres,2], colors=col, alpha=0.2 )

    if not cfield is None :
        ax.contour( np.transpose(cfield), levels=[thres], linewidths=1.0, colors=cfield_color )
        if not cfield_alpha < 1e-6 :
            ax.contourf( np.transpose(cfield), levels=[thres,2], colors=cfield_color, alpha=cfield_alpha )

    if not save_to is None :
        fig.savefig( save_to, bbox_inches='tight' )

    if close_fig :
        plt.close(fig)

#-------------------------------------------------------------------------------
def plot_experiments( xps, args ) :

    lat = None
    lon = None

    # determine times
    valid_times = common_subset([ xp.valid_times['fg'] for xp in xps ])
    if not args.time_range is None :
        t0, t1 = args.time_range.split(',')
        valid_times = [ t for t in valid_times if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]

    for t in valid_times :
        # check if files are present
        fn_obs  = xps[0].get_filename( 'seviri_obs', time_valid=t, mem=0 )
        fn_det = [ xp.get_filename( 'seviri_sim', time_valid=t, mem=0 ) for xp in xps ]

        if np.all([ os.path.exists(f) for f in fn_det + [fn_obs] ]) :
            print('Plotting reflectances for time {} --> '.format(t), end='')

            # load data
            refl_obs, lat, lon = load_seviri_file( fn_obs, lat=lat, lon=lon, datool_coords=args.datool_coords )
            refl_det = [ load_seviri_file( f, lat=lat, lon=lon, datool_coords=args.datool_coords )[0] for f in fn_det ]

            # generate plot
            cbs = True
            scl = 0.7
            ncols = len(xps) + 1
            nrows = 2 if args.diff else 1
            fig, ax = plt.subplots( nrows, ncols, gridspec_kw={'wspace':0.08,'hspace':0.01}, figsize=( scl*8*ncols, scl*7*nrows ) )

            rax = ax if nrows == 1 else ax[0,:]
            plot_reflectance( refl_obs,  name='OBS '+t,                  lat=lat, lon=lon, grid=True, fig=fig, ax=rax[0], colorbar=cbs )
            for ixp, xp in enumerate(xps) :
                plot_reflectance( refl_det[ixp], name=xp.config['CY_EXPID'], lat=lat, lon=lon, grid=True, fig=fig, ax=rax[ixp+1], colorbar=cbs )

            if args.diff :
                rax = ax[1,:]
                if len(xps) == 2 :
                    plot_reflectance( refl_det[1]-refl_det[0], vmin=-1, vmax=1, name=xps[1].config['CY_EXPID']+'-'+xps[0].config['CY_EXPID'], lat=lat, lon=lon, grid=True, fig=fig, ax=rax[0], colorbar=cbs )
                for ixp, xp in enumerate(xps) :
                    plot_reflectance( refl_det[ixp]-refl_obs, vmin=-1, vmax=1, name=xp.config['CY_EXPID']+'-OBS', lat=lat, lon=lon, grid=True, fig=fig, ax=rax[ixp+1], colorbar=cbs )

            # save plot
            fn_plot = 'seviri_obs_vs_{}_{}{}.{}'.format( '_det'.join([xp.config['CY_EXPID'] for xp in xps]), t, '_diff' if args.diff else '', args.image_type)
            if not args.output_path is None :
                fn_plot = os.path.join( args.output_path, fn_plot )
            print(fn_plot)
            fig.savefig(fn_plot, bbox_inches='tight')
            plt.close(fig)
        else :
            print( 'files missing for t=', t )


#-------------------------------------------------------------------------------
def hist_error( mdl, obs, bins, calib=1.0, dbg=True ) :

    if dbg : fig, ax = plt.subplots()

    hist_obs, edges  = np.histogram( obs, bins )

    #print('HIST_OBS ', hist_obs)
    #print(type(hist_obs))
    #print(type(obs))

    hist_obs = hist_obs / float(obs.size)

    if dbg : ax.plot( 0.5*(bins[1:]+bins[:-1]), hist_obs, 'k' )

    err = []
    for c in list(calib) :
        hist_mdl, edges  = np.histogram( mdl * c, bins )
        hist_mdl = hist_mdl / float(mdl.size)

        if dbg : ax.plot( 0.5*(bins[1:]+bins[:-1]), hist_mdl, 'r', alpha=0.3 )

        e = ( np.abs( hist_obs - hist_mdl ) * (bins[1:]-bins[:-1]) ).sum()
        #print( c, e )

        err.append( e )

    if dbg :
        fig.savefig('hist_error_dbg.png')
        plt.close(fig) 

    if len(err) == 1 :
        return err[0]
    else :
        return np.array(err)
    

#-------------------------------------------------------------------------------
def compare_visop_datool( args ) :

    import xarray as xr
    from kendapy.cosmo_grid import cosmo_grid, nonrot_to_rot_grid

    refl_bins = np.arange( 0.0, 1.2001, 0.02 )
    rbc = 0.5*(refl_bins[:-1] + refl_bins[1:])

    refl_bins_err = np.arange( args.refl_thres, 1.2001, 0.02 )


    calib = 1.0 + np.arange( -0.2, 0.201, 0.01 )

    res_fcst = {}
    res_all  = {}

    cmp_main_fcst = args.cmp_visop_main

    lat_datool = None
    lon_datool = None
    for x in args.experiments :
        xp = BacyExp(x)

        fg_times_all = xp.valid_times['fg']
        #print('available fg times : ', fg_times_all )
        if not args.time_range is None :
            t0, t1 = args.time_range.split(',')
            fg_times_all = [ t for t in fg_times_all if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]

        fg_times = []
        for fgt in fg_times_all :
            mn = midnight( to_datetime(fgt) )
            t0 = mn + to_timedelta(args.start_daily)
            t1 = mn + to_timedelta(args.end_daily)
            t  = to_datetime(fgt)
            if (t >= t0) and (t <= t1) :
                fg_times.append(fgt)

        print('selected fg times : ', fg_times)

        for fgt in fg_times : # loop over fg times
        
            # determine file names

            if args.cmp_visop_main :
                fst = t2str( to_datetime(fgt) - to_timedelta('1:00') )
                visop_fname_main = xp.get_filename( 'visop', time_start=fst, lead_time=1, visop_type=args.visop_type )
        
            visop_fname = xp.get_filename( 'visop', time_valid=fgt, visop_type=args.visop_type )

            datool_sim_fname =  xp.get_filename( 'seviri_sim', time_valid=fgt )
            datool_obs_fname =  xp.get_filename( 'seviri_obs', time_valid=fgt )
                
            if  os.path.exists( visop_fname      ) and \
                os.path.exists( datool_sim_fname ) and \
                os.path.exists( datool_obs_fname ) :
                print('- visop  sim+obs ', visop_fname )
                if args.cmp_visop_main :
                    print('- visop  main    ', visop_fname_main )
                print('- datool sim     ', datool_sim_fname )
                print('- datool obs     ', datool_obs_fname )

                # read visop and datool reflectances
                ds = xr.open_dataset(visop_fname)
                #print( ds.variables.keys())
                refl_mdl_visop = np.array( ds.synthetic_reflectance_VIS006[:] )
                refl_obs_visop = np.array( ds.observed_reflectance_VIS006[:] )
                lat_visop = np.array( ds.latitude[:] )
                lon_visop = np.array( ds.longitude[:] )
                
                print( 'REFL VISOP ', refl_obs_visop.shape, refl_obs_visop.mean(), refl_obs_visop.max() )

                if args.cmp_visop_main :
                    dsm = xr.open_dataset(visop_fname_main)
                    #print( dsm.variables.keys())
                    refl_mdl_visop_main = np.array( dsm.synthetic_reflectance_VIS006[:] )
                    refl_obs_visop_main = np.array( dsm.observed_reflectance_VIS006[:] )
                    print( 'REFL MAIN  ', refl_obs_visop_main.shape, refl_obs_visop_main.mean(), refl_obs_visop_main.max() )
                    print( 'REFL VISOP-MAIN obs max ', np.abs(refl_obs_visop-refl_obs_visop_main).max() )
                    print( 'REFL VISOP-MAIN mdl max / mean ', np.abs(refl_mdl_visop-refl_mdl_visop_main).max(),
                                                              np.abs(refl_mdl_visop-refl_mdl_visop_main).mean() )

                    plot_reflectance( np.transpose(refl_mdl_visop - refl_mdl_visop_main), vmin=-0.2, vmax=0.2,
                                      name='FG-MAIN', lat=np.transpose(lat_visop), lon=np.transpose(lon_visop), grid=True,
                                      save_to=fgt+'/fg_minus_main.png' )
                    plot_reflectance( np.transpose(refl_mdl_visop), vmin=0, vmax=1.2,
                                      name='FG '+fgt, lat=np.transpose(lat_visop), lon=np.transpose(lon_visop), grid=True,
                                      save_to=fgt+'/fg.png' )
                    plot_reflectance( np.transpose(refl_mdl_visop_main), vmin=0, vmax=1.2,
                                      name='FG '+fgt, lat=np.transpose(lat_visop), lon=np.transpose(lon_visop), grid=True,
                                      save_to=fgt+'/main.png' )


                refl_mdl_datool, lat_datool, lon_datool = load_seviri_file( datool_sim_fname, transpose=False,
                                                            lat=lat_datool, lon=lon_datool, datool_coords=args.datool_coords, verbose=True )
                refl_obs_datool, lat_datool, lon_datool = load_seviri_file( datool_obs_fname, transpose=False,
                                                            lat=lat_datool, lon=lon_datool, datool_coords=args.datool_coords, verbose=True )
                print( 'REFL DATOOL ', refl_obs_datool.shape, refl_obs_datool.mean(), refl_obs_datool.max() )

                if False : # rot-lat-lon evaluation area
                    gconf = 'COSMO-D2'
                    rlon_min, rlon_max = -5.5, 3.6
                    rlat_min, rlat_max = -5.8, 5.0
                    rlat_visop,  rlon_visop  = nonrot_to_rot_grid( lat_visop,  lon_visop,  configuration=gconf )
                    rlat_datool, rlon_datool = nonrot_to_rot_grid( lat_datool, lon_datool, configuration=gconf )

                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.scatter( rlon_datool, rlat_datool, s=1, c='r')
                    ax.scatter( rlon_visop,  rlat_visop,  s=1, c='b', alpha=0.5)
                    ax.axvline( x=rlon_min, color='k' )
                    ax.axvline( x=rlon_max, color='k' )
                    ax.axhline( y=rlat_min, color='k' )
                    ax.axhline( y=rlat_max, color='k' )
                    #ax.axvline( x=[rlon_min,rlon_max], color='k' )
                    #ax.axhline( y=[rlat_min,rlat_max], color='k' )
                    fig.savefig('rgrid.png', bbox_inches='tight')
                    plt.close(fig)

                # define lat-lon evaluation area
                lon_min, lon_max = 0.1, 15.8
                lat_min, lat_max = 44.3, 55.3


                if args.plot_each_fcst :

                    print('MKDIR ',fgt)
                    if not os.path.exists(fgt) :
                        os.mkdir(fgt)

                    # generate plots
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.scatter( lon_datool, lat_datool, s=1, c='r')
                    ax.scatter( lon_visop,  lat_visop,  s=1, c='b', alpha=0.5)
                    ax.axvline( x=lon_min, color='k' )
                    ax.axvline( x=lon_max, color='k' )
                    ax.axhline( y=lat_min, color='k' )
                    ax.axhline( y=lat_max, color='k' )
                    fig.savefig( fgt+'/grid.png', bbox_inches='tight')
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.imshow( refl_obs_visop, vmin=0, vmax=1, origin='lower' )
                    fig.savefig( fgt+'/refl_obs_visop.png', bbox_inches='tight')
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.imshow( refl_mdl_visop, vmin=0, vmax=1, origin='lower' )
                    fig.savefig( fgt+'/refl_mdl_visop.png', bbox_inches='tight')
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.imshow( np.transpose(refl_obs_datool), vmin=0, vmax=1, origin='lower' )
                    fig.savefig( fgt+'/refl_obs_datool.png', bbox_inches='tight')
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.imshow( np.transpose(refl_mdl_datool), vmin=0, vmax=1, origin='lower' )
                    fig.savefig( fgt+'/refl_mdl_datool.png', bbox_inches='tight')
                    plt.close(fig)

                # generate histograms for common evaluation area
                idcs_visop  = np.where(  (lat_visop>=lat_min) & (lat_visop<=lat_max) &
                                            (lon_visop>=lon_min) & (lon_visop<=lon_max) )
                n_visop = len(idcs_visop[0])
                idcs_datool = np.where(  (lat_datool>=lat_min) & (lat_datool<=lat_max) &
                                            (lon_datool>=lon_min) & (lon_datool<=lon_max) )
                n_datool = len(idcs_datool[0])

                hist_obs_visop, edges  = np.histogram( refl_obs_visop[idcs_visop],   refl_bins )
                hist_mdl_visop, edges  = np.histogram( refl_mdl_visop[idcs_visop],   refl_bins )
                hist_obs_datool, edges = np.histogram( refl_obs_datool[idcs_datool], refl_bins )
                hist_mdl_datool, edges = np.histogram( refl_mdl_datool[idcs_datool], refl_bins )

                calerr_visop  = hist_error( refl_mdl_visop[idcs_visop],
                                            refl_obs_visop[idcs_visop], refl_bins_err, calib=calib )
                calerr_datool = hist_error( refl_mdl_datool[idcs_datool],
                                            refl_obs_datool[idcs_datool], refl_bins_err, calib=calib )
                calerr_obs    = hist_error( refl_obs_visop[idcs_visop],
                                            refl_obs_datool[idcs_datool], refl_bins_err, calib=calib )
                calerr_mdl    = hist_error( refl_mdl_visop[idcs_visop],
                                            refl_mdl_datool[idcs_datool], refl_bins_err, calib=calib )

                # store and accumulate results
                res_fcst[fgt] = {   'rmse_visop':  np.sqrt( ((refl_mdl_visop[idcs_visop]   - refl_obs_visop[idcs_visop]  )**2).mean() ),
                                    'rmse_datool': np.sqrt( ((refl_mdl_datool[idcs_datool] - refl_obs_datool[idcs_datool])**2).mean() ),
                                    'bias_visop':            (refl_mdl_visop[idcs_visop]   - refl_obs_visop[idcs_visop]      ).mean(),
                                    'bias_datool':           (refl_mdl_datool[idcs_datool] - refl_obs_datool[idcs_datool]    ).mean(),
                                    'meanrefl_mdl_visop':refl_mdl_visop[idcs_visop].mean(),
                                    'meanrefl_obs_visop':refl_obs_visop[idcs_visop].mean(),
                                    'meanrefl_mdl_datool':refl_mdl_datool[idcs_datool].mean(),
                                    'meanrefl_obs_datool':refl_obs_datool[idcs_datool].mean(),
                                    'hist_obs_visop':hist_obs_visop,
                                    'hist_mdl_visop':hist_mdl_visop,
                                    'hist_obs_datool':hist_obs_datool,
                                    'hist_mdl_datool':hist_mdl_datool,
                                    'calerr_visop':calerr_visop,
                                    'calerr_datool':calerr_datool,
                                    'calerr_obs':calerr_obs,
                                    'calerr_mdl':calerr_mdl,
                                    'n_visop':n_visop, 'n_datool':n_datool }

                if args.cmp_visop_main :
                    res_fcst[fgt]['rmse_visop_main'] = np.sqrt( ((refl_mdl_visop_main[idcs_visop]   - refl_obs_visop_main[idcs_visop]  )**2).mean() )
                    res_fcst[fgt]['bias_visop_main'] =           (refl_mdl_visop_main[idcs_visop]   - refl_obs_visop_main[idcs_visop]      ).mean()

                for q in res_fcst[fgt] :
                    if not q in res_all :
                        res_all[q] = res_fcst[fgt][q] + 0
                    else :
                        res_all[q] += res_fcst[fgt][q]

                res_fcst[fgt]['cal_visop']  = calib[np.argmin(calerr_visop)]
                res_fcst[fgt]['cal_datool'] = calib[np.argmin(calerr_datool)]
                res_fcst[fgt]['cal_obs']    = calib[np.argmin(calerr_obs)]
                res_fcst[fgt]['cal_mdl']    = calib[np.argmin(calerr_mdl)]
                print('CALIB ', res_fcst[fgt]['cal_visop'], res_fcst[fgt]['cal_datool'],
                                res_fcst[fgt]['cal_obs'], res_fcst[fgt]['cal_mdl'] )

                if args.plot_each_fcst :
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.plot( rbc, hist_obs_visop/n_visop, 'k', label='obs visop' )
                    ax.plot( rbc, hist_mdl_visop/n_visop, 'r', label='model visop')
                    ax.plot( rbc, hist_obs_datool/n_datool, '--k', label='obs datool')
                    ax.plot( rbc, hist_mdl_datool/n_datool, '--r', label='model datool')
                    ax.legend( title=fgt, loc='center right' )
                    ax.text( 0.05, ax.get_ylim()[1]*0.99,
                        'visop obs/mdl {:.2f}, datool obs/mdl {:.2f}, obs datool/visop {:.2f}, mdl datool/visop {:.2f}'.format(
                        res_fcst[fgt]['cal_visop'], res_fcst[fgt]['cal_datool'], res_fcst[fgt]['cal_obs'], res_fcst[fgt]['cal_mdl'] ),
                        fontsize=8, color='b', va='top' )
                    fig.savefig( fgt+'/refl_hist.png', bbox_inches='tight')
                    plt.close(fig)

            else :
                print('FILE MISSING ', os.path.exists( visop_fname ), os.path.exists( datool_sim_fname ), os.path.exists( datool_obs_fname ))
                print('- visop  sim+obs ', visop_fname )
                if args.cmp_visop_main :
                    print('- visop  main    ', visop_fname_main )
                print('- datool sim     ', datool_sim_fname )
                print('- datool obs     ', datool_obs_fname )

        # correct reflectances
        for q in res_all :
            if q.startswith('refl') :
                res_all[q] /= len(list(res_fcst.keys()))

        # plots results for full period

        fcsts = sorted(list(res_fcst.keys()))
        n_fcsts = len(fcsts)

        # full period calibration factors
        # maybe not the right way -- days with low histogram errors will be underrepresented
        res_all['cal_visop']  = calib[np.argmin(res_all['calerr_visop'])]
        res_all['cal_datool'] = calib[np.argmin(res_all['calerr_datool'])]
        res_all['cal_obs']    = calib[np.argmin(res_all['calerr_obs'])]
        res_all['cal_mdl']    = calib[np.argmin(res_all['calerr_mdl'])]

        calerr_visop_period = np.zeros(( len(fcsts), res_all['calerr_visop'].size ))
        for i in range(n_fcsts) :
            calerr_visop_period[ i, : ] = res_fcst[fcsts[i]]['calerr_visop']


        fig, ax = plt.subplots(figsize=(8,8))
        mpb = ax.imshow( np.transpose(calerr_visop_period), origin='lower', cmap='magma', vmin=0 )
        ax.set_xticks( np.arange(len(fcsts)) )
        ax.set_xticklabels( ['{}/{}/{} {}UTC'.format(f[:4],f[4:6],f[6:8],f[8:10]) for f in fcsts], rotation='vertical', fontsize='small' )
        plt.colorbar(mpb, shrink=0.5)
        fig.savefig( 'calerr_visop_period.png', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,8))
        for i in range(n_fcsts) :
            ax.plot( calib, res_fcst[fcsts[i]]['calerr_visop'], 'k', alpha=0.3, label='days' if i == 0 else None )
            ii = np.argmin( res_fcst[fcsts[i]]['calerr_visop'] )
            ax.scatter( calib[ii], res_fcst[fcsts[i]]['calerr_visop'][ii], color='k' )
        ax.plot( calib, calerr_visop_period.mean(axis=0), 'k', linewidth=2, label='period' )
        ax.plot( [res_all['cal_visop']]*2, (0,calerr_visop_period.max()), 'r', label='minimum' )
        ax.legend( title=xp.exp_dir, loc='lower left' )
        ax.set_ylim(bottom=0)
        ax.set_ylabel('histogram error')
        ax.set_xlabel('observation scaling factor')
        ax.grid(alpha=0.5)
        fig.savefig( 'calerr_visop_lines.png', bbox_inches='tight')
        plt.close(fig)


        # mean reflectances
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot( [ res_fcst[fgt]['meanrefl_obs_visop']   for fgt in fcsts ], 'k', label='obs visop' )
        ax.plot( [ res_fcst[fgt]['meanrefl_mdl_visop']   for fgt in fcsts ], 'r', label='model visop')
        ax.plot( [ res_fcst[fgt]['meanrefl_obs_datool']  for fgt in fcsts ], '--k', label='obs datool')
        ax.plot( [ res_fcst[fgt]['meanrefl_mdl_datool']  for fgt in fcsts ], '--r', label='model datool')
        ax.legend()
        ax.grid(alpha=0.5)
        fig.savefig( 'mean_refl.png', bbox_inches='tight')
        plt.close(fig)

        # reflectance RMSEs
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot( [ res_fcst[fgt]['rmse_visop']   for fgt in fcsts ], 'k', label='rmse visop' )
        ax.plot( [ res_fcst[fgt]['rmse_datool']  for fgt in fcsts ], '--k', label='rmse datool')
        ax.plot( [ res_fcst[fgt]['bias_visop']   for fgt in fcsts ], 'b', label='bias visop' )
        ax.plot( [ res_fcst[fgt]['bias_datool']  for fgt in fcsts ], '--b', label='bias datool')
        if args.cmp_visop_main :
            ax.plot( [ res_fcst[fgt]['rmse_visop_main']   for fgt in fcsts ], ':k', label='rmse visop main' )
            ax.plot( [ res_fcst[fgt]['bias_visop_main']   for fgt in fcsts ], ':b', label='bias visop main' )

        ax.set_xticks( np.arange(len(fcsts)) )
        ax.set_xticklabels( ['{}/{}/{} {}UTC'.format(f[:4],f[4:6],f[6:8],f[8:10]) for f in fcsts], rotation='vertical', fontsize='small' )
        ax.legend( title=xp.exp_dir )
        ax.grid(alpha=0.5)
        fig.savefig( 'rmse_bias.png', bbox_inches='tight')
        plt.close(fig)

        # histogram for all fcsts
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot( rbc, res_all['hist_obs_visop']  / res_all['n_visop'],    'k', label='obs visop' )
        ax.plot( rbc, res_all['hist_mdl_visop']  / res_all['n_visop'],    'r', label='model visop')
        ax.plot( rbc, res_all['hist_obs_datool'] / res_all['n_datool'], '--k', label='obs datool')
        ax.plot( rbc, res_all['hist_mdl_datool'] / res_all['n_datool'], '--r', label='model datool')
        ax.legend()
        fig.savefig( 'refl_hist.png', bbox_inches='tight')
        plt.close(fig)

        # calibration factors minimizing histogram error
        fig, ax = plt.subplots(figsize=(8,8))

        ax.plot( [ res_fcst[fgt]['cal_visop']  for fgt in fcsts ],   '#666666', label='visop:  obs / mdl' )
        ax.plot( (0,len(fcsts)-1), [ res_all['cal_visop'] ]*2,    '#666666', linewidth=0.5 )

        ax.plot( [ res_fcst[fgt]['cal_datool'] for fgt in fcsts ],  '--k', label='datool: obs / mdl')
        ax.plot( (0,len(fcsts)-1), [ res_all['cal_datool'] ]*2,  '--k', linewidth=0.5 )

        ax.plot( [ res_fcst[fgt]['cal_obs']    for fgt in fcsts ],   'b', label='obs:    datool / visop')
        ax.plot( (0,len(fcsts)-1), [ res_all['cal_obs'] ]*2,     'b', linewidth=0.5 )

        ax.plot( [ res_fcst[fgt]['cal_mdl']    for fgt in fcsts ],   'r', label='model:  datool / visop')
        ax.plot( (0,len(fcsts)-1), [ res_all['cal_mdl'] ]*2,     'r', linewidth=0.5 )

        ax.set_ylabel('factor minimizing hist. error')
        #ax.set_xticklabels( ['{}'.format(f) for f in fcsts], rotation='vertical' )
        ax.set_xticks( np.arange(len(fcsts)) )
        ax.set_xticklabels( ['{}/{}/{} {}UTC'.format(f[:4],f[4:6],f[6:8],f[8:10]) for f in fcsts], rotation='vertical', fontsize='small' )
        ax.legend()
        fig.savefig( 'cal.png', bbox_inches='tight')
        plt.close(fig)


def get_forecasts_error( xps, args ) :

    lead_times  = common_subset([ xp.fc_lead_times  for xp in xps ])
    start_times = common_subset([ xp.fc_start_times for xp in xps ])
    if not args.time_range is None :
        t0, t1 = args.time_range.split(',')
        start_times = [ t for t in start_times if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]

    print('start times: ', start_times)
    print('lead  times: ', lead_times)

    # define lat-lon evaluation area
    lon_min, lon_max = 1.0, 15.0
    lat_min, lat_max = 48.0, 55.0

    res = {}   # results for all start and lead times
    metrics = ['rmse','mae','bias','meanrefl_mdl','meanrefl_obs','fss24km03','fss24km05','fss24km07']
    if args.belscl :
        metrics += ['belscl03', 'belscl05', 'belscl07']
    for metric in metrics  :
        res[metric] = np.zeros(( len(start_times), len(lead_times), len(xps) ))
        res[metric][...] = np.nan

    for ifst, fst in enumerate(start_times) :
        print()
        print('START TIME ', fst, ' -> LEAD TIMES ', end='')

        for ildt, ldt in enumerate(lead_times) :

            # check if we are in the daily time window
            t  = to_datetime(fst) + to_timedelta(str(ldt)+'h')
            mn = midnight( t )
            t0 = mn + to_timedelta(args.start_daily)
            t1 = mn + to_timedelta(args.end_daily)
            if (t >= t0) and (t <= t1) :

                nfls = 0
                for ixp, xp in enumerate(xps) :

                    # read data
                    visop_fname = xp.get_filename( 'visop', time_start=fst, lead_time=ldt, visop_type=args.visop_type )
                    if os.path.exists(visop_fname) :
                        nfls += 1

                        #print('reading ',visop_fname)
                        ds = xr.open_dataset(visop_fname)
                        #print( ds.variables.keys())
                        refl_mdl_visop = np.array( ds.synthetic_reflectance_VIS006[:] )
                        refl_obs_visop = np.array( ds.observed_reflectance_VIS006[:] )
                        lat_visop = np.array( ds.latitude[:] )
                        lon_visop = np.array( ds.longitude[:] )

                        # compute metrics
                        idcs_visop  = np.where( (lat_visop>=lat_min) & (lat_visop<=lat_max) &
                                                (lon_visop>=lon_min) & (lon_visop<=lon_max) )
                        n_visop = len(idcs_visop[0])

                        #import pdb; pdb.set_trace()
                        res['rmse'        ][ ifst, ildt, ixp ] = np.sqrt( (        ( refl_mdl_visop[idcs_visop] - refl_obs_visop[idcs_visop] )**2 ).mean() )
                        res['mae'         ][ ifst, ildt, ixp ] =          np.abs(    refl_mdl_visop[idcs_visop] - refl_obs_visop[idcs_visop]      ).mean()
                        res['bias'        ][ ifst, ildt, ixp ] =          (          refl_mdl_visop[idcs_visop] - refl_obs_visop[idcs_visop]      ).mean()
                        res['meanrefl_mdl'][ ifst, ildt, ixp ] = refl_mdl_visop[idcs_visop].mean()
                        res['meanrefl_obs'][ ifst, ildt, ixp ] = refl_obs_visop[idcs_visop].mean()
                        
                        # fss qwer
                        #print('visop image size ', refl_mdl_visop.shape) # (243, 455)
                        #d_sl=87, d_fov=103
                        i1 = 87;  i2 = i1 + 128
                        j1 = 103; j2 = j1 + 256

                        #refl_mdl_visop[i1:i2,j1:j2] = 0.5
                        #plot_reflectance( np.transpose(refl_mdl_visop), vmin=0, vmax=1.2, name='mdl',
                        #                  lat=np.transpose(lat_visop), lon=np.transpose(lon_visop), grid=True, save_to='mdl.png' )

                        # observation and model equivalent with (approx.) 6km x 6km Pixels
                        rmq = 0.5*( refl_mdl_visop[i1:i2,j1:j2-1:2] + refl_mdl_visop[i1:i2,j1+1:j2:2] )
                        roq = 0.5*( refl_obs_visop[i1:i2,j1:j2-1:2] + refl_obs_visop[i1:i2,j1+1:j2:2] )
                        #print('quadratic pixel image size ', rmq.shape )

                        if args.belscl : # takes some time because we need a lot of windows...
                            #fd = fss_dict( rmq, roq, [1,2,4,8,16,32], [0.3,0.5,0.7], believable_scale=True, target=True )
                            fd = fss_dict( rmq, roq, np.arange(2,33), [0.3,0.5,0.7], believable_scale=True, target=True )
                            #print('FSS ',  fd )                    
                            res['belscl03'][ ifst, ildt, ixp ] = fd['belscl'][0] * 6 # km
                            res['belscl05'][ ifst, ildt, ixp ] = fd['belscl'][1] * 6 # km
                            res['belscl07'][ ifst, ildt, ixp ] = fd['belscl'][2] * 6 # km

                        # FSS
                        fd03 = fss_dict( rmq, roq, [4], [0.3], believable_scale=False, target=True )
                        res['fss24km03'][ ifst, ildt, ixp ] = fd03['fss'][0,0]
                        fd05 = fss_dict( rmq, roq, [4], [0.5], believable_scale=False, target=True )
                        res['fss24km05'][ ifst, ildt, ixp ] = fd05['fss'][0,0]
                        fd07 = fss_dict( rmq, roq, [4], [0.7], believable_scale=False, target=True )
                        res['fss24km07'][ ifst, ildt, ixp ] = fd07['fss'][0,0]

                        #sys.exit(0)
                        #fd = fss_dict( fcst, obs, windows, levels, believable_scale=False, target=False, periodic=False ) :
                    #else :
                    #    print('MISSING FILE: ', visop_fname)

                print('[{}/{}] '.format(ldt,nfls), end='')

    print()
    return start_times, lead_times, res
                    
def plot_forecasts_error( xps, args ) :

    start_times, lead_times, res = get_forecasts_error( xps, args )

    start_hour = [ to_datetime(t).hour for t in start_times ]
    print('start hour vector ', start_hour)
    start_hours = set(start_hour)
    print('start hour set ', start_hours)

    colors = default_color_sequence() # ['r','b','k'] 

    # RMSE, MAE time evolution
    # for metric in ['rmse','mae'] :
    #     fig, ax = plt.subplots(figsize=(10,3))
    #     for ixp, xp in enumerate(xps) :
    #         for ifst, fst in enumerate(start_times) :
    #             dt = [ to_datetime(fst) + to_timedelta(str(ldt)+'h') for ldt in lead_times ]
    #             ax.plot_date( dt, [ res[metric][ ifst, ildt, ixp ] for ildt in range(len(lead_times)) ], '-', color=colors[ixp] )
    #             ax.plot_date( dt, [ res['bias'][ ifst, ildt, ixp ] for ildt in range(len(lead_times)) ], '--', color=colors[ixp] )
    #             #if ixp == 0 :
    #             #    ax.plot_date( [dt[0],dt[0]], [0,0.1], '--k', linewidth=0.5 )
    #     #ax.grid()
    #     fig.savefig('fcsts_refl_{}.png'.format(metric))
    #     plt.close(fig)

    # mean rmse as function of lead time
    #for metric in ['rmse','mae'] :
    #    fig, ax = plt.subplots(figsize=(10,3))
    #    for ixp, xp in enumerate(xps) :
    #        mean_metric = [ np.nanmean( np.array([ res[metric][ ifst, ildt, ixp ] for ifst in range(len(start_times)) ]) )
    #                        for ildt in range(len(lead_times)) ]
    #        ax.plot( lead_times, mean_metric, '-', color=colors[ixp] )
    #    fig.savefig('fcsts_mean_refl_{}.png'.format(metric))
    #    plt.close(fig)

    # separate plots for different start hours
    for sh in start_hours :
        ifsts = [ i for i in range(len(start_times)) if start_hour[i] == sh ]
        print('sh=', sh, 'ifsts=', ifsts)

        # RMSE, MAE time evolution
        for metric in ['rmse','mae','fss24km03','fss24km05','fss24km07'] :
            plot_bias = True if metric in ['rmse','mae'] else False
            fig, ax = plt.subplots(figsize=(10,3))
            for ixp, xp in enumerate(xps) :
                for ifst in ifsts : # , fst in enumerate(start_times) :
                    fst = start_times[ifst]
                    dt = [ to_datetime(fst) + to_timedelta(str(ldt)+'h') for ldt in lead_times ]
                    ax.plot_date( dt, [ res[metric][ ifst, ildt, ixp ] for ildt in range(len(lead_times)) ], '-', color=colors[ixp], label=xp.exp_dir if ifst==ifsts[0] else None )
                    if plot_bias :
                        ax.plot_date( dt, [ res['bias'][ ifst, ildt, ixp ] for ildt in range(len(lead_times)) ], '--', color=colors[ixp] )
                    #if ixp == 0 :
                    #    ax.plot_date( [dt[0],dt[0]], [0,0.1], '--k', linewidth=0.5 )
            ax.axhline( y=0, color='k',linewidth=0.5, alpha=0.5 )
            ax.set_ylabel( ('bias, ' if plot_bias else '') + metric )
            ax.legend( title='fcsts started at {}UTC'.format(sh), frameon=False )
            ax.grid(alpha=0.25)
            yl, yh = ax.get_ylim()
            for ifst in ifsts :
                dt = to_datetime(start_times[ifst])
                ax.plot_date( (dt,dt), (yl,yh), '--r', alpha=0.5 )
            #fig.savefig('fcsts_refl_{}.png'.format(metric))
            fig.savefig('fcsts_refl_{}_starthour_{:02d}UTC.png'.format(metric,sh), bbox_inches='tight')
            plt.close(fig)

        # mean rmse as function of lead time
        metrics = ['rmse','mae','bias','fss24km03','fss24km05','fss24km07']
        if args.belscl :
            metrics += ['belscl03','belscl05','belscl07']
        for metric in metrics :
            fig, ax = plt.subplots(figsize=(5,3))
            for ixp, xp in enumerate(xps) :
                mean_metric = [ np.nanmean( np.array([ res[metric][ ifst, ildt, ixp ] for ifst in ifsts ]) )
                            for ildt in range(len(lead_times)) ]
                tms = np.array(lead_times) + sh
                ax.plot( tms, mean_metric, '-', color=colors[ixp], label=xp.exp_dir )
            ax.set_xticks(      [ t      for t in tms if t % 6 == 0 ] )
            ax.set_xticklabels( [ t % 24 for t in tms if t % 6 == 0 ] ) 
            ax.grid(alpha=0.5)
            if metric != 'bias': 
                ax.set_ylim(bottom=0)
            if metric.startswith('fss') :
                ax.set_ylim((0,1))
            ax.set_xlim(left=sh)
            ax.set_xlabel('hour [UTC]')
            ax.set_ylabel(metric)
            ax.legend( title='fcsts started at {}UTC'.format(sh), frameon=False )
            fig.savefig('fcsts_mean_refl_{}_starthour_{:02d}UTC.png'.format(metric,sh), bbox_inches='tight')
            plt.close(fig)


    # RMSE/MAE time evolution for constant start hour
    # for metric in ['rmse','mae'] :
    #     for sh in start_hours :
    #         ifsts = [ i for i in range(len(start_times)) if start_hour[i] == sh ]

    #         fig, ax = plt.subplots(figsize=(10,3))
    #         for ixp, xp in enumerate(xps) :
    #             for ifst in ifsts :
    #                 fst = start_times[ifst]
    #                 dt = [ to_datetime(fst) + to_timedelta(str(ldt)+'h') for ldt in lead_times ]
    #                 ax.plot_date( dt, [ res[metric][ ifst, ildt, ixp ] for ildt in range(len(lead_times)) ], '-', color=colors[ixp] )
    #                 if ixp == 0 :
    #                     ax.plot_date( [dt[0],dt[0]], [0,0.28], '--k', linewidth=0.5 )
    #         #ax.grid()
    #         fig.savefig('fcsts_refl_{}_starthour{:02d}.png'.format(metric,sh))
    #         plt.close(fig)


def compare_visop_types( args ) :

    use_datool = args.datool

    xp = BacyExp( args.experiments[0] )

    fg_times_all = xp.valid_times['fg']
    if not args.time_range is None :
        t0, t1 = args.time_range.split(',')
        fg_times_all = [ t for t in fg_times_all if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]

    fg_times = []
    for fgt in fg_times_all :
        mn = midnight( to_datetime(fgt) )
        t0 = mn + to_timedelta(args.start_daily)
        t1 = mn + to_timedelta(args.end_daily)
        t  = to_datetime(fgt)
        if (t >= t0) and (t <= t1) :
            fg_times.append(fgt)

    print('selected fg times : ', fg_times)

    # evaluation rectangle
    latmin, lonmin, latmax, lonmax = [ float(v) for v in args.latlon.split(',') ]

    refl_bins = np.arange( 0.0, 1.2001, 0.02 )
    rbc = 0.5*(refl_bins[:-1] + refl_bins[1:])
    refl_binsize = refl_bins[1] - refl_bins[0]

    if args.visop_type is None :
        vtypes = ['std']
    else :
        vtypes = args.visop_type.split(',')

    hist_obs, hist_mdl = {}, {}
    for vt in vtypes :
        hist_obs[vt] = None
        hist_mdl[vt] = None
    
    lat_datool, lon_datool = None, None
    hist_obs_datool = None
    hist_mdl_datool = None

    for fgt in fg_times : # loop over fg times
        print('-- TIME ', fgt )

        for vt in vtypes :

            # determine file names       
            visop_fname = xp.get_filename( 'visop', time_valid=fgt, visop_type=vt if vt != 'std' else '' )
            if not args.visop_path is None :
                visop_fname = args.visop_path + visop_fname.split('visop/')[-1]
            #print('visop_fname = ', visop_fname)


            # read visop reflectances
            ds = xr.open_dataset(visop_fname)
            #print( ds.variables.keys())
            refl_mdl_visop = np.array( ds.synthetic_reflectance_VIS006[:] )
            refl_obs_visop = np.array( ds.observed_reflectance_VIS006[:] )
            lat_visop = np.array( ds.latitude[:] )
            lon_visop = np.array( ds.longitude[:] )
            
            valid_visop = np.array( ds.pixel_valid[:] )

            valid_visop[ np.where(lon_visop < lonmin) ] = -1
            valid_visop[ np.where(lon_visop > lonmax) ] = -1
            valid_visop[ np.where(lat_visop < latmin) ] = -1
            valid_visop[ np.where(lat_visop > latmax) ] = -1

            idcs = np.where(valid_visop > 0 )

            hist_obs_visop, edges  = np.histogram( refl_obs_visop[idcs],   refl_bins )
            hist_mdl_visop, edges  = np.histogram( refl_mdl_visop[idcs],   refl_bins )

            hist_obs[vt] = hist_obs_visop + hist_obs[vt] if not hist_obs[vt] is None else 0
            hist_mdl[vt] = hist_mdl_visop + hist_mdl[vt] if not hist_mdl[vt] is None else 0

            print( '  -- {:30s}  /  MEAN obs {:5.3f} mdl {:5.3f}  /  MAX obs {:5.3f} mdl {:5.3f}  /  {} pixels = {:.1f} percent valid '.format( \
                vt, refl_obs_visop[idcs].mean(), refl_mdl_visop[idcs].mean(), refl_obs_visop[idcs].max(), refl_mdl_visop[idcs].max(), \
                len(idcs[0]), 100.0*len(idcs[0])/refl_obs_visop.size ))

            plot_reflectance( np.transpose(refl_mdl_visop - refl_obs_visop), vmin=-0.5, vmax=0.5,
                                name='FG-OBS '+vt+(' ({}px valid)'.format(len(idcs[0]))), lat=np.transpose(lat_visop), lon=np.transpose(lon_visop), grid=True,
                                cfield=np.transpose(valid_visop), cfield_alpha=0.2, cfield_color='#009900', thres=0.5,
                                save_to=fgt+'_'+vt+'__fg_minus_obs.png' )

        if use_datool :
            lat_datool, lon_datool = None, None
            
            datool_mdl_fname =  xp.get_filename( 'seviri_sim', time_valid=fgt )
            datool_obs_fname =  xp.get_filename( 'seviri_obs', time_valid=fgt )
            if not os.path.exists(datool_mdl_fname) :
                if os.path.exists(datool_mdl_fname.replace('.grb','_ens000.grb')) :
                    datool_mdl_fname = datool_mdl_fname.replace('.grb','_ens000.grb')

            refl_mdl_datool, lat_datool, lon_datool = load_seviri_file( datool_mdl_fname, transpose=False,
                                                        lat=lat_datool, lon=lon_datool, datool_coords=args.datool_coords, verbose=False )
            refl_obs_datool, lat_datool, lon_datool = load_seviri_file( datool_obs_fname, transpose=False,
                                                        lat=lat_datool, lon=lon_datool, datool_coords=args.datool_coords, verbose=False )

            idcs_datool = np.where( (lat_datool >= latmin) & (lat_datool <= latmax) & (lon_datool >= lonmin) & (lon_datool <= lonmax) & (refl_mdl_datool >= 0) & (refl_obs_datool >= 0))
            valid_datool = np.zeros(lat_datool.shape)
            valid_datool[idcs_datool] = 1.0

            hist_obs_datool_, edges  = np.histogram( refl_obs_datool[idcs_datool],   refl_bins )
            hist_mdl_datool_, edges  = np.histogram( refl_mdl_datool[idcs_datool],   refl_bins )

            hist_obs_datool = hist_obs_datool_ + hist_obs_datool if not hist_obs_datool is None else 0
            hist_mdl_datool = hist_mdl_datool_ + hist_mdl_datool if not hist_mdl_datool is None else 0

            print( '  -- {:30s}  /  MEAN obs {:5.3f} mdl {:5.3f}  /  MAX obs {:5.3f} mdl {:5.3f}  /  {} pixels = {:.1f} percent valid '.format( \
                'DATOOL', refl_obs_datool[idcs_datool].mean(), refl_mdl_datool[idcs_datool].mean(), \
                          refl_obs_datool[idcs_datool].max(),  refl_mdl_datool[idcs_datool].max(), \
                          len(idcs_datool[0]), 100.0*len(idcs_datool[0])/refl_obs_datool.size ))

            refldif = refl_mdl_datool * 0
            refldif[idcs_datool] = refl_mdl_datool[idcs_datool] - refl_obs_datool[idcs_datool]

            plot_reflectance( refldif[::-1,::-1], vmin=-0.5, vmax=0.5, aspect=1.3,
                                name='FG-OBS datool ({}px valid)'.format(len(idcs_datool[0])), lat=lat_datool[::-1,::-1], lon=lon_datool[::-1,::-1], grid=True,
                                cfield=valid_datool[::-1,::-1], cfield_alpha=0.2, cfield_color='#009900', thres=0.5,
                                save_to=fgt+'_datool__fg_minus_obs.png' )


    fig, ax = plt.subplots(figsize=(8,6))
    for ivt, vt in enumerate(vtypes) :
        if ivt == 0 :
            if args.calibration is None :
                ax.plot( rbc, hist_obs[vt]/hist_obs[vt].sum()/refl_binsize, 'k', label='obs' )
            else :
                ax.plot( rbc/args.calibration, args.calibration*hist_obs[vt]/hist_obs[vt].sum()/refl_binsize, '--k', label='obs/{}'.format(args.calibration) )

            if use_datool :
                ax.plot( rbc, hist_obs_datool/hist_obs_datool.sum()/refl_binsize, '#666666', label='obs datool' )
                #ax.plot( rbc/0.92, 0.92*hist_obs_datool/hist_obs_datool.sum()/refl_binsize, '--', color='#666666', label='obs datool/0.92' )

                ax.plot( rbc, hist_mdl_datool/hist_mdl_datool.sum()/refl_binsize, 'r', label='mdl datool' )

        ax.plot( rbc, hist_mdl[vt]/hist_mdl[vt].sum()/refl_binsize, label=vt )
        print('>>> #pixels:', vt, hist_obs[vt].sum(), hist_mdl[vt].sum() )

    ax.legend(title=args.experiments[0].split('/')[-1])
    ax.set_xlabel('VIS006 reflectance')
    ax.set_ylabel('PDF')
    ax.set_xlim((0,1.1))
    ax.set_title( '{}-{}, {}{}UTC, {:.1f}<lat<{:.1f}, {:.1f}<lon<{:.1f}'.format( fg_times[0], fg_times[-1], \
        args.start_daily, '-'+args.end_daily if args.end_daily!=args.start_daily else '',
        latmin, latmax, lonmin, lonmax), fontsize='small' )
    fname = 'histograms_{}-{}_{}{}UTC_{:.1f}lat{:.1f}_{:.1f}lon{:.1f}.png'.format( fg_times[0], fg_times[-1], \
        args.start_daily.replace(':',''), '-'+args.end_daily.replace(':','') if args.end_daily!=args.start_daily else '',
        latmin, latmax, lonmin, lonmax)
    fig.savefig( fname, bbox_inches='tight')


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # parse command line arguments
    parser = define_parser()
    args = parser.parse_args()
    

    if args.plot_files : # create plots for individual files        

        lat = None
        lon = None        
        for fn in args.experiments :
            refl, lat, lon = load_seviri_file( fn, lat=lat, lon=lon, datool_coords=args.datool_coords )
            plot_reflectance( refl, name='', save_to=fn.split('/')[-1]+'.png', lat=lat, lon=lon, grid=True )
    
    elif args.visop_vs_datool : # compare visop and datool results

        compare_visop_datool( args )

    elif args.cmp_visop_types or args.histograms :

        compare_visop_types( args )

    elif args.plot_experiments : # create plots for each verification time of several experiments

        # load experiments
        xps = [ BacyExp(x) for x in args.experiments ]

        if args.compare : # comparison plots -- all experiments in each plot

            if args.forecasts :
                plot_forecasts_error( xps, args )

                #for xp in xps :
                #    print( '--', xp.exp_dir, xp.fc_start_times, xp.fc_lead_times )
            else :
                plot_experiments( xps, args )

        else : # individual plots for each experiment

            for xp in xps :

                # create / set output directory                    
                args.output_path = os.path.join(xp.exp_dir,'seviri')
                print('OUTPUT PATH ', args.output_path)
                if not os.path.exists(args.output_path) :
                    os.makedirs(args.output_path)

                # generate plots
                plot_experiments( [xp], args )
