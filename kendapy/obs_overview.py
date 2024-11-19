#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . O B S _ O V E R V I E W
#  generate observation overview plots for KENDA experiments
#
#  2017.7 L. Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

import sys, os, argparse
from kendapy.ekf import tables as ekf_tables
from numpy import *
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.colseq import colseq
from kendapy.visop_fcst import get_visop_fcst_stats
from kendapy.visop_plot_images import plot_reflectance

def observation_overview( xp, times, args, output_path='', obstypes=None, variables=None, state_filter='active',
                          file_type='png', dpi=None, no_plots=False, r_cutoff=-1 ) :

    if obstypes is None :
        obstypes = list(ekf_tables['obstypes'].values())
    if variables is None :
        variables = list(ekf_tables['variables'].values())
    n_variables = len(variables)

    op = output_path + ('/' if (output_path != '' and not output_path.endswith('/')) else '' )

    # gather data
    ekfs = {}
    n_obs = {}
    for i_time, time in enumerate(times) :
        for i_obstype, obstype in enumerate(obstypes) :
            if os.path.exists(xp.ekf_filename( time, obstype )) :
                if not obstype in ekfs : ekfs[obstype] = {}
                ekfs[obstype][time] = xp.get_ekf( time, obstype, state_filter=state_filter ) #, varname_filter=variables )
                n = ekfs[obstype][time].obs().size
                if not obstype in n_obs :
                    n_obs[obstype] = 0
                n_obs[obstype] += n
                print(('TIME %s : FOUND %d OBS. OF OBSTYPE %s' % ( time, n, obstype )))

    obstypes_all = obstypes + []
    #obstypes = list(ekfs.keys())  # FIXME: do we need this? It changes color<->obstype association...
    n_obstypes = len(obstypes)

    otlab = {}
    for obstype in n_obs :
        otlab[obstype] = "%s [%d]" % (obstype,n_obs[obstype])

    print( 'OBSTYPES : ', obstypes )
    print( 'COLORS   : ', args.colors.split(',') )

    if no_plots :
        return

    # time - something plots
    for yquan in ['lat','lon','plevel'] :

        fig = plt.figure(figsize=(16,6))
        ax  = fig.add_subplot(1, 1, 1)
        add_labels = {}
        ymin =  1e30
        ymax = -1e30
        for i_time, time in enumerate(times) :
            for i_obstype, obstype in enumerate(obstypes) :
                if obstype in ekfs and time in ekfs[obstype] :
                    ekf = ekfs[obstype][time]
                    if ekf.obs().size == 0 : continue

                    yvals = ekf.obs(param=yquan)
                    if yquan == 'plevel' : yvals /= 100.0
                    ymin = minimum( ymin, yvals.min() )
                    ymax = maximum( ymax, yvals.max() )

                    timeval = ekf.attr['verification_ref_time']/100.0 + ekf.obs(param='time')/60.0
                    if not obstype in add_labels :
                        add_labels[obstype] = True
                    plt.scatter( timeval+0.01*i_obstype/float(n_obstypes), yvals,
                                        color=colseq(i_obstype,n_obstypes), edgecolor='', alpha=0.7, s=10.0,
                                        label=otlab[obstype]+' ' if add_labels[obstype] else None )
                    add_labels[obstype] = False
        if yquan == 'plevel' :
            ax.set_ylim(ymax,ymin)
        ax.set_xlabel('time [h]')
        ax.set_ylabel(yquan)
        ax.set_title(xp.settings['EXPID'])
        #plt.legend(handles=leghandles,frameon=False)
        plt.legend(frameon=False)
        fig.savefig( op+'obs_overview_time-%s.%s' % (yquan,file_type), bbox_inches='tight', dpi=dpi )

    # lat-lon-plots
    if args.basemap :
        from mpl_toolkits.basemap import Basemap
        plt.clf()
        # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
        # are the lat/lon values of the lower left and upper right corners
        # of the map.
        # resolution = 'c' means use crude resolution coastlines.
        m = Basemap(projection='cyl',llcrnrlat=43.0, urcrnrlat=57,\
            llcrnrlon=0,urcrnrlon=22,resolution='l') # c=coarse, f=fine, l=low?
        ###m.fillcontinents( color='w', lake_color='#dddddd')
        #m.drawlsmask( land_color='w', ocean_color='0.9', resolution='l', grid=1.25 ) # DOES NOT WORK WITH PYTHON3...
        gr='#bbbbbb'
        m.drawcoastlines(color=gr,linewidth=2)
        m.drawcountries(linewidth=1.0, color=gr)
        m.fillcontinents(color='0.95',lake_color='#cccccc')
        # draw parallels and meridians.
        m.drawparallels( arange(45,60,5), labels=[1,0,0,1], color=gr )
        m.drawmeridians( arange(5,21,5), labels=[1,0,0,1], color=gr )
        #m.drawmapboundary(fill_color='aqua')

        if args.region != '' :
            from kendapy.area import parse_area_definition, convert_to_latlon
            rc, rcunits = parse_area_definition( args.region )
            n = 10
            yy = list(linspace(rc[0],rc[2],n)) + list(zeros(n)+rc[2]) + list(linspace(rc[0],rc[2],n)[::-1]) + list(zeros(n)+rc[0])
            xx = list(zeros(n)+rc[1]) + list(linspace(rc[1],rc[3],n)) + list(zeros(n)+rc[3]) + list(linspace(rc[1],rc[3],n)[::-1])
            llat, llon = convert_to_latlon( yy, xx, input_units=rcunits )
            xpt, ypt = m( llon, llat )
            plt.plot( xpt, ypt, 'k', linewidth=1.5, zorder=5)
            plt.text( xpt[0]+0.3, ypt[0]+0.5, args.region.replace('LATLON:',''), fontsize=15, zorder=5 )

        if not args.box is None :
            bname, bcolor, blonmin, blonmax, blatmin, blatmax = args.box.split(',')
            plt.plot( (float(blonmin),float(blonmax)), (float(blatmin),float(blatmin)), bcolor, linewidth=1.5, zorder=5 )
            plt.plot( (float(blonmin),float(blonmax)), (float(blatmax),float(blatmax)), bcolor, linewidth=1.5, zorder=5 )
            plt.plot( (float(blonmin),float(blonmin)), (float(blatmin),float(blatmax)), bcolor, linewidth=1.5, zorder=5 )
            plt.plot( (float(blonmax),float(blonmax)), (float(blatmin),float(blatmax)), bcolor, linewidth=1.5, zorder=5 )
            plt.text( float(blonmin), float(blatmin)-0.2, bname, va='top', color=bcolor, fontsize=15, zorder=5 )
    else :
        fig = plt.figure(figsize=(8,8))
        ax  = fig.add_subplot(1, 1, 1)

    add_labels = {}
    for i_time, time in enumerate(times) :
        for i_obstype, obstype in enumerate(obstypes) :
            if obstype in ekfs and time in ekfs[obstype] :
                ekf = ekfs[obstype][time]
                if not obstype in add_labels :
                    add_labels[obstype] = True
                lat = ekf.obs(param='lat')
                lon = ekf.obs(param='lon')+0.1*i_obstype/float(n_obstypes)
                if args.basemap :
                    xpt,ypt = m(lon,lat)
                else :
                    xpt, ypt = lon, lat
                if args.colors != '' :
                    col = args.colors.split(',')[i_obstype]
                else :
                    col = colseq(i_obstype,n_obstypes)
                if obstype in ['TEMP','PILOT'] :
                    s = 50.0
                elif obstype=='AIREP' :
                    s = 3.0
                elif obstype=='RAD' :
                    s = 3.0
                else :
                    s = 10.0
                plt.scatter( xpt, ypt, color=col, edgecolor='', alpha=0.7, s=s, zorder=4,
                             label=otlab[obstype]+' ' if add_labels[obstype] else None )
                #10+len(obstypes)-i_obstype,
                add_labels[obstype] = False

                if r_cutoff > 0 :
                    for i_obs in range(lat.size) :
                        lon0, lat0 = lon[i_obs], lat[i_obs]  # deg
                        r_lat = 360.0*r_cutoff/(2*pi*6371.0) # deg
                        r_lon = r_lat/cos(lat0*pi/180)       # deg
                        print( '**OBS ', i_obs, lon0, lat0, r_lon, r_lat )
                        lonc, latc = [], []
                        for a in range(0,361,5) :
                            alpha = a*pi/180.0
                            latc.append( lat0 + r_lat*cos(alpha) )
                            lonc.append( lon0 + r_lon*sin(alpha) )
                        if args.basemap :
                            xpt,ypt = m(lonc,latc)
                        else :
                            xpt, ypt = lonc, latc
                        plt.plot( xpt, ypt, color='k', alpha=0.7, zorder=4, linewidth=0.5 )

    if args.basemap :
        plt.legend( frameon=True, loc="lower right" )
        print('saving '+op+'obs_overview_lat-lon_basemap.'+file_type+' ...')
        plt.savefig( op+'obs_overview_lat-lon_basemap.'+file_type, bbox_inches='tight', dpi=dpi )
    else :
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title(xp.settings['EXPID'])
        #plt.legend(handles=leghandles,frameon=False)
        plt.legend(frameon=True)
        fig.savefig( op+'obs_overview_lat-lon.'+file_type, bbox_inches='tight', dpi=dpi )


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate observation overview plots for KENDA experiment')

    parser.add_argument(       '--no-plots',     dest='no_plots',     help='generate only text output', action='store_true' )

    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default='' )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default='' )

    parser.add_argument( '-V', '--variables',    dest='variables',    help='comma-separated list of variables  to be considered (default: all)', default='' )
    parser.add_argument( '-O', '--obs-types',    dest='obstypes',     help='comma-separated list of obs. types to be considered (default: all)', default='' )
    parser.add_argument( '-S', '--state-filter', dest='state_filter', help='use active / passive / valid / ... observations (default: active)', default='active' )

    parser.add_argument(       '--r-cutoff',     dest='r_cutoff',     help='plot circles with this raidus around observations', type=float, default=-1 )

    parser.add_argument( '-b', '--basemap',     dest='basemap',     help='use basemap to draw country contours ect.',   action='store_true' )
    parser.add_argument( '-r', '--region',      dest='region',      help='draw rectangle around a region known by area.py', default='LATLON:COSMO-DE' )
    parser.add_argument( '-B', '--box',         dest='box',         help='plot a lat-lon box <name>,<color>,<lonmin>,<lonmax>,<latmin>,<latmax>',  default=None )

    parser.add_argument( '-c', '--colors',      dest='colors',      help='comma-separated list of colors', default='' )

    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='auto' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument(       '--dpi',         dest='dpi',         help='image resolution in dpi [default: 100]', type=int, default=100 )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    # process all log files

    if args.obstypes != '' :
        obstypes = args.obstypes.split(',')
    else :
        obstypes = list(ekf_tables['obstypes'].values())
    if args.variables != '' :
        variables = args.variables.split(',')
    else :
        variables = list(ekf_tables['varnames'].values())

    xps = {}
    rss = {}
    for logfile in args.logfile :

        print()
        print(("processing file %s ..." % logfile))

        xp = Experiment(logfile)
        xps[logfile] = xp
        print(('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
               xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] )))
        print(('state filter : ', args.state_filter))
        print()

        # set some default values
        if args.output_path != '' :
            if args.output_path != 'auto' :
                output_path = args.output_path+'/'
            else :
                output_path = xp.settings['PLOT_DIR']+'/obs_overview/'
                if not os.path.exists(output_path) :
                    os.system('mkdir -p '+output_path)
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
            end_time = xp.veri_times[-1]

        times = []
        for t in xp.veri_times :
            #print t, start_time, end_time
            if (Time14(t) >= Time14(start_time)) and (Time14(t) <= Time14(end_time)) :
                times.append(t)

        observation_overview( xp, times, args, output_path=output_path, obstypes=obstypes, variables=','.join(variables),
                              state_filter=args.state_filter, file_type=args.file_type, dpi=args.dpi,
                              no_plots=args.no_plots, r_cutoff=args.r_cutoff )

