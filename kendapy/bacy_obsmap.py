#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ O B S M A P
#  Generate map with observation locations for bacy experiment
#
#  2021.3 L. Scheck

import sys, os, argparse
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import cartopy
import cartopy.crs as ccrs
import numpy as np

from kendapy.ekf        import Ekf
from kendapy.bacy_utils import t2str, str2t, add_branch, common_subset, default_color_sequence, expand_color_name, adjust_time_axis
from kendapy.bacy_exp   import BacyExp


#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="Generate map with observation locations")

    parser.add_argument( '-r', '--region',      dest='region',      help='draw rectangle around a known region [e.g. ICON-D2]', default=None )
    parser.add_argument( '-F', '--filter',     dest='filter',         default='state=active',    help='filter string (default: "state=active")' )
    parser.add_argument( '-O', '--obstypes',   dest='obstypes',       default=None,              help='comma-separated list of observation types, e.g. SYNOP,TEMP,RAD' )
    #parser.add_argument( '-V', '--varnames',   dest='varnames',       default=None,              help='comma-separated list of variable names, e.g. U,V,T' )

    parser.add_argument(       '--time',       dest='ana_time',     default=None,              help='time range, e.g. 20190629170000,20190630120000' )
    #parser.add_argument(       '--yrange',     dest='yrange',         default=None,              help='y-axis range <y-min>,<y-max>' )

    # output options
    #parser.add_argument( '-o', '--output-path', dest='output_path',        default=None,              help='output path' )
    #parser.add_argument( '-i', '--image-type',  dest='image_type',         default='png',             help='[ png | eps | pdf ... ]' )
    #parser.add_argument(       '--dpi',         dest='dpi',                default=100, type=int,     help='dots per inch for pixel graphics (default: 100)' )
    #parser.add_argument(       '--figsize',     dest='figsize',            default='5,4',             help='<figure width>,<figure height> [inch]' )
    #parser.add_argument(       '--colors',      dest='colors',             default=None,              help='comma-separated list of colors (e.g. "r,#ff0000,pink")' )

    parser.add_argument( '-v', '--verbose',     dest='verbose', action='store_true',  help='be extremely verbose' )

    parser.add_argument( 'experiments', metavar='<experiment path(s)>', help='path(s) to experiment(s)', nargs='*' )

    return parser


#----------------------------------------------------------------------------------------------------------------------
def plot_obsmap( xp, vargs, verbose=True ) :
    
    # determine time
    if 'ana_time' in vargs :
        ana_time = vargs['ana_time']
    else :
        ana_time = xp.valid_times['ekf'][0]

    if ('obstypes' in vargs) and (not vargs['obstypes'] is None) and ( vargs['obstypes'] != '' ) :
        obstypes = vargs['obstypes'].split(',')
    else :
        obstypes = xp.obs_types

    if 'filter' in vargs :
        ekf_filter = vargs['filter']
    else :
        ekf_filter = None

    if verbose :
        print('  [plot_obs_evo] analysis time:     ', ana_time )
        print('                 observation types: ', obstypes )

    # gather data
    ekfs = {}
    n_obs = {}

    for i_obstype, obstype in enumerate(obstypes) :
        ekf_filename = xp.get_filename('ekf', time_valid=ana_time, obs_type=obstype)
        if os.path.exists(ekf_filename) :
            ekf = Ekf( ekf_filename, filter=ekf_filter )
            if ekf.obs().size == 0 :
                print('no obs left in {}...'.format(ekf_filename))
            else :
                print('found {} obs in {}...'.format( ekf.obs().size, ekf_filename ))
                ekfs[obstype] = ekf

    obstypes = [ o for o in obstypes if o in list(ekfs.keys()) ]
    n_obstypes = len(obstypes)
    if verbose :
        print('                 obstypes left after filtering: ', obstypes)


    #cartopy.crs.PlateCarree( central_longitude=0.0, globe=None)
    #cartopy.crs.Geostationary(central_longitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None)

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    #plt.contourf(lons, lats, sst, 60,  transform=ccrs.PlateCarree())

    ax.coastlines( edgecolor='#666666', linewidth=0.5 )
    ax.add_feature(cartopy.feature.OCEAN, facecolor='#e9e9e9')

    if not vargs['region'] is None :
        from kendapy.cosmo_grid import cosmo_grid
        grdlat, grdlon = cosmo_grid( configuration=vargs['region'] )
        print('>>> plotting region ', vargs['region'], grdlat.shape, grdlat.mean() )
        ax.plot( grdlon[0,:], grdlat[0,:], 'r' )
        ax.plot( grdlon[-1,:], grdlat[-1,:], 'r' )
        ax.plot( grdlon[:,0], grdlat[:,0], 'r' )
        ax.plot( grdlon[:,-1], grdlat[:,-1], 'r' )
        #plt.plot( xpt, ypt, 'k', linewidth=1.5, zorder=5)
        
        # lower left
        #plt.text( grdlon[0,0]+0.3, grdlat[0,0]+0.8, vargs['region'], color='r', fontsize=15, zorder=5 )
        # upper left
        plt.text( grdlon[-1,0]+0.8, grdlat[-1,0]-0.3, vargs['region'], color='r', fontsize=15, zorder=5,va='top' )

    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    colnam = {'AIREP':'deepskyblue',
              'RAD':'limegreen',
              'RADAR':'gray',
              'DRIBU':'navy',
              'PILOT':'darkred',
              'TEMP':'teal',
              'SYNOP':'magenta'}
    for c in colnam :
        print( c, expand_color_name(colnam[c]) )

    #col = { k:v for k,v in colnam }
    col = { k:expand_color_name(v) for (k,v) in colnam.items() }

    for obstype in obstypes :
        n_obs = ekfs[obstype].n_obs()        
        i_hdr = ekfs[obstype].reports()
        n_reps = len(i_hdr)
        lon, lat = ekfs[obstype].data['lon'][i_hdr], ekfs[obstype].data['lat'][i_hdr]
        print('>>> plotting ', obstype, n_obs, n_reps)
        #lon, lat = ekfs[obstype].obs(param='lon'), ekfs[obstype].obs(param='lat')
        if lon.size > 10000 :
            s = 0.2
        elif lon.size > 1000 :
            s = 1
        elif lon.size > 100 :
            s = 3
        else :
            s = 10
        if obstype == 'RADAR' :
            s = 100
        ax.scatter( lon, lat, s=s, color=col[obstype], alpha=1.0, label='{} [{}]'.format(obstype,n_obs) )

    #ax.legend(loc='lower right')
    ax.legend(title=ana_time[:4]+'/'+ana_time[4:6]+'/'+ana_time[6:8]+', '+ana_time[8:10]+'UTC',
              loc='upper left', bbox_to_anchor=(1., 0., 0.3, 1.0), frameon=False, fontsize='large')
    fig.savefig('map.pdf', bbox_inches='tight')


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # parse command line arguments
    parser = define_parser()
    args = parser.parse_args()
    
    # convert argparse object to dictionary
    vargs = vars(args)

    for xp_path in args.experiments :
        print()
        print('processing {}...'.format(xp_path))

        xp = BacyExp( xp_path )
        plot_obsmap( xp, vargs )

