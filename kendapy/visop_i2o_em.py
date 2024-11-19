#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . I 2 O _ E M
#  compute filtered/averaged/transformed observations + model equivalents
#  from synthetic and real satellite images
#
#  2017.5 L.Scheck 

from __future__ import absolute_import, division, print_function
import sys, getpass, subprocess, argparse, time, re, gc, socket, os
#from geometry import cosmo_indices
import numpy as np
import numpy.ma as ma
from kendapy.cosmo_grid import nonrot_to_rot_grid, cosmo_grid

#-------------------------------------------------------------------------------
def superobb( obs, meq, lat, lon, fac=None, verbose=False ) :
    """Apply superobbing operator"""

    import skimage.measure

    f = fac
    if f is None :
        fx, fy, = 1, 1
    elif len(f) == 2 :
        fx, fy = f
    else :
        fx, fy = f[0], f[0]

    if fx==1 and fy==1 :
        return obs, meq, lat, lon

    else :
        nx, ny = obs.shape
        # crop fields such that the dimensions are divisible by fx,fy
        cx, cy = (nx//fx)*fx, (ny//fy)*fy
        if verbose : print('superobb: factor %d x %d --> cropping from %d x %d to %d x %d' % (fx,fy,nx,ny,cx,cy), end=' ')

        robs, rmeq, rlat, rlon = skimage.measure.block_reduce( obs[:cx,:cy], (fx,fy), func=np.mean ), \
                                 skimage.measure.block_reduce( meq[:cx,:cy], (fx,fy), func=np.mean ), \
                                 skimage.measure.block_reduce( lat[:cx,:cy], (fx,fy), func=np.mean ), \
                                 skimage.measure.block_reduce( lon[:cx,:cy], (fx,fy), func=np.mean )

        # pixel invalid -> mask=True => compute maximum when coarsening mask        
        robs_ = ma.array( robs, mask=skimage.measure.block_reduce( obs.mask[:cx,:cy], (fx,fy), func=np.max ) )

        if verbose : print((' --> resulting field size : ', robs_.shape))
        return robs_, rmeq, rlat, rlon

#-------------------------------------------------------------------------------
def thin( obs, meq, lat, lon, fac=None, verbose=False ) :
    """Apply thinning operator"""

    f = fac
    ox, oy = 0, 0            # default: no offset
    if f is None :
        fx, fy = 1, 1        # no thinning
    elif len(f) == 1 :
        fx, fy = f[0], f[0]  # same thinning factor in both dimensions
    elif len(f) == 2 :
        fx, fy = f           # different thinning factors
    elif len(f) == 4 :
        fx, fy, ox, oy = f   # different thinning factors + offsets
    else :
        raise ValueError('thin: fac must have 0,1,2 or 4 elements')

    if fx==1 and fy==1 :
        return obs, meq, lat, lon

    else :
        robs, rmeq, rlat, rlon = obs[ ox::fx, oy::fy ],\
                                 meq[ ox::fx, oy::fy ],\
                                 lat[ ox::fx, oy::fy ],\
                                 lon[ ox::fx, oy::fy ]
        if verbose : print(('thinning: factor %d x %d --> reducing original size %d x %d to %d x %d' \
                                    % (fx,fy,obs.shape[0],obs.shape[1],robs.shape[0],robs.shape[1])))
        return robs, rmeq, rlat, rlon

#-------------------------------------------------------------------------------
def restrict( obs, meq, lat, lon, latlon, units='DEG', verbose=False ) :
    """Apply area restriction operator"""

    if latlon is None :
        return obs.ravel(), meq.ravel(), lat.ravel(), lon.ravel()

    else :
        if units == 'DEG' : # restrict to lat-lon rectangle
            latmin, lonmin, latmax, lonmax = latlon
            idcs = np.where( (lat>=latmin) & (lat<=latmax) & (lon>=lonmin) & (lon<=lonmax) )
            if verbose : print(('restricting area --> reducing number of observations from %d to %d...' % (obs.size,len(idcs[0]))))
            return obs[idcs], meq[idcs], lat[idcs], lon[idcs]

        elif units == 'DEGROT' :  # restrict to lat-lon rectangle in rotated grid
            # transform rotate grid rectangle to indices in non-rotated grid
            #rotlat, rotlon = latlon_nonrot_to_rot( lat, lon )
            rotlat, rotlon = nonrot_to_rot_grid( lat, lon )

            latmin, lonmin, latmax, lonmax = latlon
            idcs = np.where( (rotlat>=latmin) & (rotlat<=latmax) & (rotlon>=lonmin) & (rotlon<=lonmax) )
            if verbose : print(('restricting area --> reducing number of observations from %d to %d...' % (obs.size,len(idcs[0]))))
            return obs[idcs], meq[idcs], lat[idcs], lon[idcs]

        elif units == 'PX' : # restrict to SEVIRI pixel rectangle
            # PLEASE NOTE: This is supposed to happen before superobbing and thinning!
            latmin, lonmin, latmax, lonmax = list(map( int, latlon ))
            # return 2d arrays, not lists
            return ( obs[latmin:latmax+1,lonmin:lonmax+1],
                     meq[latmin:latmax+1,lonmin:lonmax+1],
                     lat[latmin:latmax+1,lonmin:lonmax+1],
                     lon[latmin:latmax+1,lonmin:lonmax+1] )

#-------------------------------------------------------------------------------
def area_settings( settings ) :
    """Extract area restriction part from settings string"""

    area = ''
    for t in settings.split('_') :
        k, v = t.split(':')
        if k.startswith('LATLON') :
            if area != '' :
                area += '_'
            area += t
    return area

#-------------------------------------------------------------------------------
def parse_settings( settings ) :
    """Convert visop_i2o settings string to settings dictionary"""

    tokens = settings.split('_')
    latlon_name = ''

    kw = {}
    latlon_name = ''
    for t in tokens :
        #print('trying to split ', t)
        k, v = t.split(':')
        #print 'token ', t, '->', k, ',', v
        if   k == 'SUPEROBB'    : kw[k]  = list(map(int,v.split(',')))
        elif k == 'THINNING'    : kw[k]  = v #map(int,v.split(',')) -- will be done in transform_variables
        elif k == 'ERROR'       : kw[k]  = v
        elif k == 'HLOC'        : kw[k]  = float(v)
        elif k == 'VLOC'        : kw[k]  = float(v)
        elif k == 'LATLON'      :
            if v.upper().startswith('COSMO-DE') :
                latlon_name = v.upper()
            else :
                kw[k]       = list(map(float,v.split(',')))
        elif k == 'LATLONUNITS' : kw[k] = v.upper()
        elif k == 'FUNC'        : kw[k] = v
        else : raise ValueError('Unknown image2obs parameter '+k)

    if not 'SUPEROBB'    in kw : kw['SUPEROBB']=[1]
    if not 'THINNING'    in kw : kw['THINNING']='1' # [1]
    if not 'ERROR'       in kw : kw['ERROR'] = '0.15'
    if not 'HLOC'        in kw : kw['HLOC'] = 35.0
    if not 'VLOC'        in kw : kw['VLOC'] = 10.0
    if not 'LATLONUNITS' in kw : kw['LATLONUNITS'] = 'DEG'
    if not 'FUNC'        in kw : kw['FUNC'] = ''

    # default area: COSMO-DE
    if not 'LATLON' in kw and latlon_name == '':
        print('WARNING: using default value COSMO-DE for LATLON...')
        latlon_name = 'COSMO-DE'

    # FIXME: THIS SHOULD BE MOVED TO AREA.PY
    if latlon_name.startswith('COSMO-DE') :

        # COSMO-DE domain rotated lats and lons
        latr, lonr = cosmo_grid(configuration='COSMO-DE', rotated=True, onedim=True )

        # area affected drawn to lateral boundary conditions
        latmarg = 0.75
        lonmarg = 0.75

        if latlon_name == 'COSMO-DE' : # the full region
            kw['LATLON'] = ( latr.min(), lonr.min(), latr.max(), lonr.max() ) # rotated COSMO-DE grid ranges

        elif latlon_name == 'COSMO-DE-CORE' : # everything except the boundary condition affected margins
            kw['LATLON'] = ( latr.min()+latmarg, lonr.min()+lonmarg, latr.max()-latmarg, lonr.max()-lonmarg )

        elif latlon_name == 'COSMO-DE-NOALPS' : # like the previous, minus the alps
            kw['LATLON'] = (               -1.5, lonr.min()+lonmarg, latr.max()-latmarg, lonr.max()-lonmarg )

        elif latlon_name == 'COSMO-DE-ADAPT' : # like the previous, minus a superobbing-dependent additional margin
            if len(kw['SUPEROBB']) == 1 :
                soflon, soflat = kw['SUPEROBB'][0], kw['SUPEROBB'][0]
            else :
                soflon, soflat = kw['SUPEROBB'][0], kw['SUPEROBB'][1]

            #           __________km__________ * deg/km
            solatmarg = 0.5 * (soflat-1) * 6.0 * (360.0/(2*np.pi*6371.0))
            solonmarg = 0.5 * (soflon-1) * 3.0 * (360.0/(2*np.pi*6371.0*np.cos(50.0*np.pi/180.0)))
            print(('adding margins to compensate for superobbing: ', solatmarg, solonmarg))
            kw['LATLON'] = (               -2.5+solatmarg, lonr.min()+lonmarg+solonmarg, (latr.max()-latmarg)-solatmarg, (lonr.max()-lonmarg)-solonmarg )
        else :
            raise ValueError('unknown latlon region name '+latlon_name)

        kw['LATLONUNITS'] = 'DEGROT'
        print(('LATLON=', latlon_name,' -> rot. coord. rect. ', kw['LATLON']))
    return kw

#-------------------------------------------------------------------------------
def superobb_around_points( obs, meq, lat, lon, ptlat, ptlon, fac=None, verbose=False ) :
    """Apply superobbing operator"""

    #import skimage.measure
    if verbose :
        print()
        print('>>> superobbing around points with lat = ', ptlat)
        print('                                   lot = ', ptlon)
        print('>>> input field shapes : ', obs.shape, meq.shape, lat.shape, lon.shape)

    f = fac
    if f is None :
        fx, fy, = 1, 1
    elif len(f) == 2 :
        fx, fy = f
    else :
        fx, fy = f[0], f[0]

    robs, rmeq, rlat, rlon = [], [], [], []

    for k in range(len(ptlat)) :
        # find pixel indices nearest to point coordinates
        dist = (lat-ptlat[k])**2 + (lon-ptlon[k])**2
        ipx, jpx = np.unravel_index( np.argmin( dist ), obs.shape )

        if verbose : print('>>> point %2d : %f %f %f %f' % (k,ptlat[k],lat[ipx,jpx],ptlon[k],lon[ipx,jpx]))

        # average over neighbourhood
        di1 = -fx//2
        di2 = di1 + fx
        dj1 = -fy//2
        dj2 = dj1 + fy
        robs.append( obs[ ipx+di1:ipx+di2, jpx+dj1:jpx+dj2 ].mean() )
        rmeq.append( meq[ ipx+di1:ipx+di2, jpx+dj1:jpx+dj2 ].mean() )
        rlat.append( lat[ ipx+di1:ipx+di2, jpx+dj1:jpx+dj2 ].mean() )
        rlon.append( lon[ ipx+di1:ipx+di2, jpx+dj1:jpx+dj2 ].mean() )

    return np.array(robs), np.array(rmeq), np.array(rlat), np.array(rlon)

#-------------------------------------------------------------------------------
def transform_variables( obs, meq, lat, lon, settings, verbose=False ) :
    """Transform observation, model equivalent, lat and lon fields according to the specified settings"""

    if type(settings) == dict :
        kw = settings
    else :
        kw = parse_settings(settings)

    if kw['THINNING'].startswith('latlon') : # contains list of lat|lon pairs (see --temp-locations in ekf.py)

        if verbose : print('transforming list of lat-lon pairs...')
        ll = kw['THINNING'][6:].split(',')
        ptlat = [float(x.split('/')[0]) for x in ll]
        ptlon = [float(x.split('/')[1]) for x in ll]
        p_obs, p_meq, p_lat, p_lon = restrict( *superobb_around_points( obs, meq, lat, lon, ptlat, ptlon, fac=kw['SUPEROBB']),
                                               latlon=kw['LATLON'], units=kw['LATLONUNITS'] )

    else : # thinning factors are provided

        thinfac = list(map(int,kw['THINNING'].split(',')))
        if kw['LATLONUNITS'].upper() == 'PX' :
            if verbose : print('transforming pixel image (starting with "restrict")...')
            # restrict area first (pixel indices are only available as long as obs, meq, lat, lon are still 2d arrays)
            p_obs, p_meq, p_lat, p_lon = thin( *superobb( *restrict( obs, meq, lat, lon, latlon=kw['LATLON'], units=kw['LATLONUNITS'] ),
                                                          fac=kw['SUPEROBB'] ), fac=thinfac )

        else :
            if verbose : print('transforming pixel image (last step is "restrict")...')
            # area restriction is last step (superobb, thin require 2d arrays, restrict for units!=PX creates 1d arrays)
            p_obs, p_meq, p_lat, p_lon = restrict( *thin( *superobb( obs, meq, lat, lon, fac=kw['SUPEROBB'] ), fac=thinfac ),
                                                   latlon=kw['LATLON'], units=kw['LATLONUNITS'] )

    return apply_function(p_obs,kw['FUNC']), apply_function(p_meq,kw['FUNC']), p_lat, p_lon


#-------------------------------------------------------------------------------
def apply_function( refl, func ) :
    if func == '' or func == 'identity' or func == '1' :
        return refl
    elif func.startswith('arctanh') :
        tk = list(map( float, func.split(',')[1:] ))
        return np.arctanh((refl-tk[0])/tk[1])
    else :
        raise ValueError('I do not understand the function definition '+func)


#-------------------------------------------------------------------------------
def image2obs( path, times, channels, settings=None, members=None, write_to=None, plots=False,
    superobbfac=1, thinfac=1, e_o='0.15', h_loc=30.0, v_loc=10.0, latlon=None, latlon_units='DEG',
    memdir='ens%03d', detdir='det') :
    """Read vis_operator_cosmo.py results in <path> for the specified times and channels,
       convert them to observations according to the settings string or explicitely
       given keyword parameters and write the results to a text file obsmeq.dat
       to be processed by visop_iofdbk.f90.
       If members=(first,last) is specified, perform conversion for the specified range of members.
       In this case also error models involving ensemble spread, mean etc. are allowed.
       """

    if not settings is None : # parse settings string
        #print 'visop_i2o : settings string is "%s"' % settings
        kw = parse_settings( settings )
    else :
        kw = {}
        kw['SUPEROBB']=superobbfac
        kw['THINNING']=thinfac
        kw['ERROR']=e_o
        kw['HLOC']=h_loc
        kw['VLOC']=v_loc
        kw['LATLON']=latlon
        kw['LATLONUNITS']=latlon_units
        kw['FUNC']=''

    print(('visop_i2o : ', kw))

    if members is None :
        obsmeq = []
    else :
        first_member, n_ens = members
        obsmeq = {}
        for i in range(first_member,n_ens+1) :
            obsmeq[i] = []

    print( 'members to be processed:', range(first_member,n_ens+1) )
    print( 'times   to be processed:', times )

    for channel in channels :
        for hhmm in times :
            if hhmm != '' :
                minutes = int(hhmm[0:2])*60 + int(hhmm[2:])
                hhmm_str = '.'+hhmm
                print('visop_i2o: processing time %s, corresponding to %d minutes...' % (hhmm,minutes), end=' ')
            else :
                minutes = 60
                hhmm_str = ''

            if not members is None : # process full ensemble ................................................
                
                # read ensemble
                p_meq = {}
                for i in range(first_member,n_ens+1) :
                    if i > 0 :
                        #mempath = "%s/ens%03d" %(path,i)
                        mempath = ("%s/"+memdir) %( path,i)
                    else :
                        #mempath = "%s/det" % path
                        mempath =  "%s/%s" % (path,detdir)

                    nc_fname = '%s/refl_sat_SEVIRI_%s%s.nc'  % (mempath,channel,hhmm_str)
                    nc_present = False
                    vld = None
                    if os.path.exists( nc_fname ) :
                        import xarray as xr
                        ds = xr.open_dataset(nc_fname)
                        #print( 'VARIABLES :: ', ds.variables)
                        #obs = np.array( ds.variables['observed_reflectance_'+channel][:] )
                        lat = np.transpose( np.array( ds.latitude[:] ) )
                        lon = np.transpose( np.array( ds.longitude[:] ) )
                        if i == first_member :
                            print('[nc] image: %dpx x %dpx ' % lat.shape, end=' ')
                        if 'pixel_valid' in ds.variables.keys() :
                            vld = np.transpose( np.array( ds.pixel_valid[:] ) )
                            margin = np.array( vld == 0, dtype=np.bool_ )
                            if i == first_member :
                                print(' -- valid pixels: {} of {} ({:.1f}%)'.format(vld.sum(), vld.size, 100*float(vld.sum())/vld.size ) )
                        else :
                            margin = np.zeros( lat.shape, dtype=np.bool_ )
                        obs = ma.array( np.transpose( np.array( ds.variables['observed_reflectance_' +channel][:] ) ), mask=margin )
                        meq =           np.transpose( np.array( ds.variables['synthetic_reflectance_'+channel][:] ) )
                        nc_present = True

                    if i == first_member and not nc_present :

                        if os.path.exists('%s/lat_seviri%s.%s.npy'  % (mempath,hhmm_str,channel)) : # old convention
                            lat = np.load('%s/lat_seviri%s.%s.npy'  % (mempath,hhmm_str,channel))
                            lon = np.load('%s/lon_seviri%s.%s.npy'  % (mempath,hhmm_str,channel))
                            print('[old] image: %dpx x %dpx ' % lat.shape, end=' ')
                            margin = np.zeros( lat.shape, dtype=np.bool_ )
                            obs = ma.array( np.load('%s/refl_seviri%s.%s.npy' % (mempath,hhmm_str,channel)), mask=margin )
                            new_convention = False

                        else : # new convention
                            lat = np.transpose(np.load('%s/lat_sat_SEVIRI%s.npy'  % (mempath,hhmm_str))) * 180/np.pi
                            lon = np.transpose(np.load('%s/lon_sat_SEVIRI%s.npy'  % (mempath,hhmm_str))) * 180/np.pi
                            print('[new] image: %dpx x %dpx ' % lat.shape, end=' ')
                            margin = np.zeros( lat.shape, dtype=np.bool_ )
                            obs = ma.array( np.transpose(np.load('%s/refl_sat_obs_SEVIRI_%s%s.npy' % (mempath,channel,hhmm_str))), mask=margin )
                            new_convention = True                                

                    if not nc_present :
                        # read member
                        if new_convention :
                            meq = ma.array( np.transpose(np.load('%s/refl_sat_model_SEVIRI_%s%s.npy'  % (mempath,channel,hhmm_str))), mask=margin )
                        else :
                            meq = ma.array( np.load('%s/refl_visop%s.%s.npy'  % (mempath,hhmm_str,channel)), mask=margin )

                    # transform member
                    p_obs, p_meq[i], p_lat, p_lon = transform_variables( obs, meq, lat, lon, kw )

                # compute ensemble mean
                p_meq_mean = np.zeros(p_meq[1].shape)
                for i in range(1,n_ens+1) :
                    p_meq_mean += p_meq[i]/float(n_ens)

                # compute ensemble spread
                p_meq_std = np.zeros(p_meq[1].shape)
                for i in range(1,n_ens+1) :
                    p_meq_std += (p_meq[i]-p_meq_mean)**2
                p_meq_std = np.sqrt(p_meq_std/float(n_ens-1))

                # compute  p_meqobs_mean = (B+O)/2  and  p_dist = |B-O|
                p_meqobs_mean = 0.5*(p_meq_mean+p_obs)
                p_dist        = abs(p_meq_mean - p_obs)

                # ignore double clear sky cases
                valid = np.where(                       (p_meqobs_mean >= 0.2) | (p_dist >= 0.05) )
                #valid = np.where( (p_meq_std >= 0.1) | (p_meqobs_mean >= 0.2) | (p_dist >= 0.05) )
                #valid = np.where( (p_meq_std >= 0.1) | (p_meqobs_mean >= 0.3) | (p_dist >= 0.05) )
                print('double clear sky fraction : ', 1.0 - len(valid[0])/float(p_obs.size), ' ', end=' ')

                # determine observation error
                e_o_specs = kw['ERROR'].split(',')
                e_o_0 = float(e_o)
                if len(e_o_specs) == 1 : 
                    # special case : disable double clear sky treatment
                    # (specify explicitely ERROR=constant,<value>,0 to get e_o=constant and ignore double clear sky cases)
                    e_o_model = 'constant'
                    e_o_0 = float(e_o_specs[0])
                    e_o_p = []
                    valid = np.where(p_obs > -999)
                else :
                    e_o_model = e_o_specs[0]
                    e_o_0 = float(e_o_specs[1])
                    e_o_p = list(map( float, e_o_specs[2:] ))
                print(('error model : ', e_o_model, e_o_0, e_o_p))

                if e_o_model   == 'constant' :
                    p_e_o = np.zeros(p_obs.shape) + e_o_0

                elif e_o_model   == 'step' :               # for avoiding problematic nonlinearity errors at small |B-O|
                    p_e_o = np.zeros(p_obs.shape) + e_o_0  # error = first parameter
                    valid = np.where( p_dist >= e_o_p[0] ) # discard cases with |B-O| < second parameter

                elif e_o_model   == 'nocc' :               # for avoiding clear sky cases with small |B+O|/2
                    p_e_o = np.zeros(p_obs.shape) + e_o_0  # error = first parameter
                    valid = np.where( p_meqobs_mean >= e_o_p[0] ) # discard cases with |B+O|/2 < second parameter

                elif e_o_model   == 'desnl' :
                    # Motivated by desroziers (see *e_desroz_sd* plots generated by visop_departures.py)
                    # and nonlinearity analysis (which suggests that small |O-B| cases should be discarded).
                    # - first  parameter : constant error offset
                    # - second parameter : factor for linear    dependency on |B-O|
                    # - third  parameter : factor for quadratic dependency on |B-O|
                    # - cases with |B-O| < fourth parameter are discarded
                    # The error is limited by (O+B)/2, as also suggested by Desroziers results.
                    p_e_o = np.minimum( p_meqobs_mean, np.zeros(p_obs.shape) + e_o_0 + e_o_p[0]*p_dist + e_o_p[1]*(p_dist**2) )
                    valid = np.where( p_dist >= e_o_p[2] )

                elif e_o_model   == 'relconst' :
                    p_e_o = np.minimum( p_meqobs_mean, e_o_0 )

                elif e_o_model   == 'gaussdep2' : # a*gauss(O+B) + b*(O-B)**2 + c
                    p_e_o = np.minimum( p_meqobs_mean, e_o_0*np.exp(-((p_meqobs_mean-e_o_p[0])/e_o_p[1])**2) \
                                                       + e_o_p[2]*p_dist**2 + e_o_p[3] )

                elif e_o_model   == 'dep2' :
                    p_e_o = np.minimum( p_meqobs_mean, e_o_0 + e_o_p[0]*p_dist**2 )

                elif e_o_model   == 'mz17relconst' :
                    p_e_o = np.sqrt( np.maximum( np.minimum( p_meqobs_mean, e_o_0 )**2, p_dist**2 - p_meq_std**2 ))

                elif e_o_model == 'linear' :
                    p_dist_corr   = np.maximum( 0, p_dist - p_meq_std )
                    p_e_o = e_o_0 * (1.0 + e_o_p[0]*p_dist_corr)
                    
                elif e_o_model == 'squared' :
                    p_dist_corr   = np.sqrt( np.maximum( 0, p_dist**2 - p_meq_std**2 ) )
                    p_e_o = e_o_0 * (1.0 + e_o_p[0]*p_dist_corr)
            
                elif e_o_model == 'mz17like' :                    
                    p_e_o = np.sqrt(np.maximum( e_o_0**2, p_dist**2 - p_meq_std**2 ))

                else :
                    raise ValueError('Unknown error model')

                # assemble output structure
                for i in range(first_member,n_ens+1) :
                    for k in range(len(p_obs[valid].ravel())) :
                        obsmeq[i].append({ 'time':minutes,
                                           'obs':p_obs[   valid].ravel()[k],
                                           'meq':p_meq[i][valid].ravel()[k],
                                           'e_o':p_e_o[   valid].ravel()[k],
                                           'h_loc':h_loc,
                                           'v_loc':v_loc,
                                           'lat':p_lat[   valid].ravel()[k],
                                           'lon':p_lon[   valid].ravel()[k] })

                if plots :
                    print('generating plots...')
                    #import pylab as plt
                    from matplotlib import pyplot as plt

                    dep_bins = np.linspace(-5.0,5.0,201)
                    dep_bin_centers = 0.5*(dep_bins[1:]+dep_bins[:-1])

                    #dep = np.zeros(list(p_obs.shape)+[n_ens])
                    dep = []
                    for i in range(1,n_ens+1) :
                        #dep[:,i-1] = (p_meq[i] - p_obs) / p_e_o
                        dep.append( ((p_meq[i] - p_obs) / p_e_o)[valid] )

                    dep_hist = np.histogram( dep, bins=dep_bins )[0]

                    plt.figure(1)
                    plt.semilogy( dep_bin_centers, dep_hist )
                    plt.savefig('dephist_'+channel+'_'+hhmm+'.png')

            else : # process single member ...................................................

                lat = np.load('%s/lat_seviri.%s.%s.npy'  % (path,hhmm,channel))
                lon = np.load('%s/lon_seviri.%s.%s.npy'  % (path,hhmm,channel))
                print('image: %dpx x %dpx ' % lat.shape, end=' ')

                #ic,jc = cosmo_indices(lat,lon)
                margin = np.zeros( lat.shape, dtype=np.bool_ )
                #margin[ np.where((ic<0) | (jc<0)) ] = True
                meq = ma.array( np.load('%s/refl_visop.%s.%s.npy'  % (path,hhmm,channel)), mask=margin )
                obs = ma.array( np.load('%s/refl_seviri.%s.%s.npy' % (path,hhmm,channel)), mask=margin )

                p_obs, p_meq, p_lat, p_lon = transform_variables( obs, meq, lat, lon, settings )

                print(' --> %d observations' % p_obs.size, end=' ')
                print((' -- shape ', p_obs.shape))
                print((p_lat.min(), p_lat.max(), p_lon.min(), p_lon.max()))

                e_o_spec = kw['ERROR'].split(',')
                if len(e_o_spec) == 1 :
                    eo = float(kw['ERROR'])
                else :
                    eo = float(e_o_spec[1])
                    if e_o_spec[0] != 'constant' :
                        print(('WARNING: cannot evaluate error model, using constant error ', eo))

                for i in range(len(p_obs.ravel())) :
                    #obsmeq.append({ 'time':minutes, 'obs':p_obs.ravel()[i], 'meq':p_meq.ravel()[i], 'e_o':e_o,
                    #                'h_loc':h_loc, 'v_loc':v_loc, 'lat':p_lat.ravel()[i], 'lon':p_lon.ravel()[i] })
                    obsmeq.append({ 'time':minutes, 'obs':p_obs.ravel()[i], 'meq':p_meq.ravel()[i], 'e_o':eo,
                                    'h_loc':kw['HLOC'], 'v_loc':kw['VLOC'], 'lat':p_lat.ravel()[i], 'lon':p_lon.ravel()[i] })

                if plots :
                    print('generating plots...')
                    #import pylab as plt
                    from matplotlib import pyplot as plt
                    plt.figure(1,figsize=(10,10*np.cos(p_lat.mean()*np.pi/180)))
                    plt.clf()
                    plt.gray()
                    plt.scatter( p_lon, p_lat, c=p_obs, edgecolors='face', s=0.7*kw['SUPEROBB'][0]**2 ) #, s=0.1, alpha=0.05 )
                    latc, lonc = cosmo_grid(configuration='COSMO-DE')
                    plt.plot(lonc[0,:],latc[0,:],color='r')
                    plt.plot(lonc[-1,:],latc[-1,:],color='r')
                    plt.plot(lonc[:,0],latc[:,0],color='r')
                    plt.plot(lonc[:,-1],latc[:,-1],color='r')
                    if args.point != '' :
                        plat, plon = list(map( float, args.point.split(',')))
                        plt.scatter( plon, plat, color='b', s=5.0 )
                    plt.title(str(len(p_obs))+" obs. "+settings, fontsize=8)
                    plt.grid()
                    plt.savefig( 'obsloc_%s_%s.png' % (settings,hhmm), bbox_inches='tight' )

                    if p_obs.ndim == 2 : # it is a potentially superobbed/thinned pixel image
                        plt.clf()
                        plt.imshow( np.transpose(p_obs), vmin=0, vmax=1.0, origin='lower', interpolation='nearest' )
                        plt.colorbar(shrink=0.5)
                        plt.savefig( 'obsimg_%s_%s.png' % (settings,hhmm), bbox_inches='tight' )
                        plt.clf()
                        plt.imshow( np.transpose(p_meq), vmin=0, vmax=1.0, origin='lower', interpolation='nearest' )
                        plt.colorbar(shrink=0.5)
                        plt.savefig( 'meqimg_%s_%s.png' % (settings,hhmm), bbox_inches='tight' )

            # end of 'process single member' ...................................................

    if not write_to is None :

        if members is None : # single file
            fnames  = [ "%s/%s" % (path,write_to) ]
            obsmeqs = [ obsmeq ]
        else :
            fnames  = [ (("%s/"+memdir+"/%s") % (path,i,write_to)).replace('ens000','det').replace('.000','') \
                                  for i in range(first_member,n_ens+1) ]
            obsmeqs = [ obsmeq[i] for i in range(first_member,n_ens+1) ]

        for i in range(len(obsmeqs)) :
            obsm  = obsmeqs[i]
            fname = fnames[i]

            # masked=True == invalid
            n_valid = np.array([ not np.ma.is_masked(obsm[k]['obs']) for k in range(len(obsm)) ]).sum()

            #print(('visop_i2o: writing %d observations to %s ...' % ( len(obsm), fname )))
            print(('visop_i2o: writing %d observations to %s ...' % ( n_valid, fname )))
            #n_written = 0
            with open(fname, 'w') as f :
                f.write( "%d\n" % n_valid )
                for k in range(len(obsm)) :
                    om = obsm[k]
                    if not np.ma.is_masked(om['obs']) :
                        f.write( "%4d %5.3f %5.3f %5.3f %5.1f %6.2f %5.2f %5.2f\n" % \
                            (om['time'], om['obs'],om['meq'],om['e_o'],om['h_loc'],om['v_loc'],om['lat'],om['lon']))
                        #n_written += 1
            #print('written obs: ', n_written)

            if plots :
                idcs_vld = [ k for k in range(len(obsm)) if not np.ma.is_masked(obsm[k]['obs']) ]
                idcs_ivd = [ k for k in range(len(obsm)) if     np.ma.is_masked(obsm[k]['obs']) ]
                fig, ax = plt.subplots()
                ax.scatter( [obsm[k]['lon'] for k in idcs_vld], [obsm[k]['lat'] for k in idcs_vld], c='b', marker='.', s=1 )
                ax.scatter( [obsm[k]['lon'] for k in idcs_ivd], [obsm[k]['lat'] for k in idcs_ivd], c='r', marker='.', s=1 )
                ax.set_title('{} observations'.format(n_valid))
                ax.set_xlabel('LON')
                ax.set_ylabel('LAT')
                fig.savefig(fname+'.png', bbox_inches='tight')
                plt.close(fig)

        print('visop_i2o: done')

    return obsmeq

#-------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description='Convert real + synthetic sat images to observations + model equivalents')
    
    parser.add_argument(      '--latlon-units',  dest='latlon_units', default='DEG',     help='units for latlon [DEG,DEGROT,PX]')
    parser.add_argument('-L', '--latlon',        dest='latlon',       default='',        help='restrict to rectangle <latmin>,<lonmin>,<latmax>,<lonmax>')
    parser.add_argument('-S', '--superobb',      dest='superobb',     default='3',       help='superobbing factor[s in longitude and latitude], e.g. "2,4"')
    parser.add_argument('-T', '--thinning',      dest='thinning',     default='1',       help='thinning factor[s in longitude and latitude]')
    parser.add_argument('-e', '--error',         dest='e_o',          default='0.15',    help='observation error')
    parser.add_argument('-H', '--h-loc',         dest='h_loc',        default='30.0',    help='horizontal localization radius')
    parser.add_argument('-V', '--v-loc',         dest='v_loc',        default='0.3',     help='vertical localization radius')
    # the options above may also be provided by means of a settings string
    parser.add_argument('-s', '--settings',      dest='settings',     default='',        help='settings string from which options will be parsed')
    parser.add_argument('-m', '--members',       dest='members',      default='',        help='<first member>,<laster member> to be processed')

    parser.add_argument(    '--memdir',       dest='memdir',      default='ens%03d',     help='name of member directories')
    parser.add_argument(    '--detdir',       dest='detdir',      default='det',         help='name of deterministic member directory')

    parser.add_argument('-c', '--channel',       dest='channel',  default='VIS006',  help='comma-separated list of SEVIRI channels (default: VIS006)')
    parser.add_argument('-t', '--times',         dest='times',    default='',        help='comma-separated list of output times, e.g. 0015,0030,0045')
    parser.add_argument('-w', '--write-to',      dest='write_to', default='obsmeq.dat', help='name of observation file to be written')
    parser.add_argument('-p', '--plots',         dest='plots',    help='generate some diagnostic plots', action='store_true' )
    parser.add_argument(      '--test',          dest='test',     help='run tests', action='store_true' )
    parser.add_argument(      '--point',         dest='point',    help='<lat>,<lon> of a point to be marked in plots', default='' )
    parser.add_argument( 'paths', metavar='paths', help='paths to visop results', nargs='*' )

    return parser

#-------------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------------
#-------------------------------------------------------------------------------------

    parser = define_parser()
    args = parser.parse_args()

    if args.plots :
        from matplotlib import pyplot as plt

    for path in args.paths :

        if args.settings == '' : # old method : use separate command line options for all settings

            if args.latlon == '' :
                latlon=None
            else :
                latlon = tuple(map(float,args.latlon.split(',')))
                if len(latlon) != 4 : raise ValueError('latlon argument must contain 4 comma-separated values <latmin>,<lonmin>,<latmax>,<lonmax>')
                print(('latlon = ', latlon))

            image2obs( path, args.times.split(','), args.channel.split(','),
                       write_to=args.write_to,
                       superobbfac=list(map(int,args.superobb.split(','))),
                       thinfac=list(map(int,args.thinning.split(','))),
                       e_o=float(args.e_o), h_loc=float(args.h_loc), v_loc=float(args.v_loc),
                       latlon=latlon, latlon_units=args.latlon_units )

        else : # new method : all settings are contained in one string

            if args.test :
                obsmeq = image2obs( path, args.times.split(','), args.channel.split(','), settings=args.settings, write_to=None, plots=True )
                sys.exit(0)

            if args.settings.startswith('file:') : # should we read the settings from a file?
                sfname = args.settings.replace('file:','')
                print('visop_i2o_em: reading settings from file {}...'.format(sfname))
                with open(sfname,'r') as f :
                    lines = f.readlines()
                settings = '_'.join([ l.strip() for l in lines if not (l.startswith('#') or l.strip() == '') ])
            else :
                settings = args.settings

            if args.members == '' :
                image2obs( path, args.times.split(','), args.channel.split(','), settings=settings,
                           memdir=args.memdir, detdir=args.detdir, write_to=args.write_to, plots=args.plots )
            else :
                image2obs( path, args.times.split(','), args.channel.split(','), members=list(map(int,args.members.split(','))), settings=settings,
                           memdir=args.memdir, detdir=args.detdir, write_to=args.write_to, plots=args.plots )
