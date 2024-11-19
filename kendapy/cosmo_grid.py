#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . C O S M O _ G R I D
#  COSMO model grid related routines
#
#  2018.8 L.Scheck

from __future__ import absolute_import, division, print_function
from numpy import *

#-----------------------------------------------------------------------------------------------------------------------
def cosmo_grid( configuration='COSMO-DE', rotated=False, onedim=False, area=False, definition=False ) :
    """Returns latitude and longitude arrays (2d), cell areas or grid definition for known model configurations."""

    # print 'cosmo_grid was called with parameter ', configuration

    if isinstance( configuration, str ) :

        if configuration == 'COSMO-DE' :

            # meta information in grib files of this type
            meta = { 'longitudeOfSouthernPoleInDegrees'  : 10.0,
                     'latitudeOfSouthernPoleInDegrees'   :-40.0,
                     'latitudeOfFirstGridPointInDegrees' : -5.0,
                     'latitudeOfLastGridPointInDegrees'  :  6.5,
                     'longitudeOfFirstGridPointInDegrees': -5.0,
                     'longitudeOfLastGridPointInDegrees' :  5.5,
                     'iDirectionIncrementInDegrees'      : 0.025,
                     'jDirectionIncrementInDegrees'      : 0.025,
                     'nlat':461,
                     'nlon':421 }

        elif configuration == 'COSMO-D2' or configuration == 'ICON-D2' :

            # meta information in grib files of this type
            meta = { 'longitudeOfSouthernPoleInDegrees'  : 10.0,
                     'latitudeOfSouthernPoleInDegrees'   :-40.0,
                     'latitudeOfFirstGridPointInDegrees' : -6.3,
                     'latitudeOfLastGridPointInDegrees'  :  8.0,
                     'longitudeOfFirstGridPointInDegrees': 352.5,
                     'longitudeOfLastGridPointInDegrees' :  5.5,
                     'iDirectionIncrementInDegrees'      : 0.02,
                     'jDirectionIncrementInDegrees'      : 0.02,
                     'nlat':716,
                     'nlon':651 }

        else :
            raise ValueError('unknown model configuration '+configuration)

    else : # configuration = dictionary containing grid definition parameters

        meta = configuration

    if definition :
        return meta

    # construct rotated coordinate arrays
    latfi = float(meta['latitudeOfFirstGridPointInDegrees'])
    latla = float(meta['latitudeOfLastGridPointInDegrees'])
    lonfi = float(meta['longitudeOfFirstGridPointInDegrees'])
    lonla = float(meta['longitudeOfLastGridPointInDegrees'])

    if lonfi > 180.0 : lonfi -= 360.0
    if lonla > 180.0 : lonla -= 360.0

    lat_rc1d = linspace( latfi, latla, int(meta['nlat']) )
    lon_rc1d = linspace( lonfi, lonla, int(meta['nlon']) )
    lat_rc, lon_rc = meshgrid( lat_rc1d, lon_rc1d, indexing='ij' )

    if rotated :
        if onedim :
            lat, lon = lat_rc1d, lon_rc1d
        else :
            lat, lon = lat_rc, lon_rc
    else :
        # convert to standard coordinates
        lat, lon = rot_to_nonrot_grid( lat_rc, lon_rc,
                                       pollon=float(meta['longitudeOfSouthernPoleInDegrees']) - 180,
                                       pollat=-float(meta['latitudeOfSouthernPoleInDegrees']) )
    if area :
        R_e = 6371e3
        dx = 2*pi*R_e*(lat_rc1d[1]-lat_rc1d[0])/360.0
        return dx*dx*cos(lat_rc*pi/180)

    else :
        return lat, lon


#-----------------------------------------------------------------------------------------------------------------------
def nonrot_to_rot_grid( lat, lon, pollat=40.0, pollon=-170.0, configuration=None, verbose=False ) :
    """Convert non-rotated coordinates to rotated grid coordinates -- see COSMO documentation part I, Eq. (3.74)"""

    if not configuration is None :
        # get pollat, pollon from grid definition
        gdef = cosmo_grid( configuration=configuration, definition=True )
        pollat, pollon = -gdef['latitudeOfSouthernPoleInDegrees'], gdef['longitudeOfSouthernPoleInDegrees']-180
        if verbose : print('using pollat={}, pollon={}'.format(pollat,pollon))

    if verbose : print('nonrot. grid : ', lat.min(), ' <    lat < ', lat.max(), '  ,  ', lon.min(), ' <    lon < ', lon.max())

    sin_lat    = sin(deg2rad(lat))
    sin_pollat = sin(deg2rad(pollat))
    cos_lat    = cos(deg2rad(lat))
    cos_pollat = cos(deg2rad(pollat))
    cos_dlon   = cos(deg2rad(lon-pollon))
    sin_dlon   = sin(deg2rad(lon-pollon))

    rotlat = rad2deg( arcsin( sin_lat*sin_pollat + cos_lat*cos_pollat * cos_dlon  ) )
    rotlon = rad2deg( arctan( cos_lat*sin_dlon / ( cos_lat*sin_pollat*cos_dlon - sin_lat*cos_pollat ) ) )

    if verbose : print('rotated grid : ', rotlat.min(), ' < rotlat < ', rotlat.max(), '  ,  ', rotlon.min(), ' < rotlon < ', rotlon.max())

    return rotlat, rotlon


#-----------------------------------------------------------------------------------------------------------------------
def rot_to_nonrot_grid( rotlat, rotlon, pollat=40.0, pollon=-170.0, configuration=None, verbose=False ) :
    """Convert rotated coordinates to non-rotated grid coordinates -- see COSMO documentation part I, Eq. (3.74)"""

    if not configuration is None :
        # get pollat, pollon from grid definition
        gdef = cosmo_grid( configuration=configuration, definition=True )
        pollat, pollon = -gdef['latitudeOfSouthernPoleInDegrees'], gdef['longitudeOfSouthernPoleInDegrees']-180
        if verbose : print('using pollat={}, pollon={}'.format(pollat,pollon))

    if verbose : print('rotated grid : ', rotlat.min(), ' < rotlat < ', rotlat.max(), '  ,  ', rotlon.min(), ' < rotlon < ', rotlon.max())

    sin_lat    = sin(deg2rad(rotlat))
    cos_lat    = cos(deg2rad(rotlat))

    sin_lon    = sin(deg2rad(rotlon))
    cos_lon    = cos(deg2rad(rotlon))

    sin_pollat = sin(deg2rad(pollat))
    cos_pollat = cos(deg2rad(pollat))

    sin_pollon = sin(deg2rad(pollon))
    cos_pollon = cos(deg2rad(pollon))

    lat = rad2deg( arcsin( sin_lat*sin_pollat + cos_lat*cos_lon * cos_pollat  ) )

    # Last line of Eq. (3.74) seems to be wrong...
    #lon = rad2deg( arctan( cos_lat*sin_lon / ( cos_lat*sin_pollat*cos_lon - sin_lat*cos_pollat ) ) ) + pollon
    #lon = rad2deg( arctan2( cos_lat*sin_lon , ( cos_lat*sin_pollat*cos_lon - sin_lat*cos_pollat ) ) ) + pollon

    # backtransformation from cosmo/src/utilities.f90 works:
    zarg1   = sin_pollon * (-sin_pollat * cos_lon * cos_lat  + cos_pollat * sin_lat) -  cos_pollon * sin_lon * cos_lat
    zarg2   = cos_pollon * (-sin_pollat * cos_lon * cos_lat  + cos_pollat * sin_lat) +  sin_pollon * sin_lon * cos_lat
    lon = rad2deg( arctan2( zarg1, zarg2 ))

    if verbose : print('nonrot. grid : ', lat.min(), ' <    lat < ', lat.max(), '  ,  ', lon.min(), ' <    lon < ', lon.max())

    return lat, lon

