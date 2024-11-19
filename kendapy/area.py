#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y / A R E A . P Y
#  area definitions and tests
#
#  2017.7 L.Scheck

from __future__ import absolute_import, division, print_function
import numpy as np
from kendapy.cosmo_grid import rot_to_nonrot_grid, nonrot_to_rot_grid, cosmo_grid

def parse_area_definition( areadef, kw=None, verbose=False ) :

    rect  = None
    units = None

    for token in areadef.split('_') :
        k, v = token.split(':')
        if k == 'LATLON' :
            if v[0].isdigit() or v[0] == '-' or v[0]=='.' :
                rect = list(map( float, v.split(',') ))
            else :
                rect = None
                latlon_name = v
        if k == 'LATLONUNITS' :
            units = v

    if rect is None :
        if latlon_name.startswith('COSMO-DE') :

            # COSMO-DE domain rotated lats and lons
            latr, lonr = cosmo_grid(configuration='COSMO-DE', rotated=True, onedim=True )

            # area affected drawn to lateral boundary conditions
            latmarg = 0.75
            lonmarg = 0.75

            if latlon_name == 'COSMO-DE' : # the full region
                rect = ( latr.min(), lonr.min(), latr.max(), lonr.max() ) # rotated COSMO-DE grid ranges

            elif latlon_name == 'COSMO-DE-CORE' : # everything except the boundary condition affected margins
                rect = ( latr.min()+latmarg, lonr.min()+lonmarg, latr.max()-latmarg, lonr.max()-lonmarg )

            elif latlon_name == 'COSMO-DE-NOALPS' : # like the previous, minus the alps
                rect = (               -1.5, lonr.min()+lonmarg, latr.max()-latmarg, lonr.max()-lonmarg )

            elif latlon_name == 'COSMO-DE-ADAPT' : # like the previous, minus a superobbing-dependent additional margin
                if len(kw['SUPEROBB']) == 1 :
                    soflon, soflat = kw['SUPEROBB'][0], kw['SUPEROBB'][0]
                else :
                    soflon, soflat = kw['SUPEROBB'][0], kw['SUPEROBB'][1]

                #           __________km__________ * deg/km
                solatmarg = 0.5 * (soflat-1) * 6.0 * (360.0/(2*np.pi*6371.0))
                solonmarg = 0.5 * (soflon-1) * 3.0 * (360.0/(2*np.pi*6371.0*np.cos(50.0*np.pi/180.0)))
                print(('adding margins to compensate for superobbing: ', solatmarg, solonmarg))
                rect = (               -2.5+solatmarg, lonr.min()+lonmarg+solonmarg, (latr.max()-latmarg)-solatmarg, (lonr.max()-lonmarg)-solonmarg )
            else :
                raise ValueError('unknown latlon region name '+latlon_name)

            units = 'DEG_ROT'
            if verbose : print(('LATLON=', latlon_name,' -> rot. coord. rect. ', rect))

        else :
            raise ValueError('I do not understand the aera definition '+latlon_name)

    if units is None :
        units = 'DEG'

    return rect, units

def is_in_area( lat, lon, areadef, input_units='DEG', vens=None ) :

    dbg=False

    rect, units = parse_area_definition(areadef)

    if dbg : print(('IS_IN_AREA ', rect, units))

    if units == input_units :
        x = lat
        y = lon

    elif input_units == 'DEG' and units == 'DEG_ROT' :
        x, y = nonrot_to_rot_grid( lat, lon )

    elif input_units == 'DEG' and units == 'PX' :
        if vens is None :
            raise ValueError('transformation to/from SEVIRI pixel coordinates requires visop ensemble to be specified')
        x, y = vens.coordinates_to_indices( lon, lat )

    else :
        print(('input_units = ', input_units))
        print(('area_units  = ', units))
        raise ValueError('Coordinate transformation not yet implemented.')

    inside = np.zeros(lat.shape,dtype=bool)
    inside[ np.where( (x>=rect[0]) & (x<=rect[2]) & (y>=rect[1]) & (y<=rect[3]) ) ] = True

    if dbg :
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(7,7))
        ax  = fig.add_subplot(1, 1, 1)
        cols = ['r' if x else 'b' for x in inside]
        ax.scatter( lon, lat, color=cols )
        fig.savefig( "area_dbg.png", bbox_inches='tight' )

        print(("IS_IN_AREA : %d of %d obs are inside..." % (np.count_nonzero(inside),inside.size)))

    return inside

def convert_to_latlon( y, x, input_units='DEG', vens=None ) :

    if input_units == 'DEG' :
        return y, x
    elif input_units == 'DEG_ROT' :
        return rot_to_nonrot_grid( y, x )
    else :
        raise ValueError('Implement me!')
        #elif input_units == 'PX' :
#-------------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------------
#-------------------------------------------------------------------------------------

    from matplotlib import pyplot as plt

    areas=['LATLON:COSMO-DE','LATLON:COSMO-DE-CORE','LATLON:COSMO-DE-NOALPS']
    lat, lon = cosmo_grid(configuration='COSMO-DE', rotated=False, onedim=False )
    for area in areas :
        rect, units = parse_area_definition( area )
        print((area, ' : ', rect, units))
        plt.figure(1)
        plt.clf()
        plt.scatter( lon, lat, alpha=0.1, linewidths=0 )
        inside = np.where(is_in_area( lat, lon, area, input_units='DEG' ))
        plt.scatter( lon[inside], lat[inside], c='r', linewidths=0 )

        n = 10
        yy = list(np.linspace(rect[0],rect[2],n)) + list(np.zeros(n)+rect[2]) + list(np.linspace(rect[0],rect[2],n)[::-1]) + list(np.zeros(n)+rect[0])
        xx = list(np.zeros(n)+rect[1]) + list(np.linspace(rect[1],rect[3],n)) + list(np.zeros(n)+rect[3]) + list(np.linspace(rect[1],rect[3],n)[::-1])
        print(('xx ', xx))
        print(('yy ', yy))
        print(('units', units))
        llat, llon = convert_to_latlon( yy, xx, input_units=units )
        plt.plot( llon, llat, 'k', linewidth=3 )

        plt.xlabel('LON')
        plt.ylabel('LAT')
        plt.title(area)
        plt.savefig(area.replace(':','_')+'.png')

