#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P R E C P R O D _ R E A D
#  reads radar precipitation products
#
#  2018.1 L.Scheck

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.ma as ma
import os, sys, argparse

from kendapy.time14 import Time14

#-----------------------------------------------------------------------------------------------------------------------
def read_product( t, tresmin=60, product='EY', cutoff=100.0 ) :
    """Read precipitation product and return average precipitation rate in [t,t+tresmin]"""

    if product == 'EY' :
        res = read_EY_product( t, tresmin=tresmin, cutoff=cutoff )
    else :
        raise ValueError('Only EY is implemented')

    return res

#-----------------------------------------------------------------------------------------------------------------------
def read_EY_product( t, tresmin=60, cutoff=100, verbose=False ) :
    """Read EY product and return average precipitation rate in [t,t+tresmin]"""

    try:
        import gribapi as grib_api
    except ImportError:
        try:
            import grib_api
        except ImportError:
            print('Found neither gribapi (versions <= 1.15) nor grib_api (versions >= 1.16)')
            sys.exit(-1)

    prodpath = os.getenv('HOME')+'/radar/EY_gribs/'
    t14 = Time14(t)
    t8 = t14.string(format='$y$M$D$h')
    fname = prodpath + t8 + '.grib1'

    if verbose : print('opening %s ...' % fname)

    f = open( fname,'r')

    records = []

    record_index = 0
    while 1:
            gid = grib_api.grib_new_from_file(f)
            if gid is None : break

            Ni = grib_api.grib_get_long(gid,"Ni")
            Nj = grib_api.grib_get_long(gid,"Nj")

            vtime = grib_api.grib_get_string(gid,'validityTime')
            vdate = grib_api.grib_get_string(gid,'validityDate')

            sn = grib_api.grib_get_string(gid,'shortName')

            mv = grib_api.grib_get(gid,'missingValue')

            if verbose : print(( 'reading %s (%d x %d) valid for %s / %s ...' % (sn,Nj,Ni,vdate,vtime) ))
            records.append( grib_api.grib_get_values(gid).reshape((Nj,Ni)) )
            #print records[-1].mean()

            record_index += 1

    if verbose : print('mean before masking ', np.stack(records).mean())
    tp = ma.masked_greater( np.stack(records), np.minimum( cutoff, mv-1) )
    if verbose : print('mean, max after masking ', tp.mean(), tp.max())

    if tresmin > 0 :

        # average over the indices coressponding to [t,t+tresmin]
        ifirst = t14.minute()//5
        ilast  = ifirst + tresmin//5 - 1

        if ilast >= 12 :
            raise ValueError('time + resolution must be smaller than 60 minutes!')

        return tp[ifirst:ilast+1,:,:].mean(axis=0)

    else :
        # tresmin == 0 --> return the full file
        return tp

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": # -------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='read radar precipitation product')
    parser.add_argument( '-P', '--product',    dest='product',    help='EY|EW|RW', default='EY' )
    parser.add_argument( '-r', '--resolution', dest='resolution', help='time resolution in minutes', type=int, default=60 )
    parser.add_argument( '-c', '--cutoff',     dest='cutoff',     help='mask precipitation rates larger than this', type=float, default=100.0 )
    parser.add_argument( '-p', '--plot',       dest='plot',       help='plot statistics', action='store_true' )
    parser.add_argument( '-v', '--verbose',    dest='verbose',    help='be more verbose', action='store_true' )
    parser.add_argument( 'times', metavar='times', help='times', nargs='*' )
    args = parser.parse_args()

    for t in args.times :
        p = read_product( t, product=args.product, tresmin=args.resolution, cutoff=args.cutoff )
        print(('time ', t, ' : min/mean/max precipitation rate = ', p.min(), p.mean(), p.max()))

        if args.plot :
            from matplotlib import pyplot as plt

            plt.figure(1)
            plt.imshow( p, origin='lower' )
            plt.colorbar()
            plt.title( '%s %s %smin [mm/h]' % ( args.product, t, args.resolution ) )
            plt.savefig( '%s_%s_%smin.png' % ( args.product, t, args.resolution ), bbox_inches='tight' )

            plt.clf()
            tpbins = np.arange(1,31,1)
            tphist, tpedges = np.histogram( p, tpbins )
            plt.plot( tpbins[:-1], tphist/float(p.count()) )
            plt.title( '%s %s %smin [mm/h]' % ( args.product, t, args.resolution ) )
            plt.savefig( '%s_%s_%smin_hist.png' % ( args.product, t, args.resolution ), bbox_inches='tight' )
