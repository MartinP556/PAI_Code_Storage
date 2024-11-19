#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P R E C P R O D _ E N S
#  read precipitation product ensemble + observation
#
#  2018.1 L.Scheck

from __future__ import absolute_import, division, print_function
from numpy import *
import numpy.ma as ma
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.precprod_read import read_product
from kendapy.cosmo_state import cosmo_grid
from score_fss import fss_dict, fss_ens_dict, fss_random_target

#-----------------------------------------------------------------------------------------------------------------------
def get_precprod_ens( xp_, start_time, output_time, tresmin=None, lfcst=False, verbose=False, cache=None, copy_mask=True, clear_mask=True ) :
    """Retrieve precipitation ensemble and observation
    
       start_time : Time14
       output_time : in minutes
       tresmin : time resolution in minutes

       Will return the precipitation in the time window [ start_time+output_time, start_time+output_time+tresmin ]
    """

    xp = Experiment(xp_) if isinstance(xp_, str) else xp_

    if tresmin is None :
        if lfcst :
            tresmin = xp.lfcst_sfc_output_times_min[1] - xp.lfcst_sfc_output_times_min[0]
        else :
            tresmin = xp.sfc_output_times_min[1] - xp.sfc_output_times_min[0]

    if verbose :
        print('get_precprod_ens : %s fcst start time = %s, output time = %dmin, time resolution is %d minutes...' % \
              ('lfcst' if lfcst else 'cycling', start_time, output_time, tresmin))

    # read precipitation observations (mean precip rate for t in [output_time,output_time+tresmin] in mm/h)
    tabs = str( Time14(start_time) + Time14(output_time*60) )
    tp_obs = read_product( tabs, tresmin=tresmin, product='EY', cutoff=100.0 )
    obs_mask = tp_obs.mask

    # read accumulated precipitation at begin and end of time window
    tp_start = ma.masked_array( get_accumulated_precprod_ens( xp, start_time, output_time,           lfcst=lfcst, cache=cache ) )
    tp_end   = ma.masked_array( get_accumulated_precprod_ens( xp, start_time, output_time + tresmin, lfcst=lfcst, cache=cache ) )

    # de-accumulate and convert to precipitation rate with units [mm/h]
    tp = (tp_end-tp_start) / (tresmin/60.0)

    if copy_mask : # invalid cells in observations will be also marked invalid in the model eqivalents
        tp.mask = tp_obs.mask

    if clear_mask : # all invalid cells will be set to zero and marked as valid
        idcs = where(tp.mask == True)
        tp[idcs] = 0
        idcs = where(tp_obs.mask == True)
        tp_obs[idcs] = 0

        tp_obs.mask = False
        tp.mask = False

    #cs = xp.get_cosmo( output_time, lfcst_time = start_time if lfcst else None, prefix='lfff', suffix='sfc', member=1 )
    lat, lon = cosmo_grid()

    return {'ens':tp, 'obs':tp_obs, 'lat':lat, 'lon':lon, 'mask':obs_mask}

#-----------------------------------------------------------------------------------------------------------------------
def get_accumulated_precprod_ens( xp_, start_time, output_time, lfcst=False, verbose=True, cache=None ) :
    """Retrieve accumulated precipitation ensemble at start_time+output_time"""

    xp = Experiment(xp_) if isinstance(xp_, str) else xp_

    id = '%s_%s_%d' % (xp.settings['EXPID'],start_time, output_time)

    if cache is None :
        cache = {}

    if not id in cache :
        tp_mem = []
        for m in range(xp.n_ens) :
            #print('get_accumulated_precprod_ens ', output_time, start_time, lfcst, 'lfff', 'sfc', m+1 )
            cs = xp.get_cosmo( start_time, output_time=output_time, lfcst=lfcst, prefix='lfff', suffix='sfc', member=m+1 )
            try :
                tp_mem.append( cs['TP'] )
            except :
                tp_mem.append( cs['TOT_PREC'] ) # name for older grib_api/table versions

        cache[id] = stack(tp_mem)

    return cache[id]

#-----------------------------------------------------------------------------------------------------------------------
def get_precprod_member( xp_, start_time, output_time, member=0, tresmin=None, lfcst=False, verbose=False, cache=None, copy_mask=True, clear_mask=True ) :
    """Retrieve precipitation ensemble and observation

       start_time : Time14
       output_time : in minutes
       tresmin : time resolution in minutes

       Will return the precipitation in the time window [ start_time+output_time, start_time+output_time+tresmin ]
    """

    xp = Experiment(xp_) if isinstance(xp_, str) else xp_

    if tresmin is None :
        if lfcst :
            tresmin = xp.lfcst_sfc_output_times_min[1] - xp.lfcst_sfc_output_times_min[0]
        else :
            tresmin = xp.sfc_output_times_min[1] - xp.sfc_output_times_min[0]

    if verbose :
        print('get_precprod_member : %s fcst start time = %s, output time = %dmin, time resolution is %d minutes...' % \
              ('lfcst' if lfcst else 'cycling', start_time, output_time, tresmin))

    # read precipitation observations (mean precip rate for t in [output_time,output_time+tresmin] in mm/h)
    tabs = str( Time14(start_time) + Time14(output_time*60) )
    tp_obs = read_product( tabs, tresmin=tresmin, product='EY', cutoff=100.0 )
    obs_mask = tp_obs.mask

    # read accumulated precipitation at begin and end of time window
    tp_start = ma.masked_array( get_accumulated_precprod_member( xp, start_time, output_time,           member=member, lfcst=lfcst, cache=cache ) )
    tp_end   = ma.masked_array( get_accumulated_precprod_member( xp, start_time, output_time + tresmin, member=member, lfcst=lfcst, cache=cache ) )

    # de-accumulate and convert to precipitation rate with units [mm/h]
    tp = (tp_end-tp_start) / (tresmin/60.0)

    if copy_mask : # invalid cells in observations will be also marked invalid in the model eqivalents
        tp.mask = tp_obs.mask

    if clear_mask : # all invalid cells will be set to zero and marked as valid
        idcs = where(tp.mask == True)
        tp[idcs] = 0
        idcs = where(tp_obs.mask == True)
        tp_obs[idcs] = 0

        tp_obs.mask = False
        tp.mask = False

    #cs = xp.get_cosmo( output_time, lfcst_time = start_time if lfcst else None, prefix='lfff', suffix='sfc', member=1 )
    lat, lon = cosmo_grid()

    return {'model':tp, 'obs':tp_obs, 'lat':lat, 'lon':lon, 'mask':obs_mask}



#-----------------------------------------------------------------------------------------------------------------------
def get_accumulated_precprod_member( xp_, start_time, output_time, member=0, lfcst=False, verbose=True, cache=None ) :
    """Retrieve accumulated precipitation for a certain member at start_time+output_time"""

    xp = Experiment(xp_) if isinstance(xp_, str) else xp_

    id = '%s_%s_%d_%d' % (xp.settings['EXPID'],start_time, output_time, member)

    if cache is None :
        cache = {}

    if not id in cache :
        cs = xp.get_cosmo( start_time, output_time=output_time, lfcst=lfcst, prefix='lfff', suffix='sfc', member=member )
        cache[id] = cs['TOT_PREC']

    return cache[id]

#-------------------------------------------------------------------------------
def get_fss( ppens, thres, windows, believable_scale=False, target=False, member=-1 ) : # fractions skill score and decomposition

    if member < 0 : # full ensemble
        fss = fss_ens_dict( ppens['ens'], ppens['obs'], windows, thres, believable_scale=believable_scale, target=target )

    else : # one specific member
        print("ppens['ens'].shape = ", ppens['ens'].shape)
        fss = fss_dict( ppens['ens'][member,...], ppens['obs'], windows, thres, believable_scale=believable_scale, target=target )

    return fss

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    import sys
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from copy import copy
    from kendapy.cosmo_state import cosmo_grid

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate precipitation')

    parser.add_argument( '-d', '--day',       dest='day',   help='day, e.g. 0605 (default)', default='0605' )
    parser.add_argument( '-t', '--threshold', dest='level', help='precip threshold[s]',      default='1.0' )
    parser.add_argument( '-s', '--scale',     dest='win',   help='scale[s]',                 default='11' )

    parser.add_argument( '-X', '--experiment', dest='experiment', help='path to experiment log file', default='' )
    parser.add_argument( '-S', '--start-time', dest='start_time', help='fcst start time, e.g. 201606051200' )
    parser.add_argument( '-o', '--output-time', dest='output_time', help='output time [minutes, relative to start time]', type=int, default=0 )

    parser.add_argument( '-f', '--fss',     dest='plot_fss',     help='plot fss evolution', action='store_true' )
    parser.add_argument(       '--mean-only',  dest='mean_only',  help='plot only mean, not all members', action='store_true' )
    parser.add_argument( '-F', '--fss-dwd', dest='plot_fss_dwd', help='reproduce dwd fss plots', action='store_true' )
    parser.add_argument( '-m', '--member',  dest='member',       help='member index (default: -1 -> use full ensemble)', type=int, default=-1 )
    parser.add_argument( '-B', '--belscl',  dest='plot_belscl',  help='plot believable scale evolution', action='store_true' )
    parser.add_argument(       '--belscl-min', dest='belscl_min',  help='mininum believable scale to be plotted', type=float, default=0 )
    parser.add_argument(       '--belscl-max', dest='belscl_max',  help='mininum believable scale to be plotted', type=float, default=200 )
    parser.add_argument(       '--belscl-log', dest='belscl_log',  help='generate logarithmic plot of believable scale (default: linear)', action='store_true' )
    parser.add_argument(       '--colors',     dest='colors',      help='comma-separated list of colors for the experiments', default='k,b,r,y,g' )
    parser.add_argument( '-i', '--image-type', dest='image_type',  help='image file type [default:png]', default='png' )

    parser.add_argument( '-I', '--ids', dest='ids', help='comma-separated list of experiment ids', default='101,104' )
    parser.add_argument( '-N', '--names', dest='names', help='comma-separated list of experiment names', default='' )
    parser.add_argument( '-w', '--windows', dest='windows', help='comma-separated list of window sizes [in units of COSMO-DE grid cells]', default='5,11,25,51,101' )
    parser.add_argument( '-l', '--levels',  dest='levels',  help='comma-separated list of threshold values', default='0.1,1,5' )
    parser.add_argument( '-r', '--resolution',  dest='tresmin',       help='time resolution in minutes (default 60)', type=int, default=60 )
    args = parser.parse_args()

    if args.names != '' :
        names = args.names.split(',')
    else :
        names = args.ids.split(',')

    if args.plot_fss :
        import glob
        ids = args.ids.split(',')
        windows = [ int(w) for w in args.windows.split(',') ]
        levels  = [ float(l) for l in args.levels.split(',') ]
        hours = r_[6,9,12,15]
        #minutes = r_[60,120,180]
        tresmin = args.tresmin
        minutes = arange(tresmin,181,tresmin)
        print('plot_fss: ', ids, windows, levels, hours, minutes)

        fss = dict()
        for id in ids :
            day = id[:4]
            fss[id] = dict()
            for h in hours :
                fss[id][h] = dict()
                for t in minutes :
                    logfile = glob.glob('/project/meteo/work/Leonhard.Scheck/kenda_experiments/cosmo_letkf/settings/{}/run_cycle_*.log'.format(id))[0]
                    print('logfile = ', logfile)
                    xp = Experiment(logfile)
                    pp = get_precprod_ens( xp, '2016{}{:02d}0000'.format(day,h), t-tresmin, tresmin=tresmin, lfcst=True, verbose=True )
                    fss[id][h][t] = get_fss( pp, levels, windows, member=args.member-1, target=True )
                    print(id, h, t, fss[id][h][t]['fss'])

        from matplotlib import pyplot as plt
        #cols = ['k', 'b', 'r',  'y',  'g' ]
        cols = args.colors.split(',')
        fig, ax = plt.subplots(figsize=(4,3)) # 10,3
        for iid, id in enumerate(ids) :
            for l, lev in enumerate(levels) :
                for w, win in enumerate(windows) :
                    for h in hours :
                        tim = h + minutes/60.0
                        if not args.mean_only :
                            for m in range(40) :
                                mem_fss = [ fss[id][h][t]['members'][m]['fss'][l,w] for t in minutes ]
                                ax.plot( tim, mem_fss, color=cols[iid], alpha=0.1 )
                        mean_fss = [ fss[id][h][t]['fss'][l,w] for t in minutes ]
                        ax.plot( tim, mean_fss, color=cols[iid], linewidth=2,
                                 label='{}'.format(names[iid],lev,win*2.8) if h==hours[0] else None )
                                 #label='{}  {:.1f}mm/h, {:.0f}km'.format(names[iid],lev,win*2.8) if h==hours[0] else None )
                        if iid == 0 :
                            ax.plot( tim, [ fss[id][h][t]['fss_target'][l] for t in minutes ], '--k', alpha=0.5 )
                            ax.plot( tim, [ fss[id][h][t]['fss_random'][l] for t in minutes ], ':k', alpha=0.5 )
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim((0,1))
        ax.set_xlim((6,19))
        ax.set_title('day: 2016{}, scale: {} cells, threshold: {}mm'.format(day,win,lev) + ', member: {}'.format(args.member) if args.member >= 0 else '')
        ax.set_xlabel('time')
        ax.set_ylabel('fss')
        if args.member > 0 :
            mem = '_member{}'.format(args.member)
        else :
            if args.mean_only :
                mem = ''
            else :
                mem = '_allmem'
        levstr = '_'.join([ '{:.1f}'.format(l) for l in levels ])
        winstr = '_'.join([ str(w)             for w in windows])
        fig.savefig('fss__{}__threshold{}mm_scale{}{}.{}'.format('_'.join(ids),levstr,winstr,mem,args.image_type), bbox_inches='tight')
        plt.close(fig)


    if args.plot_fss_dwd :
        cols = ['k',  'r',  'y',  'g',  'b']
        ids  = ['101','102','103','104','111']

        day = args.day          #sys.argv[1] #'20160605'
        lev = float(args.level) #float(sys.argv[2]) # 5.0
        win = int(args.win)     # int(sys.argv[3]) #25 #101
        hours = r_[6,9,12,15]
        minutes = r_[60,120,180]
        tresmin = 60
        fss = dict()
        fss_target = dict()
        for id in ids :
            fss[id] = dict()
            for h in hours :
                fss[id][h] = dict()
                fss_target[h] = dict()
                for t in minutes :
                    xp = Experiment('/project/meteo/work/Leonhard.Scheck/kenda_experiments/cosmo_letkf/settings/{}.{}/run_cycle_{}.{}.log'.format(day,id,day,id))
                    pp = get_precprod_ens( xp, '2016{}{:02d}0000'.format(day,h), t-tresmin, tresmin=tresmin, lfcst=True, verbose=True )
                    r = get_fss( pp, [lev], [win], member=args.member-1, target=True )
                    fss[id][h][t] = r['fss'].ravel()[0]
                    fss_target[h][t] = r['fss_target'].ravel()[0]
                    print(id, h, t, fss[id][h][t], fss_target[h][t])

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(10,3))
        for iid, id in enumerate(ids) :
            for h in hours :
                tim = h + minutes/60.0
                ax.plot( tim, [ fss[id][h][t] for t in minutes ],
                         '.' if iid==0 else '-',
                         color=cols[iid], label=day+'.'+id if h==hours[0] else None )
                if iid == 0 :
                    ax.plot( tim, [ fss_target[h][t] for t in minutes ], '--k', alpha=0.5 )

        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim((0,1))
        ax.set_xlim((6,19))
        ax.set_title('day: 2016{}, scale: {} cells, threshold: {}mm'.format(day,win,lev) + ', member: {}'.format(args.member) if args.member >= 0 else '')
        ax.set_xlabel('time')
        ax.set_ylabel('fss')
        if args.member > 0 :
            mem = '_member{}'.format(args.member)
        else :
            mem = ''
        fig.savefig('fss_dwd__{}__threshold{:.1f}mm_scale{}{}.{}'.format(day,lev,win,mem,args.image_type), bbox_inches='tight')
        plt.close(fig)

    if args.plot_belscl :
        cols = ['k',  'r']
        ids  = ['101','104']
        nms  = ['conv. only','conv.+VIS']

        day = args.day          #sys.argv[1] #'20160605'
        lev = float(args.level) #float(sys.argv[2]) # 5.0
        hours = r_[6,9,12,15]
        #minutes = r_[60,120,180]
        #tresmin = 60
        tresmin = args.tresmin
        minutes = arange(tresmin,181,tresmin)
        wins = [2,5,11,17,25,51,75,101,125,151,201]
        belscl = dict()
        for id in ids :
            belscl[id] = dict()
            for h in hours :
                belscl[id][h] = dict()
                for t in minutes :
                    print('>>> processing ', id, h, t)
                    xp = Experiment('/project/meteo/work/Leonhard.Scheck/kenda_experiments/cosmo_letkf/settings/{}.{}/run_cycle_{}.{}.log'.format(day,id,day,id))
                    print('--- getting precprod ensemble...')
                    pp = get_precprod_ens( xp, '2016{}{:02d}0000'.format(day,h), t-tresmin, tresmin=tresmin, lfcst=True, verbose=True )
                    print('--- computing belscl...')
                    belscl[id][h][t] = get_fss( pp, [lev], wins, believable_scale=True, member=args.member-1 )['belscl'].ravel()[0]
                    print('===', id, h, t, belscl[id][h][t])

        print('>>> plotting results...')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(10,3))
        ax.set_title('day: 2016{}, threshold: {}mm'.format(day,lev) + ', member: {}'.format(args.member) if args.member >= 0 else '')

        for iid, id in enumerate(ids) :
            for h in hours :
                tim = h + minutes/60.0
                bs = array([ belscl[id][h][t]*2.8 for t in minutes ])
                bs[where(bs <= 0 )] = 1000.0
                if args.belscl_log :
                    ax.semilogy( tim, bs, color=cols[iid], label=nms[iid] if h==hours[0] else None )
                else :
                    ax.plot( tim, bs, color=cols[iid], label=nms[iid] if h==hours[0] else None )

        ax.legend(frameon=False, fontsize=9, title='threshold: {:.1f}mm'.format(lev), loc='lower right')
        if args.belscl_log and args.belscl_min == 0 :
            ax.set_ylim((args.belscl_max/1e2,args.belscl_max))
        else :
            ax.set_ylim((args.belscl_min,args.belscl_max))
        ax.set_xlim((6,19))
        ax.set_xlabel('time')
        ax.set_ylabel('believable scale [km]')
        ax.grid()
        if args.member > 0 :
            mem = '_member{}'.format(args.member)
        else :
            mem = ''
        fig.savefig('belscl_{}_threshold{:.1f}mm{}{}_dt{}min_max{:.0f}km.{}'.format(day,lev,mem,
                    '_log' if args.belscl_log else '_lin', tresmin, args.belscl_max, args.image_type),
                    bbox_inches='tight')
        plt.close(fig)

        if False : # save data for idealised-vs-operational plot
            import pickle
            pfname = 'belscl_{}_threshold{:.1f}mm_dt{}min.pickle'.format(day,lev,tresmin)
            with open( pfname,'w') as f :
                pickle.dump( {'belscl_cells':belscl, 'starttime_hours':hours, 't_min':minutes}, f, pickle.HIGHEST_PROTOCOL )

    if args.experiment != '' :
        # plot prcipitation fields for given experiment and time

        xp = Experiment( args.experiment )
        pp = get_precprod_ens( xp, args.start_time, args.output_time, tresmin=60, lfcst=True, verbose=True )
        for k in pp :
            print(k, pp[k].shape, pp[k].mean())

        lat, lon = cosmo_grid()

        tp = pp['ens']
        tp_mean = tp.mean(axis=0)
        tp_obs = pp['obs']

        tp_cmap = copy(plt.get_cmap('ocean_r'))
        tp_cmap.set_bad( '#eeeeee', 1.0 )
        tp_cmap.set_over( '#00ff00', 1.0 )

        fig, ax = plt.subplots( 2, 2, figsize=(12,12) ) #sharey=True,

        im = ax[0,0].imshow( tp[0,:,:], origin='lower',  vmin=0, vmax=20, cmap=tp_cmap )
        divider = make_axes_locatable(ax[0,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, extend='max')
        ax[0,0].contour( lat, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        ax[0,0].contour( lon, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        #ax[0,0].plot( [imin,imax,imax,imin,imin], [jmin,jmin,jmax,jmax,jmin], 'b' )
        cs = ax[0,0].contour( lat, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[0,0].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        cs = ax[0,0].contour( lon, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[0,0].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        ax[0,0].text( 10, 10, 'mem. 0, mean=%f, max=%f' % (tp[0,:,:].mean(),tp[0,:,:].max()) )

        im = ax[1,0].imshow( tp_obs, origin='lower', vmin=0, vmax=20, cmap=tp_cmap )
        divider = make_axes_locatable(ax[1,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, extend='max')
        ax[1,0].contour( lat, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        ax[1,0].contour( lon, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        #ax[1,0].plot( [imin,imax,imax,imin,imin], [jmin,jmin,jmax,jmax,jmin], 'b' )
        cs = ax[1,0].contour( lat, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[1,0].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        cs = ax[1,0].contour( lon, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[1,0].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        ax[1,0].text( 10, 10, 'obs, mean=%f, max=%f' % (tp_obs.mean(),tp_obs.max()) )

        im = ax[0,1].imshow( tp_mean, origin='lower',  vmin=0, vmax=20, cmap=tp_cmap )
        divider = make_axes_locatable(ax[0,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, extend='max')
        ax[0,1].contour( lat, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        ax[0,1].contour( lon, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        #ax[0,1].plot( [imin,imax,imax,imin,imin], [jmin,jmin,jmax,jmax,jmin], 'b' )
        ax[0,1].contour( tp_obs, levels=[1.0], colors='r', linewidths=0.5 )
        cs = ax[0,1].contour( lat, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[0,1].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        cs = ax[0,1].contour( lon, levels=arange(0,90,5), colors='k', linewidths=0.5 )
        ax[0,1].clabel( cs, cs.levels, fmt="%4.0f", inline=True, fontsize=10 )
        ax[0,1].text( 10, 10, 'ens. mean, mean=%f, max=%f' % (tp_mean.mean(),tp_mean.max()) )

        #plt.colorbar()
        tpbins = arange(0.5,20.1,0.5)

        for m in range(1,xp.n_ens) :
            tp_mem_hist, tpedges = histogram( tp[m,:,:], tpbins )
            ax[1,1].semilogy( tpbins[:-1], tp_mem_hist, '#cccccc' )

        tp_mean_hist, tpedges = histogram( tp_mean, tpbins )
        ax[1,1].semilogy( tpbins[:-1], tp_mean_hist, 'r', label='ens. mean' )

        tp_mem0_hist, tpedges = histogram( tp[0,:,:], tpbins )
        ax[1,1].semilogy( tpbins[:-1], tp_mem0_hist, 'b', label='mem. 0' )

        tp_obs_hist, tpedges = histogram( tp_obs, tpbins )
        ax[1,1].semilogy( tpbins[:-1], tp_obs_hist,  'k', label='obs' )

        ax[1,1].set_ylabel('#columns')
        ax[1,1].set_xlabel('total precipitation [mm]')
        ax[1,1].legend(title="precprod %s %s %03d" % ( xp.settings['EXPID'], args.start_time, args.output_time ), frameon=False)
        plt.savefig( "precprod_%s_%s_%03dmin.{}" % ( xp.settings['EXPID'], args.start_time, args.output_time, args.image_type ), bbox_inches='tight' )
        plt.close(fig)

        # compute fractions skill score & believable scale
        #fss = get_fss( pp, [0.1, 5.0], [11,25,51,101], believable_scale=True )


        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow( pp['ens'].mask[0,:,:], origin='lower',  vmin=-1, vmax=2, cmap='jet' )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, extend='max')
        ax.contour( lat, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        ax.contour( lon, levels=arange(0,90,1), colors='#cccccc', linewidths=0.3 )
        fig.savefig('mask.png')
        plt.close(fig)

        levs = r_[0.1,1,5]
        scls = r_[5,11,25,51,101]
        fss = get_fss( pp, levs, scls, believable_scale=True )

        print('fss shape ', fss['fss'].shape)

        print('level   : ', end=' ')
        for l, lev in enumerate(levs) :
            print('{:.3f}'.format(levs[l]), end=' ')
        print()
        for s, scl in enumerate(scls) :
            print('scl={:3d} : '.format(scl), end=' ')
            for l, lev in enumerate(levs) :
                print('{:.3f}'.format(fss['fss'][l,s]), end=' ')
            print()
        #print "fss={:.2f}, fss_target={:.2f}, belscl={:.0f}".format(fss['fss'], fss['fss_target'], fss['belscl'])





