#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P L O T _ D E P A R T U R E S
#  analyze departures (non-specialized version of visop_departures.py)
#
#  2018.4 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#from mpl_toolkits.basemap import Basemap, cm
import matplotlib.gridspec as gridspec
from kendapy.colseq import colseq

import sys, os, argparse, copy
from numpy import *
from scipy.optimize import curve_fit

from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.binplot import binplot
import kendapy.ekf

#-----------------------------------------------------------------------------------------------------------------------
def get_departure_statistics( xp, t, obstype, varname, use_fg_spread=False, dep_max=None, val_min=None, val_max=None,
                              n_dep_bins=31, n_ndep_bins=51, n_val_bins = 31 ) :

    # read ekf file
    print("trying to read", xp.ekf_filename( t, obstype ))
    ekf = xp.get_ekf( t, obstype, state_filter='active', varname_filter=varname )

    # compute departures and their standard deviations
    fgdep = ekf.departures( 'fgens', normalize=False )
    fgdep_std = fgdep.std(axis=0,ddof=1)
    anadep = ekf.departures( 'anaens', normalize=False )
    anadep_std = anadep.std(axis=0,ddof=1)

    depmax = dep_max if not dep_max is None else maximum( abs(fgdep).max(),   abs(anadep).max() )
    valmin = val_min if not val_min is None else minimum( ekf.fgmean().min(), ekf.anamean().min() )
    valmax = val_max if not val_max is None else maximum( ekf.fgmean().max(), ekf.anamean().max() )

    dep_bins = linspace(-depmax,depmax,n_dep_bins)
    dep_bin_centers = 0.5*(dep_bins[1:]+dep_bins[:-1])
    delta_dep = dep_bins[1] - dep_bins[0]

    ndep_bins = linspace(-3,3,n_ndep_bins)
    ndep_bin_centers = 0.5*(ndep_bins[1:]+ndep_bins[:-1])
    delta_ndep = ndep_bins[1] - ndep_bins[0]


    opbh_bins = linspace( valmin, valmax, n_val_bins )

    print('departure axis : ', dep_bins[0], dep_bins[-1])
    print('value     axis : ', opbh_bins[0], opbh_bins[-1])

    # compute desroziers estimate for e_o
    e2 = (fgdep*anadep).mean(axis=0)
    if e2.min() < 0 :
        idcs = where(e2 < 0)
        n_inv = len(idcs[0])
        fr_inv = float(n_inv)/e2.size
        print('WARNING: %d negative numbers (%.1f%%) encountered in desroziers calculation -> set to zero!' % (n_inv,fr_inv))
        e2[ where( e2 < 0 ) ] = 0
    e_desroz = sqrt( e2 )

    print('mean desroziers error ', e_desroz[where(e_desroz>0)].mean())

    # determine assumed observation error
    e_o = ekf.obs(param='e_o')

    # compute departure histograms .....................................................................................

    # normalize fg departures by expectation value
    fgdepn = fgdep / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.fgspread().reshape((1,e_o.size))**2 )
    print('mean            fg departure : ', abs(fgdep).mean())
    print('mean normalized fg departure : ', abs(fgdepn).mean())

    if use_fg_spread : # normalize with e_o**2 + fg_spread**2
        anadepn    = anadep    / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.fgspread().reshape((1,e_o.size))**2 )
    else : # normalize with e_o**2 + ana_spread**2
        anadepn    = anadep / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.anaspread().reshape((1,e_o.size))**2 )

    # departure histogram
    fgdepn_hist = histogram( fgdepn, bins=ndep_bins )[0]
    anadepn_hist = histogram( anadepn, bins=ndep_bins )[0]
    n_ens, n_full = fgdepn.shape

    res = { 'dep_bins':dep_bins, 'dep_bin_centers':dep_bin_centers, 'delta_dep':delta_dep,
            'ndep_bins':ndep_bins, 'ndep_bin_centers':ndep_bin_centers, 'delta_ndep':delta_ndep,
            'fgdepn_hist':fgdepn_hist, 'anadepn_hist':anadepn_hist, 'n_ens':n_ens, 'n_full':n_full }

    # compute std(FG departures), mean spread and desroziers estimates as functions of 0.5*(O+B) .......................

    opbh_bins = linspace( valmin, valmax, n_val_bins )
    opbh_bin_centers = 0.5*(opbh_bins[1:]+opbh_bins[:-1])

    opbh   = 0.5*( ekf.fgmean() + ekf.obs() )
    omba =         ekf.fgmean() - ekf.obs()

    n_opbh             = zeros(opbh_bins.size-1, dtype=int)

    fgmeandep_std_opbh = zeros(opbh_bins.size-1)
    meanfgspread_opbh  = zeros(opbh_bins.size-1)
    e_o_opbh           = zeros(opbh_bins.size-1)
    e_desroz_opbh      = zeros(opbh_bins.size-1)

    for i in range(opbh_bins.size-1) :
        idcs = where( (opbh >= opbh_bins[i]) & (opbh < opbh_bins[i+1]) )
        n_opbh[i] = len(idcs[0])
        if n_opbh[i] > 0 :
            fgmeandep_std_opbh[i] = omba[          idcs].std()  # standard deviation of ensemble mean departures
            meanfgspread_opbh[i]  = ekf.fgspread()[idcs].mean() # mean ensemble standard deviation
            e_o_opbh[i]           = e_o[           idcs].mean()
            e_desroz_opbh[i]      = e_desroz[      idcs].mean()

    res.update({'opbh_bins':opbh_bins, 'opbh_bin_centers':opbh_bin_centers, 'fgmeandep_std_opbh':fgmeandep_std_opbh,
                'meanfgspread_opbh':meanfgspread_opbh, 'n_opbh':n_opbh, 'n_opbh_tot':n_opbh.sum(), 'n_opbh_max':ekf.n_body,
                'e_o_opbh':e_o_opbh, 'e_desroz_opbh':e_desroz_opbh })

    # compute (O+B)/2 histograms for fg, ana
    obs_opbh_hist       = histogram( ekf.obs(),    bins=opbh_bins )[0]/1.0
    fgens_opbh_hist     = histogram( ekf.fgens(),  bins=opbh_bins )[0]/float(xp.n_ens)
    anaens_opbh_hist    = histogram( ekf.anaens(), bins=opbh_bins )[0]/float(xp.n_ens)
    res.update({'obs_opbh_hist':obs_opbh_hist, 'fgens_opbh_hist':fgens_opbh_hist, 'anaens_opbh_hist':anaens_opbh_hist })

    # compute everything as functions of (O+B)/2 and O-B ...............................................................

    omba_bins = linspace( -depmax, depmax, n_dep_bins )
    omba_bin_centers = 0.5*(omba_bins[1:]+omba_bins[:-1])

    n_symdif = zeros((opbh_bins.size-1,omba_bins.size-1), dtype=int)

    fgmeandep_std_sd  = zeros((opbh_bins.size-1,omba_bins.size-1))
    meanfgspread_sd   = zeros((opbh_bins.size-1,omba_bins.size-1))
    meaneo_sd         = zeros((opbh_bins.size-1,omba_bins.size-1))
    e_desroz_sd       = zeros((opbh_bins.size-1,omba_bins.size-1))
    std_e_desroz_sd   = zeros((opbh_bins.size-1,omba_bins.size-1))

    for i in range(opbh_bins.size-1) :
        for j in range(omba_bins.size-1) :
            idcs = where(   (opbh >= opbh_bins[i]) & (opbh < opbh_bins[i+1]) \
                          & (omba >= omba_bins[j]) & (omba < omba_bins[j+1]) )
            n_symdif[i,j] = len(idcs[0])
            if n_symdif[i,j] > 0 :
                fgmeandep_std_sd[i,j]  = omba[          idcs].std()
                meanfgspread_sd[i,j]   = ekf.fgspread()[idcs].mean()
                meaneo_sd[i,j]         = e_o[           idcs].mean()
                e_desroz_sd[i,j]       = e_desroz[      idcs].mean()
                std_e_desroz_sd[i,j]   = e_desroz[      idcs].std()

    res.update({ 'omba_bins':omba_bins, 'omba_bin_centers':omba_bin_centers,
                 'n_symdif':n_symdif, 'fgmeandep_std_sd':fgmeandep_std_sd, 'meanfgspread_sd':meanfgspread_sd,
                 'meaneo_sd':meaneo_sd, 'e_desroz_sd':e_desroz_sd, 'std_e_desroz_sd':std_e_desroz_sd })

    return res


#-----------------------------------------------------------------------------------------------------------------------
def plot_cumulative_departure_statistics( xp, obstype, varname, times=None, use_fg_spread=False, output_path='',
                                          file_type='png', dep_max=None, val_min=None, val_max=None ) :

    # retrieve individual statistics ...................................................................................
    if times is None :
        times = xp.veri_times
    print('times : ', times)

    istats = []
    times_available = []
    for time in times :
        try :
            stats = get_departure_statistics( xp, time, obstype, varname, use_fg_spread=use_fg_spread,
                                              dep_max=dep_max, val_min=val_min, val_max=val_max )
            print('got ', time)
        except IOError :
            break
        last_time = time
        istats.append( stats )
        times_available.append( time )

    # compute cumulative statistics ....................................................................................
    cstats = {}
    # constant scalars or arrays
    for q in ['dep_bins', 'dep_bin_centers', 'delta_dep',
              'ndep_bins', 'ndep_bin_centers', 'delta_ndep', 'n_ens',
              'opbh_bins', 'opbh_bin_centers', 'omba_bins', 'omba_bin_centers'] :
        cstats[q] = istats[0][q]
    # additive scalars
    for q in ['n_full', 'n_opbh_tot', 'n_opbh_max' ] :
        cstats[q] = istats[0][q] + 0
        for i in range(1,len(istats)) :
            cstats[q] += istats[i][q]
        print(q, istats[0][q], cstats[q])
    # additive arrays
    for q in ['fgdepn_hist', 'anadepn_hist', 'n_opbh', 'n_symdif',
              'obs_opbh_hist', 'fgens_opbh_hist', 'anaens_opbh_hist' ] :
        cstats[q] = istats[0][q] + 0
        for i in range(1,len(istats)) :
            cstats[q] += istats[i][q]
    # weighted average
    # ...of functions of symmetric reflectance
    for q in ['fgmeandep_std_opbh', 'meanfgspread_opbh', 'e_o_opbh', 'e_desroz_opbh' ] :
        cstats[q] = istats[0][q]*istats[0]['n_opbh']
        for i in range(1,len(istats)) :
            cstats[q] += istats[i][q]*istats[i]['n_opbh']
        cstats[q] /= maximum( cstats['n_opbh'], 1 )
        cstats[q][where(cstats['n_opbh']<5)] = nan
    # ...of functions of symmetric reflectance and mean fg mean departure
    for q in ['fgmeandep_std_sd', 'meanfgspread_sd', 'meaneo_sd', 'e_desroz_sd', 'std_e_desroz_sd' ] :
        cstats[q] = istats[0][q]*istats[0]['n_symdif']
        for i in range(1,len(istats)) :
            cstats[q] += istats[i][q]*istats[i]['n_symdif']
        cstats[q] /= maximum( cstats['n_symdif'], 1 )
        cstats[q][where(cstats['n_symdif']<5)] = nan
        cstats[q][where(abs(cstats[q])<1e-10)] = nan

    # plot .............................................................................................................

    # plot cumulative statistics
    plot_departure_statistics( cstats, name=xp.expid, output_path=output_path, file_type=file_type,
                               prefix=times[0]+'-'+last_time+'_' )

    # plot statistics for individual times
    for i, time in enumerate(times_available) :
        stats = istats[i]
        for q in ['fgmeandep_std_opbh', 'meanfgspread_opbh', 'e_o_opbh', 'e_desroz_opbh'] :
            stats[q][where(stats['n_opbh']==0)] = nan
        plot_departure_statistics( stats, name=xp.expid+' t='+time,output_path=output_path, file_type=file_type,
                                   prefix=time+'_' )


#-----------------------------------------------------------------------------------------------------------------------
def plot_departure_statistics( stats, output_path='', file_type='png', prefix='', name='' ) :

    #print 'plot_departure_statistics : received'
    #for s in stats :
    #    print ' -- ', s, stats[s]

    # normalization factor to convert from histogram to PDF
    nrm = stats['delta_dep'] * stats['n_full'] * stats['n_ens']

    print('normalization : ', stats['delta_dep'], stats['n_full'], stats['n_ens'], ' -> ', nrm)

    # compute gaussian fits

    nmin = 100
    imin = 0
    for i in range(stats['fgdepn_hist'].size) :
        if stats['fgdepn_hist'][i] > nmin :
            imin = i
            break
    imax = stats['fgdepn_hist'].size-1
    for i in arange(stats['fgdepn_hist'].size)[::-1] :
        if stats['fgdepn_hist'][i] > nmin :
            imax = i
            break
    print('considering %d < i < %d for fit...' % (imin,imax))
    popt, pcov = curve_fit( gaussian, stats['ndep_bin_centers'][imin:imax+1], stats['fgdepn_hist'][imin:imax+1], bounds=([1e3,0.1],[1e12,1]) )
    dep_hist_fit = gaussian( stats['ndep_bin_centers'], *popt)
    print('gaussian fit parameters : ', popt)

    popt1, pcov1 = curve_fit( gaussian1, stats['ndep_bin_centers'][imin:imax+1], stats['fgdepn_hist'][imin:imax+1] )
    dep_hist_fit1 = gaussian1( stats['ndep_bin_centers'], *popt1)
    idcs = where(dep_hist_fit1 > nmin)

    # PDF plot
    fig, ax = plt.subplots()
    ax.semilogy( stats['ndep_bin_centers'], stats['fgdepn_hist']     /nrm,  color='b', label='B-O (%d obs.)'      % stats['n_full'], linewidth=1   )
    ax.semilogy( stats['ndep_bin_centers'], stats['anadepn_hist']    /nrm,  color='r', label='A-O', linewidth=1   )
    ax.semilogy( stats['ndep_bin_centers'], dep_hist_fit1/nrm, ':', color='k', label='w=1 fit' )
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle('FG departures for '+ name )
    ax.set_xlabel('H(x)-y / sqrt(e_o**2+spread**2)')
    ax.set_ylabel('PDF')
    ax.grid()
    maxval = maximum( stats['fgdepn_hist'].max()/nrm, stats['anadepn_hist'].max()/nrm ) * 1.1
    ax.set_ylim((1e-4*maxval,maxval))
    fig.savefig(prefix+'fg_dep_components.'+file_type)
    plt.close(fig)

    # PDF plot (linear scale)
    fig, ax = plt.subplots()
    ax.plot( stats['ndep_bin_centers'], stats['fgdepn_hist']     /nrm,  color='b', label='B-O (%d obs.)'      % stats['n_full'], linewidth=1   )
    ax.plot( stats['ndep_bin_centers'], stats['anadepn_hist']    /nrm,  color='r', label='A-O', linewidth=1   )
    ax.plot( stats['ndep_bin_centers'], dep_hist_fit1/nrm, ':', color='k', label='w=1 fit' )
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle('FG departures for '+ name )
    ax.set_xlabel('H(x)-y / sqrt(e_o**2+spread**2)')
    ax.set_ylabel('PDF')
    ax.grid()
    ax.set_ylim((0,maxval))
    fig.savefig(prefix+'fg_dep_components_linscl.'+file_type)
    plt.close(fig)

    # Similar to Harnisch+2016 Fig. 7 with value instead of cloud impact
    fig, ax = plt.subplots()
    ax.plot( stats['opbh_bin_centers'], stats['fgmeandep_std_opbh'], 'k',   label='std(FG mean departures)' )
    ax.plot( stats['opbh_bin_centers'], stats['meanfgspread_opbh'],  '--b', label='mean spread'  )
    #ax.plot( stats['opbh_bin_centers'], sqrt( maximum( fgdep_std_opbh**2 - fg_std_opbh**2, 0 ) ), ':r', label='e_o estimate'  )
    ax.legend(loc='upper left')
    fig.savefig(prefix+'fg_dep_std_opbh.'+file_type)
    plt.close(fig)

    # Desroziers estimate as function of value
    fig, ax = plt.subplots()
    ax.plot( stats['opbh_bin_centers'], stats['e_desroz_opbh'], 'k',   label='desroziers est.' )
    ax.plot( stats['opbh_bin_centers'], stats['e_o_opbh'],      'b', label='assumed error'  )
    ax.plot( stats['opbh_bin_centers'], stats['n_opbh']/float(10*stats['n_opbh'].max()),      'r', label='num. fraction / 10'  )
    ax.text( ax.get_xlim()[0], ax.get_ylim()[0], ' fraction active: %5.3f (%d of %d)' % (stats['n_opbh_tot']/float(stats['n_opbh_max']),
                                                                stats['n_opbh_tot'], stats['n_opbh_max']), ha='left', fontsize=10 )
    ax.grid()
    ax.legend(loc='upper left', frameon=False, fontsize=10, title=name )
    fig.savefig(prefix+'desroz_opbh.'+file_type)
    plt.close(fig)


    # value histograms
    fig, ax = plt.subplots()
    ax.plot( stats['opbh_bin_centers'], stats['obs_opbh_hist'], 'k', label='O' )
    ax.plot( stats['opbh_bin_centers'], stats['fgens_opbh_hist'], 'b', label='B' )
    ax.plot( stats['opbh_bin_centers'], stats['anaens_opbh_hist'], color='r', label='A' )
    ax.grid()
    ax.legend(loc='upper right', frameon=False, fontsize=10, title=name )
    fig.savefig(prefix+'opbh_hist.'+file_type)
    plt.close(fig)

    # mean desroziers error estimate as a function of O-B and O+B
    nmin = 5
    e_desroz_sd     = stats['e_desroz_sd'] + 0
    idcs = where( e_desroz_sd > 1e-10 )
    print('/////////// e_desroz_sd min/max ', e_desroz_sd[idcs].min(), e_desroz_sd[idcs].max())
    vmin, vmax = e_desroz_sd[idcs].min(), e_desroz_sd[idcs].max()
    std_e_desroz_sd = stats['std_e_desroz_sd'] + 0
    e_desroz_sd[     where( stats['n_symdif'] < nmin ) ] = nan
    std_e_desroz_sd[ where( stats['n_symdif'] < nmin ) ] = nan

    fig, ax = plt.subplots()
    if True :
        im = ax.imshow( e_desroz_sd, origin='lower', interpolation='nearest', aspect='auto',
                        extent=[ stats['omba_bins'].min(), stats['omba_bins'].max(),
                                 stats['opbh_bins'].min(), stats['opbh_bins'].max() ],
                        cmap='gist_stern', vmin=vmin, vmax=vmax ) #, vmax=e_desroz_sd.max(), vmin=0 ) # RdBu # vmin=0, vmax=0.5,
    else :
        im = ax.contourf( stats['omba_bin_centers'], stats['opbh_bin_centers'], e_desroz_sd,
                          levels=linspace(-0.1,0.1,11) )
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'e_desroz_sd.'+file_type)
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow( stats['meaneo_sd'], origin='lower', interpolation='nearest', aspect='auto',
                        extent=[ stats['omba_bins'].min(), stats['omba_bins'].max(),
                                 stats['opbh_bins'].min(), stats['opbh_bins'].max() ],
                        cmap='gist_stern', vmin=vmin, vmax=vmax  ) #, vmax=e_desroz_sd.max(), vmin=0 )
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'e_o_sd.'+file_type)
    plt.close(fig)

    # how many cases are in each bin
    fig, ax = plt.subplots()
    im = ax.imshow( stats['n_symdif'], origin='lower', interpolation='nearest', aspect='auto',
                        extent=[ stats['omba_bins'].min(), stats['omba_bins'].max(),
                                 stats['opbh_bins'].min(), stats['opbh_bins'].max() ],
                        cmap='terrain' ) # RdBu # vmin=0, vmax=0.5,
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'ncases_sd.'+file_type)
    plt.close(fig)

    if True :
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in range(len(stats['opbh_bin_centers'])) :
            ax[0].plot( stats['omba_bin_centers'], e_desroz_sd[i,:], color=colseq(i,len(stats['opbh_bins'])) )
            ax[0].fill_between( stats['omba_bin_centers'],
                                 e_desroz_sd[i,:]-0.5*std_e_desroz_sd[i,:],
                                 e_desroz_sd[i,:]+0.5*std_e_desroz_sd[i,:], color=colseq(i,len(stats['opbh_bins'])), alpha=0.25 )
            #ax[0].plot( stats['omba_bin_centers'], stats['meaneo_sd'][i,:], color=colseq(i,len(stats['opbh_bins'])) )
        #x = linspace(-0.4,0.4,50)
        #ax[0].plot( x, 0.1 + 0.25*(x/0.4)**2, ':k' )
        #ax[0].set_xlabel('B-O')
        #ax[0].set_xlim((-0.5,0.5))
        #ax[0].set_ylim((0,0.4))
        for i in range(len(stats['omba_bin_centers'])) :
            ax[1].plot( stats['opbh_bin_centers'], e_desroz_sd[:,i], color=colseq(i,len(stats['omba_bins'])) )
            ax[1].fill_between( stats['opbh_bin_centers'],
                                 e_desroz_sd[:,i]-0.5*std_e_desroz_sd[:,i],
                                 e_desroz_sd[:,i]+0.5*std_e_desroz_sd[:,i], color=colseq(i,len(stats['omba_bins'])), alpha=0.25 )

        #x = linspace(0.0,0.4,50)
        #ax[1].plot( x, x, ':k' )
        #ax[1].set_xlabel('(B+O)/2')
        #ax[1].set_ylim((0,0.4))
        fig.savefig(prefix+'e_desroz_sd_curves.'+file_type)
        plt.close(fig)


def gaussian( x, a, b ) :
    return a*exp( -(x/b)**2 )

def gaussian1( x, a ) :
    return a*exp( -x**2 )

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Generate plots for KENDA experiment')

    #parser.add_argument(       '--e-o',         dest='e_o',   help='if specified, use this value instead of the ekf file values', type=float, default=-1.0 )
    parser.add_argument( '-F', '--use-fg-spread', dest='use_fg_spread', action='store_true',
                         help='use fg spread in normalization of all histograms (default: use ana spread for ana)' )

    parser.add_argument( '-V', '--variable',   dest='varname',    help='variable name', default='' )
    parser.add_argument( '-O', '--obstype',    dest='obstype',    help='observation type', default='' )

    parser.add_argument( '--depmax', dest='depmax', help='maximum departure', default='' )
    parser.add_argument( '--valmaxmin', dest='valmaxmin', help='maximum,minimum observation value', default='' )

    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default='' )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default='' )
    parser.add_argument( '-d', '--delta-time',  dest='delta_time',  help='time interval', default='' )
    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )

    parser.add_argument( 'logfile', metavar='logfile', help='log file name', nargs='*' )
    args = parser.parse_args()

    # process all log files

    for logfile in args.logfile :

        print('processing '+logfile)

        xp = Experiment(logfile)
        print('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
               xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] ))

        # set some default values

        if args.output_path != '' :
            if args.output_path != 'auto' :
                output_path = args.output_path+'/'
            else :
                output_path = xp.settings['PLOT_DIR']+'/departures/'
                if not os.path.exists(output_path) :
                    os.system('mkdir '+output_path)
        else :
            output_path = ''

        if (args.start_time != '') or (args.end_time != '') or (args.delta_time != '') :
            start_time = args.start_time if args.start_time != '' else xp.veri_times[0]
            end_time   = args.end_time   if args.end_time   != '' else xp.veri_times[-1]
            delta_time = args.delta_time if args.delta_time != '' else int(xp.settings['ASSINT'])
            times = time_range( start_time, end_time, delta_time  )
        else :
            times = xp.veri_times[:-2]

        plot_cumulative_departure_statistics( xp, args.obstype, args.varname, times=times,
                                              use_fg_spread=args.use_fg_spread,
                                              dep_max = float(args.depmax) if args.depmax != '' else None,
                                              val_min = float(args.valmaxmin.split(',')[1]) if args.valmaxmin != '' else None,
                                              val_max = float(args.valmaxmin.split(',')[0]) if args.valmaxmin != '' else None,
                                              output_path=output_path, file_type=args.file_type )
