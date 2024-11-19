#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . V I S O P _ D E P A R T U R E S
#  analyze reflectance departures
#
#  2016.10 L.Scheck

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os, argparse, copy
from numpy import *
from scipy.optimize import curve_fit

from kendapy.colseq import colseq
from kendapy.experiment import Experiment
from kendapy.time14 import Time14, time_range
from kendapy.binplot import binplot
import kendapy.ekf
from kendapy.bacy_exp import BacyExp
from kendapy.ekf import Ekf
from kendapy.bacy_utils import str2t, to_timedelta, to_datetime, midnight

#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description='Generate plots for KENDA experiment')

    #parser.add_argument(       '--e-o',         dest='e_o',   help='if specified, use this value instead of the ekf file values', type=float, default=-1.0 )
    parser.add_argument( '-F', '--use-fg-spread', dest='use_fg_spread', action='store_true',
                         help='use fg spread in normalization of all histograms (default: use ana spread for ana)' )
    parser.add_argument( '-M', '--error-model', dest='error_model', help='error model to be used instead of errors from ekf file (default:None) [constant<abs>|mz17like|linear<amp>|squared<amp>]', default='' )
    
    parser.add_argument( '-s', '--start-time',  dest='start_time',  help='start time',    default=None )
    parser.add_argument( '-e', '--end-time',    dest='end_time',    help='end time',      default=None )
    parser.add_argument( '-S', '--start-daily', dest='start_daily', default='6:00',         help='daily start hour' )
    parser.add_argument( '-E', '--end-daily',   dest='end_daily',   default='17:00',        help='daily end hour' )

    parser.add_argument( '-d', '--delta-time',  dest='delta_time',  help='time interval', default='' )
    parser.add_argument( '-p', '--path',        dest='output_path', help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument( '-f', '--filetype',    dest='file_type',   help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )

    parser.add_argument( '-C', '--compare',     dest='compare',     help='generate comparison plots',   action='store_true' )
    parser.add_argument(       '--colors',      dest='colors',      help='comma-separated list of colors for the experiments', default='' )
    parser.add_argument(       '--names',       dest='names',       help='comma-separated list of names (use ,, for commas within the names)', default='' )

    parser.add_argument( 'xp_path', metavar='xp_path', help='path to experiment', nargs='*' )
    
    return parser

#-----------------------------------------------------------------------------------------------------------------------
def get_nonlinear_anaens( xp, t, ekf=None, channel='VIS006' ) :
    """Return analysis ensemble of model equivalents computed from """

    if ekf is None :
        ekf = xp.get_ekf( t, 'RAD', varname_filter='REFL' )

    vens = xp.get_visop( t, fcsttime=0, channel=channel, preload=True )
    n_ens, n_obs_v = vens.i2o('meq').shape
    n_obs = ekf.obs().size
    print('#obs in ekf : ', n_obs)
    print('#obs in npy : ', n_obs_v)

    # print 'anaens shape : ', ekf.anaens().shape # is n_ens, n_obs

    lat, lon   = ekf.obs(param='lat'), ekf.obs(param='lon')
    elat, elon = vens.i2o('lat'),      vens.i2o('lon')

    anaens_nl = zeros((n_ens,n_obs))

    for i in range(n_obs) : # loop over ekf observations
        # find corresponding npy observation
        j = argmin( abs(lat[i]-elat) + abs(lon[i]-elon) )
        if abs(lat[i]-elat[j]) + abs(lon[i]-elon[j]) > 0.01 : # errors are O(0.001) degree -- why?
            print('WARNING: no match for observations ', i, j, abs(lat[i]-elat[j]) + abs(lon[i]-elon[j]))
        anaens_nl[:,i] = vens.i2o('meq')[:,j]

    return anaens_nl


#-----------------------------------------------------------------------------------------------------------------------
def get_departure_statistics( xp, t, error_model='', use_fg_spread=False, quad_mean_e=False,
                              obs_type='RAD', varname='REFL', state='active', use_ana_nl=False,
                              verbose=True ) :

    if verbose :
        print()
        print('>>>>>>>>>> GET_DEPARTURE_STATISTICS', xp.exp_dir, t)

    # read ekfRAD file
    ekf_filename = xp.get_filename('ekf', time_valid=t, obs_type=obs_type)
    #print("trying to read", xp.ekf_filename( t, 'RAD' ))
    print("trying to read", ekf_filename)
    #ekf = xp.get_ekf( t, 'RAD', state_filter='active', varname_filter='REFL' )
    ekf = Ekf( ekf_filename, filter='state={} varname={}'.format(state,varname) )
    newshape = [1]+list(ekf.obs().shape)

    # compute departures and their standard deviations
    fgdep = ekf.obs().reshape(newshape) - ekf.fgens() #ekf.departures( 'fgens', normalize=False )
    fgdep_std = fgdep.std(axis=0,ddof=1)

    anadep = ekf.obs().reshape(newshape) - ekf.anaens() #ekf.departures( 'anaens', normalize=False )
    anadep_std = anadep.std(axis=0,ddof=1)

    fgmean_err    = ekf.fgmean()           - ekf.obs() 
    anamean_err   = ekf.anamean()          - ekf.obs() 

    if use_ana_nl :
        # recompute ana departures using nonlinear analysis ensemble [WARNING: contains also contribution from inflation!]
        anaens_nl = get_nonlinear_anaens( xp, t, ekf=ekf )
        anadep_nl = ekf.obs().reshape(newshape) - anaens_nl

        # bias due to operator nonlinearity
        anaensbias_nl  = ekf.anaens() - anaens_nl
        anameanbias_nl = ekf.anamean() - anaens_nl.mean(axis=0)

        anameannl_err = anaens_nl.mean(axis=0) - ekf.obs()

    if False :
        fig, ax = plt.subplots()
        ax.scatter( fgmean_err, anamean_err, c='b', alpha=0.1, edgecolor='' )
        ax.scatter( fgmean_err, anameannl_err, c='r', alpha=0.1, edgecolor='' )
        ax.plot( (-0.7,0.7), (-0.7,0.7), '--k' )
        fig.savefig(t+'_error_reduction_scatterplot.png')
        sys.exit()

    # compute desroziers estimate for e_o
    e2 = (fgdep*anadep).mean(axis=0)
    #e2 = ensstat.desroz( fgdep, anadep, diagonal_only=True )
    if e2.min() < 0 :
        idcs = where( e2 < 0 )
        print('WARNING: Negative numbers encountered in desroziers calculation (%d of %d) -> set to zero!' % (len(idcs[0]),ekf.obs().size))
        e2[ idcs ] = 0
    e_desroz = sqrt( e2 )

    if use_ana_nl :
        # ...do the same with the nonlinear analysis departures
        e2 = (fgdep*anadep_nl).mean(axis=0)
        if e2.min() < 0 :
            idcs = where( e2 < 0 )
            print('WARNING: Negative numbers encountered in nonlinear desroziers calculation (%d of %d) -> set to zero!' % (len(idcs[0]),ekf.obs().size))
            e2[ idcs ] = 0
        e_desroz_nl = sqrt( e2 )

    print('mean desroziers error ', e_desroz[where(e_desroz>0)].mean(), (e_desroz_nl[where(e_desroz_nl>0)].mean() if use_ana_nl else ''))

    # compute desroziers estimate for sqrt(diag(HBH^t)) = FG spread in observation space
    increments = ekf.anaens() - ekf.fgens() #ekf.increments('ens')
    s2 = (fgdep*increments).mean(axis=0)
    if s2.min() < 0 :
        idcs = where( s2 < 0 )
        print('WARNING: Negative numbers encountered in desroziers spread calculation (%d of %d) -> set to zero!' % (len(idcs[0]),ekf.obs().size))
        s2[ idcs ] = 0
    spread_desroz = sqrt( s2 )

    if use_ana_nl :
        increments_nl = anaens_nl - ekf.fgens() #ekf.increments('ens',ana=anaens_nl)
        s2 = (fgdep*increments_nl).mean(axis=0)
        if s2.min() < 0 :
            idcs = where( s2 < 0 )
            print('WARNING: Negative numbers encountered in nonlinear desroziers spread calculation (%d of %d) -> set to zero!' % (len(idcs[0]),ekf.obs().size))
            s2[ idcs ] = 0
        spread_desroz_nl = sqrt( s2 )


    # determine assumed observation error
    e_o_1d = ekf.obs(param='e_o')

    if error_model.startswith('constant') :
        em_amp = float(error_model[8:])
        e_o = e_o_1d*0 + em_amp

    elif error_model.startswith('linear') :
        em_amp = float(error_model[6:])
        dist = maximum( 0, abs(ekf.fgmean() - ekf.obs()) - ekf.fgspread() )
        e_o = e_o_1d*( 1.0 + em_amp*dist )

    elif error_model.startswith('squared') :
        em_amp = float(error_model[7:])
        dist = sqrt(maximum( 0, (ekf.fgmean() - ekf.obs())**2 - ekf.fgspread()**2 ))
        e_o = e_o_1d*( 1.0 + em_amp*dist )

    elif error_model == 'mz17like' :
        e_o = sqrt(maximum( e_o_1d**2, (ekf.fgmean() - ekf.obs())**2 - ekf.fgspread()**2 ))

    else :
        # default: Use unmodified errors from EKF file
        e_o = e_o_1d


    # compute departure histograms .....................................................................................

    dep_bins = linspace(-3.0,3.0,151)
    dep_bin_centers = 0.5*(dep_bins[1:]+dep_bins[:-1])
    delta_dep = dep_bins[1] - dep_bins[0]

    # normalize fg departures by expectation value
    fgdepn = fgdep / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.fgspread().reshape((1,e_o.size))**2 )
    print('mean            fg departure : ', abs(fgdep).mean())
    print('mean normalized fg departure : ', abs(fgdepn).mean())

    if use_fg_spread : # normalize with e_o**2 + fg_spread**2
        anadepn        = anadep    / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.fgspread().reshape((1,e_o.size))**2 )
        if use_ana_nl :
            anadepn_nl = anadep_nl / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.fgspread().reshape((1,e_o.size))**2 )

    else : # normalize with e_o**2 + ana_spread**2
        #anadepn    = anadep / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.statistics()['anaens']['spread2d'].reshape((1,e_o.size))**2 )
        anadepn        = anadep / sqrt( e_o.reshape((1,e_o.size))**2 + ekf.anaspread().reshape((1,e_o.size))**2 )
        if use_ana_nl :
            anadepn_nl = anadep_nl / sqrt( e_o.reshape((1,e_o.size))**2 + anaens_nl.std(axis=0,ddof=1).reshape((1,e_o.size))**2 )

    # full departure histogram
    fgdepn_hist = histogram( fgdepn, bins=dep_bins )[0]
    anadepn_hist = histogram( anadepn, bins=dep_bins )[0]
    n_ens, n_full = fgdepn.shape
    
    if use_ana_nl :
        anadepn_nl_hist = histogram( anadepn_nl, bins=dep_bins )[0]

    #ddd = abs(fgdepn_hist-anadepn_hist)
    #print 'MEAN/MAX DIFFERENCE BETWEEN fgdepn_hist and anadepn_hist : ', ddd.mean(), ddd.max()


    # clear sky - clear sky cases
    obmean_ens = zeros(fgdepn.shape) + 0.5*( ekf.obs() + ekf.fgmean() ).reshape((1,e_o.size))
    dist_ens   = zeros(fgdepn.shape) + abs(  ekf.obs() - ekf.fgmean() ).reshape((1,e_o.size))
    idcs = where((obmean_ens < 0.2) & (dist_ens < 0.05) )
    fgdepn_hist_cs2 = histogram( fgdepn[idcs], bins=dep_bins )[0]
    n_cs2 = len(idcs[0])//n_ens

    # rest
    idcs = where((obmean_ens >= 0.2) | (dist_ens > 0.05) )
    fgdepn_hist_rest = histogram( fgdepn[idcs], bins=dep_bins )[0]
    n_rest = len(idcs[0])//n_ens

    res = { 'dep_bins':dep_bins, 'dep_bin_centers':dep_bin_centers, 'delta_dep':delta_dep,
            'fgdepn':fgdepn, 'anadepn':anadepn, 
            'fgdepn_hist':fgdepn_hist, 'fgdepn_hist_cs2':fgdepn_hist_cs2, 'fgdepn_hist_rest':fgdepn_hist_rest,
            'anadepn_hist':anadepn_hist, 
            'n_ens':n_ens, 'n_full':n_full, 'n_cs2':n_cs2, 'n_rest':n_rest, 'error_model':error_model }

    if use_ana_nl :
        res.update({'anadepn_nl':anadepn_nl, 'anadepn_nl_hist':anadepn_nl_hist})

    res.update({'fgmean':ekf.fgmean(),'fgspread':ekf.fgspread()})


    # compute reflectance histograms for fg, ana_est, ana_nl ...........................................................

    refl_bins = linspace( 0, 1.0, 51 )
    refl_bin_centers = 0.5*(refl_bins[1:]+refl_bins[:-1])

    obs_refl_hist       = histogram( ekf.obs(),    bins=refl_bins )[0]/1.0
    fgens_refl_hist     = histogram( ekf.fgens(),  bins=refl_bins )[0]/float(xp.n_ens)
    anaens_refl_hist    = histogram( ekf.anaens(), bins=refl_bins )[0]/float(xp.n_ens)
    if use_ana_nl :
        anaens_nl_refl_hist = histogram( anaens_nl,    bins=refl_bins )[0]/float(xp.n_ens)
    res.update({'obs_refl_hist':obs_refl_hist, 'fgens_refl_hist':fgens_refl_hist, 'anaens_refl_hist':anaens_refl_hist})
    if use_ana_nl :
        res.update({'anaens_nl_refl_hist':anaens_nl_refl_hist })

    # compute std(FG departures), mean spread and desroziers estimates as functions of symmetric mean reflectance ......

    obmean = 0.5*( ekf.obs() + ekf.fgmean() )
    fgmeandep =    ekf.obs() - ekf.fgmean()

    fgmeandep_std_refl = zeros(refl_bins.size-1)
    meanfgspread_refl  = zeros(refl_bins.size-1)
    n_refl             = zeros(refl_bins.size-1, dtype=int)
    e_o_refl           = zeros(refl_bins.size-1)
    e_desroz_refl      = zeros(refl_bins.size-1)
    e_desroz_nl_refl   = zeros(refl_bins.size-1)
    spread_refl            = zeros(refl_bins.size-1)
    spread_desroz_refl     = zeros(refl_bins.size-1)
    if use_ana_nl :
        spread_desroz_nl_refl  = zeros(refl_bins.size-1)
        anamean_meannlerr_refl = zeros(refl_bins.size-1)
        anamean_maxnlerr_refl  = zeros(refl_bins.size-1)

    for i in range(refl_bins.size-1) :
        idcs = where( (obmean >= refl_bins[i]) & (obmean < refl_bins[i+1]) )
        n_refl[i] = len(idcs[0])
        if n_refl[i] > 0 :
            fgmeandep_std_refl[i] = fgmeandep[     idcs].std()  # standard deviation of ensemble mean departures
            meanfgspread_refl[i]  = ekf.fgspread()[idcs].mean() # mean ensemble standard deviation
            if quad_mean_e :
                e_o_refl[i]              = sqrt( (e_o_1d[        idcs]**2).mean() )
                e_desroz_refl[i]         = sqrt( (e_desroz[      idcs]**2).mean() )
                spread_refl[i]           = sqrt( (ekf.fgspread()[  idcs]**2).mean() )
                spread_desroz_refl[i]    = sqrt( (spread_desroz[   idcs]**2).mean() )
                if use_ana_nl :
                    spread_desroz_nl_refl[i] = sqrt( (spread_desroz_nl[idcs]**2).mean() )
                    e_desroz_nl_refl[i]      = sqrt( (e_desroz_nl[   idcs]**2).mean() )
            else :
                e_o_refl[i]              = e_o_1d[        idcs].mean()
                e_desroz_refl[i]         = e_desroz[      idcs].mean()
                spread_refl[i]           = ekf.fgspread()[  idcs].mean()
                spread_desroz_refl[i]    = spread_desroz[   idcs].mean()
                if use_ana_nl :
                    e_desroz_nl_refl[i]      = e_desroz_nl[   idcs].mean()
                    spread_desroz_nl_refl[i] = spread_desroz_nl[idcs].mean()

            if use_ana_nl :
                anamean_meannlerr_refl[i] = abs(anameanbias_nl[idcs]).mean()
                anamean_maxnlerr_refl[i]  = abs(anameanbias_nl[idcs]).max()

    res.update({'refl_bins':refl_bins, 'refl_bin_centers':refl_bin_centers, 'fgmeandep_std_refl':fgmeandep_std_refl,
                'meanfgspread_refl':meanfgspread_refl, 'n_refl':n_refl, 'n_refl_tot':n_refl.sum(), 'n_refl_max':ekf.n_body,
                'e_o_refl':e_o_refl, 'e_desroz_refl':e_desroz_refl, 
                'spread_refl':spread_refl, 'spread_desroz_refl':spread_desroz_refl})
    if use_ana_nl :
        res.update({'anamean_meannlerr_refl':anamean_meannlerr_refl, 'anamean_maxnlerr_refl':anamean_maxnlerr_refl,
                    'e_desroz_nl_refl':e_desroz_nl_refl, 'spread_desroz_nl_refl':spread_desroz_nl_refl })


    # compute everything as functions of symmetric mean reflectance (O+B)/2 and O-B ....................................
    rsym = 0.5*( ekf.fgmean() + ekf.obs() )
    rdif =       ekf.fgmean() - ekf.obs()

    rsym_bins = linspace( 0, 1.0, 21 )
    rsym_bin_centers = 0.5*(rsym_bins[1:]+rsym_bins[:-1])

    rdif_bins = linspace( -1.0, 1.0, 20 )
    rdif_bin_centers = 0.5*(rdif_bins[1:]+rdif_bins[:-1])

    n_symdif = zeros((rsym_bins.size-1,rdif_bins.size-1), dtype=int)

    fgmeandep_std_sd  = zeros((rsym_bins.size-1,rdif_bins.size-1))
    meanfgspread_sd   = zeros((rsym_bins.size-1,rdif_bins.size-1))
    meaneo_sd         = zeros((rsym_bins.size-1,rdif_bins.size-1))
    e_desroz_sd       = zeros((rsym_bins.size-1,rdif_bins.size-1))
    std_e_desroz_sd   = zeros((rsym_bins.size-1,rdif_bins.size-1))
    spread_desroz_sd     = zeros((rsym_bins.size-1,rdif_bins.size-1))    
    std_spread_desroz_sd = zeros((rsym_bins.size-1,rdif_bins.size-1))
    if use_ana_nl :
        e_desroz_nl_sd    = zeros((rsym_bins.size-1,rdif_bins.size-1))
        spread_desroz_nl_sd  = zeros((rsym_bins.size-1,rdif_bins.size-1))
        anameanbias_nl_sd = zeros((rsym_bins.size-1,rdif_bins.size-1))
        anameanerr_nl_sd =  zeros((rsym_bins.size-1,rdif_bins.size-1))

    fgmean_err_sd    = zeros((rsym_bins.size-1,rdif_bins.size-1))
    anamean_err_sd   = zeros((rsym_bins.size-1,rdif_bins.size-1))
    err_red_lin_sd = zeros((rsym_bins.size-1,rdif_bins.size-1))
    if use_ana_nl :
        anameannl_err_sd = zeros((rsym_bins.size-1,rdif_bins.size-1))
        err_red_nl_sd = zeros((rsym_bins.size-1,rdif_bins.size-1))
    
    for i in range(rsym_bins.size-1) :
        for j in range(rdif_bins.size-1) :
            idcs = where(   (rsym >= rsym_bins[i]) & (rsym < rsym_bins[i+1]) \
                          & (rdif >= rdif_bins[j]) & (rdif < rdif_bins[j+1]) )
            n_symdif[i,j] = len(idcs[0])
            if n_symdif[i,j] > 0 :
                fgmeandep_std_sd[i,j]  = fgmeandep[idcs].std()
                meanfgspread_sd[i,j]   = ekf.fgspread()[idcs].mean()
                meaneo_sd[i,j]         = e_o_1d[        idcs].mean()
                if use_ana_nl :
                    anameanbias_nl_sd[i,j] = anameanbias_nl[idcs].mean()
                    anameanerr_nl_sd[i,j]  = abs(anameanbias_nl[idcs]).mean()
                if quad_mean_e :
                    e_desroz_sd[i,j]      = sqrt( (e_desroz[      idcs]**2).mean() )
                    spread_desroz_sd[i,j] = sqrt( (spread_desroz[ idcs]**2).mean() )
                    if use_ana_nl :
                        e_desroz_nl_sd[i,j]   = sqrt( (e_desroz_nl[   idcs]**2).mean() )
                        spread_desroz_nl_sd[i,j] = sqrt( (spread_desroz_nl[ idcs]**2).mean() )
                else :
                    e_desroz_sd[i,j]      = e_desroz[      idcs].mean()
                    spread_desroz_sd[i,j] = spread_desroz[ idcs].mean()
                    if use_ana_nl :
                        e_desroz_nl_sd[i,j]   = e_desroz_nl[   idcs].mean()
                        spread_desroz_nl_sd[i,j] = spread_desroz_nl[ idcs].mean()
                std_e_desroz_sd[i,j]      = e_desroz[      idcs].std()
                std_spread_desroz_sd[i,j] = spread_desroz[ idcs].std()

                fgmean_err_sd[i,j]    = fgmean_err[   idcs].mean()
                anamean_err_sd[i,j]   = anamean_err[  idcs].mean()
                err_red_lin_sd[i,j] = ( (fgmean_err[idcs] - anamean_err[idcs]  ) / fgmean_err[idcs] ).mean()
                if use_ana_nl :
                    anameannl_err_sd[i,j] = anameannl_err[idcs].mean()
                    err_red_nl_sd[i,j]  = ( (fgmean_err[idcs] - anameannl_err[idcs]) / fgmean_err[idcs] ).mean()
                #err_red_lin_sd[i,j] = (fgmean_err_sd[i,j] - anamean_err_sd[i,j]) / fgmean_err_sd[i,j]
                #err_red_nl_sd[i,j] = (fgmean_err_sd[i,j] - anameannl_err_sd[i,j]) / fgmean_err_sd[i,j] 
               
                
    res.update({ 'rsym_bins':rsym_bins, 'rsym_bin_centers':rsym_bin_centers,
                 'rdif_bins':rdif_bins, 'rdif_bin_centers':rdif_bin_centers,
                 'n_symdif':n_symdif, 'fgmeandep_std_sd':fgmeandep_std_sd, 'meanfgspread_sd':meanfgspread_sd,
                 'meaneo_sd':meaneo_sd, 'e_desroz_sd':e_desroz_sd,  'std_e_desroz_sd':std_e_desroz_sd,
                 'spread_desroz_sd':spread_desroz_sd,  'std_spread_desroz_sd':std_spread_desroz_sd,                 
                 'fgmean_err_sd':fgmean_err_sd, 'anamean_err_sd':anamean_err_sd, 
                 'err_red_lin_sd':err_red_lin_sd  })

    if use_ana_nl :
        res.update({ 'e_desroz_nl_sd':e_desroz_nl_sd, 'spread_desroz_nl_sd':spread_desroz_nl_sd, 'anameanbias_nl_sd':anameanbias_nl_sd,
        'anameanerr_nl_sd':anameanerr_nl_sd, 'anameannl_err_sd':anameannl_err_sd, 'err_red_nl_sd':err_red_nl_sd })

    # compute everything as functions of O-B ...........................................................................

    n_dif = zeros((rdif_bins.size-1), dtype=int)
    
    fgmean_err_df    = zeros((rdif_bins.size-1))
    anamean_err_df   = zeros((rdif_bins.size-1))
    anamean_err_std_df   = zeros((rdif_bins.size-1))
    err_red_lin_df     = zeros((rdif_bins.size-1))
    err_red_lin_std_df = zeros((rdif_bins.size-1))
    if use_ana_nl :
        anameannl_err_df = zeros((rdif_bins.size-1))
        anameannl_err_std_df = zeros((rdif_bins.size-1))
        err_red_nl_df      = zeros((rdif_bins.size-1))
        err_red_nl_std_df  = zeros((rdif_bins.size-1))
        
    for i in range(rdif_bins.size-1) :
        idcs = where( (rdif >= rdif_bins[i]) & (rdif < rdif_bins[i+1]) )
        n_dif[i] = len(idcs[0])
        if n_dif[i] > 0 :
                fgmean_err_df[i]    = fgmean_err[   idcs].mean()
                anamean_err_df[i]   = anamean_err[  idcs].mean()
                anamean_err_std_df[i]   = anamean_err[  idcs].std()
                err_red_lin_df[i]     = ( (fgmean_err[idcs] - anamean_err[idcs]  ) / fgmean_err[idcs] ).mean()                
                err_red_lin_std_df[i] = ( (fgmean_err[idcs] - anamean_err[idcs]  ) / fgmean_err[idcs] ).std()
                if use_ana_nl :
                    anameannl_err_df[i] = anameannl_err[idcs].mean()
                    anameannl_err_std_df[i] = anameannl_err[idcs].std()
                    err_red_nl_df[i]      = ( (fgmean_err[idcs] - anameannl_err[idcs]) / fgmean_err[idcs] ).mean()
                    err_red_nl_std_df[i]  = ( (fgmean_err[idcs] - anameannl_err[idcs]) / fgmean_err[idcs] ).std()
                #err_red_lin_df[i] = (fgmean_err_df[i] - anamean_err_df[i]) / fgmean_err_df[i]
                #err_red_nl_df[i] = (fgmean_err_df[i] - anameannl_err_df[i]) / fgmean_err_df[i]
                
    res.update({ 'n_dif':n_dif, 'fgmean_err_df':fgmean_err_df,
                 'anamean_err_df':anamean_err_df, 
                 'anamean_err_std_df':anamean_err_std_df, 
                 'err_red_lin_df':err_red_lin_df, 
                 'err_red_lin_std_df':err_red_lin_std_df })
    if use_ana_nl :
        res.update({ 'anameannl_err_df':anameannl_err_df, 'anameannl_err_std_df':anameannl_err_std_df,
                     'err_red_nl_df':err_red_nl_df, 'err_red_nl_std_df':err_red_nl_std_df })

    # compute mean desroziers error only from those O-B bins where the spread estimate does not differ too much from the
    # actual spread
    e_desroz_rsym = zeros(rsym_bins.size-1)
    if use_ana_nl :
        e_desroz_nl_rsym = zeros(rsym_bins.size-1)
    thres = 0.2
    for i in range(rsym_bins.size-1) :
        dspread =  abs( meanfgspread_sd[i,:] - spread_desroz_sd[i,:] ) / meanfgspread_sd[i,:]
        e_desroz_rsym[i] = e_desroz_sd[i,:][ where(dspread<thres)].mean()
        if use_ana_nl :
            dspread =  abs( meanfgspread_sd[i,:] - spread_desroz_nl_sd[i,:] ) / meanfgspread_sd[i,:]
            e_desroz_nl_rsym[i] = e_desroz_nl_sd[i,:][ where(dspread<thres)].mean()
    res.update({ 'e_desroz_rsym':e_desroz_rsym })
    if use_ana_nl :
        res.update({'e_desroz_nl_rsym':e_desroz_nl_rsym })

    # do the same without integrating over O-B, be more strict
    e_desroz_sd_strict = zeros(e_desroz_sd.shape)
    n_symdif_strict    = zeros(n_symdif.shape)
    idcs = where( abs( meanfgspread_sd - spread_desroz_sd ) / ( meanfgspread_sd + 1e-5 )  < 0.1 )
    e_desroz_sd_strict[idcs] = e_desroz_sd[idcs]
    n_symdif_strict[idcs]    = n_symdif[idcs]
    res.update({ 'e_desroz_sd_strict':e_desroz_sd_strict, 'n_symdif_strict':n_symdif_strict })

    # compute everything as functions of symmetric mean reflectance (O+B)/2 and |O-B|-spread ...........................
    if True :
        rnld = abs( ekf.fgmean() - ekf.obs() ) - ekf.fgspread()
        rnld_bins = linspace( -0.5, 0.5, 20 )

        #rnld = ekf.fgspread()
        #rnld_bins = linspace( 0.0, 0.7, 20 )

        rnld_bin_centers = 0.5*(rnld_bins[1:]+rnld_bins[:-1])

        n_symnld = zeros((rsym_bins.size-1,rnld_bins.size-1), dtype=int)
        fgmeandep_std_sn  = zeros((rsym_bins.size-1,rnld_bins.size-1))
        meanfgspread_sn   = zeros((rsym_bins.size-1,rnld_bins.size-1))
        meaneo_sn         = zeros((rsym_bins.size-1,rnld_bins.size-1))
        if use_ana_nl :
            anameanbias_nl_sn = zeros((rsym_bins.size-1,rnld_bins.size-1))
            anameanerr_nl_sn  = zeros((rsym_bins.size-1,rnld_bins.size-1))

        for i in range(rsym_bins.size-1) :
            for j in range(rnld_bins.size-1) :
                idcs = where(   (rsym >= rsym_bins[i]) & (rsym < rsym_bins[i+1]) \
                              & (rnld >= rnld_bins[j]) & (rnld < rnld_bins[j+1]) )
                n_symnld[i,j] = len(idcs[0])
                if n_symnld[i,j] > 0 :
                    fgmeandep_std_sn[i,j]  = fgmeandep[idcs].std()
                    meanfgspread_sn[i,j]   = ekf.fgspread()[idcs].mean()
                    meaneo_sn[i,j]         = e_o_1d[        idcs].mean()
                    if use_ana_nl :
                        anameanbias_nl_sn[i,j] = anameanbias_nl[idcs].mean()
                        anameanerr_nl_sn[i,j]  = abs(anameanbias_nl[idcs]).mean()

        res.update({ 'rnld_bins':rnld_bins, 'rnld_bin_centers':rnld_bin_centers,
                     'n_symnld':n_symnld, 'fgmeandep_std_sn':fgmeandep_std_sn, 'meanfgspread_sn':meanfgspread_sn,
                     'meaneo_sn':meaneo_sn })
        if use_ana_nl :
            res.update({'anameanbias_nl_sn':anameanbias_nl_sn, 'anameanerr_nl_sn':anameanerr_nl_sn })

    #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    #print 'mean spread = %f, mean |dif| = %f' % (ekf.fgspread().mean(),abs(rdif).mean())
    #idcs = where( anameanbias_nl > 0.05 )
    #print 'most positive anamean nl bias for rsym=%f, rdif=%f, spread=%f' % (rsym[idcs].mean(),rdif[idcs].mean(),ekf.fgspread()[idcs].mean())
    #idcs = where( anameanbias_nl < 0.05 )
    #print 'most negative anamean nl bias for rsym=%f, rdif=%f, spread=%f' % (rsym[idcs].mean(),rdif[idcs].mean(),ekf.fgspread()[idcs].mean())
    #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    return res


#-----------------------------------------------------------------------------------------------------------------------
def plot_cumulative_departure_statistics( xp, times=None, error_model='', use_fg_spread=False, output_path='',
                                          file_type='png', retrieve_only_data=False ) :

    # retrieve individual statistics ...................................................................................
    if times is None :
        times = xp.valid_times['an']
    print('times : ', times)

    istats = []
    times_available = []
    for time in times :
        try :
            stats = get_departure_statistics( xp, time, error_model=error_model, use_fg_spread=use_fg_spread )
            print('got ', time)
        except IOError :
            break
        last_time = time
        istats.append( stats )
        times_available.append( time )

    print()
    print('>>>>>>>>>> PLOTTING CUMULATIVE STATISTICS FOR ', xp.exp_dir)

    # compute cumulative statistics ....................................................................................
    cstats = {}
    # constant scalars or arrays
    for q in ['dep_bins', 'dep_bin_centers', 'delta_dep', 'n_ens', 'error_model', 'refl_bins', 'refl_bin_centers',
              'rsym_bins', 'rsym_bin_centers', 'rdif_bins', 'rdif_bin_centers', 'rnld_bins', 'rnld_bin_centers'] :
        cstats[q] = istats[0][q]
    # additive scalars
    for q in ['n_full', 'n_cs2', 'n_rest', 'n_refl_tot', 'n_refl_max' ] :
        cstats[q] = istats[0][q] + 0
        for i in range(1,len(istats)) :
            cstats[q] += istats[i][q]
        print(q, istats[0][q], cstats[q])
    # additive arrays
    for q in ['fgdepn_hist', 'fgdepn_hist_cs2', 'fgdepn_hist_rest', 'n_refl', 'n_symdif', 'n_dif', 'n_symdif_strict', 'n_symnld',
              'anadepn_hist', 'anadepn_nl_hist',
              'obs_refl_hist', 'fgens_refl_hist', 'anaens_refl_hist', 'anaens_nl_refl_hist' ] :
        if q in istats[0] :
            cstats[q] = istats[0][q] + 0
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]
    # weighted average
    # ...of functions of symmetric reflectance
    for q in ['fgmeandep_std_refl', 'meanfgspread_refl', 'e_o_refl', 'e_desroz_refl', 'e_desroz_nl_refl',
              'spread_refl', 'spread_desroz_refl', 'spread_desroz_nl_refl',
              'anamean_meannlerr_refl', 'anamean_maxnlerr_refl'] :
        if q in istats[0] :
            cstats[q] = istats[0][q]*istats[0]['n_refl']
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]*istats[i]['n_refl']
            cstats[q] /= maximum( cstats['n_refl'], 1 )
            cstats[q][where(cstats['n_refl']<5)] = nan
    # ...of functions of symmetric reflectance and mean fg mean departure
    for q in ['fgmeandep_std_sd', 'meanfgspread_sd', 'meaneo_sd', 'e_desroz_sd', 'std_e_desroz_sd',
              'spread_desroz_sd', 'std_spread_desroz_sd', 'anameanbias_nl_sd', 'anameanerr_nl_sd',
              'fgmean_err_sd', 'anamean_err_sd', 'anameannl_err_sd', 'err_red_lin_sd', 'err_red_nl_sd' ] :
        if q in istats[0] :
            cstats[q] = istats[0][q]*istats[0]['n_symdif']
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]*istats[i]['n_symdif']
            cstats[q] /= maximum( cstats['n_symdif'], 1 )
            cstats[q][where(cstats['n_symdif']<5)] = nan
            cstats[q][where(abs(cstats[q])<1e-13)] = nan
    # ...of functions of asymmetric reflectance
    for q in [ 'fgmean_err_df', 'anamean_err_df', 'anameannl_err_df', 'anamean_err_std_df', 'anameannl_err_std_df',
               'err_red_lin_df', 'err_red_nl_df', 'err_red_lin_std_df', 'err_red_nl_std_df' ] :
        if q in istats[0] :
            cstats[q] = istats[0][q]*istats[0]['n_dif']
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]*istats[i]['n_dif']
            cstats[q] /= maximum( cstats['n_dif'], 1 )
            cstats[q][where(cstats['n_dif']<5)] = nan
    # ...of functions of symmetric reflectance and nl discriminator
    for q in ['fgmeandep_std_sn', 'meanfgspread_sn', 'meaneo_sn',
              'anameanbias_nl_sn', 'anameanerr_nl_sn'] :
        if q in istats[0] :
            cstats[q] = istats[0][q]*istats[0]['n_symnld']
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]*istats[i]['n_symnld']
            cstats[q] /= maximum( cstats['n_symnld'], 1 )
            cstats[q][where(cstats['n_symnld']<5)] = nan
    # unweighted averages :
    for q in ['e_desroz_rsym', 'e_desroz_nl_rsym'] :
        if q in istats[0] :
            cstats[q] = istats[0][q]/len(istats)
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]/len(istats)

    # ...of functions of symmetric reflectance and mean fg mean departure
    for q in ['e_desroz_sd_strict'] :
        if q in istats[0] :
            cstats[q] = istats[0][q]*istats[0]['n_symdif_strict']
            for i in range(1,len(istats)) :
                cstats[q] += istats[i][q]*istats[i]['n_symdif_strict']
            cstats[q] /= maximum( cstats['n_symdif_strict'], 1 )
            #cstats[q][where(cstats['n_symdif_strict']<5)] = nan
            #cstats[q][where(abs(cstats[q])<1e-13)] = nan

    if retrieve_only_data :
        return cstats

    # plot .............................................................................................................

    # plot cumulative statistics
    plot_departure_statistics( cstats, name=xp.exp_dir, output_path=output_path, file_type=file_type,
                               prefix=times[0]+'-'+last_time+'_' )

    # plot statistics for individual times
    for i, time in enumerate(times_available) :
        stats = istats[i]
        for q in ['fgmeandep_std_refl', 'meanfgspread_refl', 'e_o_refl', 'e_desroz_refl', 'e_desroz_nl_refl'] :
            if q in stats :
                stats[q][where(stats['n_refl']==0)] = nan
        plot_departure_statistics( stats, name=xp.exp_dir+' t='+time,output_path=output_path, file_type=file_type,
                                   prefix=time+'_' )


def trafunc(x,a=0.4,b=0.45) :
    return arctanh( (x-a)/b )

def traderiv(x,a=0.4,b=0.45) :
    return (1/b) / ( 1 - ((x-a)/b)**2 )


#-----------------------------------------------------------------------------------------------------------------------
def plot_departure_statistics( stats, output_path='', nonlinear=False, file_type='png', prefix='', name='', use_ana_nl=False, verbose=True ) :

    print('=========== PLOT_DEPARTURE_STATISTICS')

    #print 'plot_departure_statistics : received'
    #for s in stats :
    #    print ' -- ', s, stats[s]

    # spread(fg mean)
    if 'fgmean' in stats :

        print('~~~~ ', type(stats['fgmean']), type(stats['fgspread']))
        print('~~~~ ', stats['fgmean'].shape, stats['fgspread'].shape)
        print('~~~~ ', stats['fgmean'].min(), stats['fgspread'].min())
        print('~~~~ ', stats['fgmean'].mean(), stats['fgspread'].mean())
        print('~~~~ ', stats['fgmean'].max(), stats['fgspread'].max())

        fig, ax = plt.subplots()
        #ax.scatter( x=stats['fgmean'], y=stats['fgspread'], s=10, c='k', alpha=0.3 ) #, 'k', alpha=0.3 )
        ax.plot( stats['fgmean'], stats['fgspread'] )
        ax.grid()
        fig.savefig(prefix+'fgspread_vs_fgmean.'+file_type)
        plt.close(fig)
        if verbose : print(prefix+'fgspread_vs_fgmean.'+file_type)

        fig, ax = plt.subplots()
        ax.plot( trafunc( stats['fgmean'] ), stats['fgspread'] * traderiv( stats['fgmean'] ), 'k', alpha=0.3 )
        ax.grid()
        fig.savefig(prefix+'fgspread_vs_fgmean_transformed.'+file_type)
        plt.close(fig)
        if verbose : print(prefix+'fgspread_vs_fgmean_transformed.'+file_type)

    # normalization factor to convert from histogram to PDF
    nrm = stats['delta_dep'] * stats['n_full'] * stats['n_ens']

    print('normalization : ', stats['delta_dep'], stats['n_full'], stats['n_ens'], ' -> ', nrm)

    # compute gaussian fits -- ICH GLAUBE DAS MACHTE KEINEN SINN. LS 2021

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
    popt, pcov = curve_fit( gaussian, stats['dep_bin_centers'][imin:imax+1], stats['fgdepn_hist_rest'][imin:imax+1], bounds=([1e3,0.1],[1e12,1]) )
    dep_hist_fit = gaussian( stats['dep_bin_centers'], *popt)
    print('gaussian fit parameters : ', popt)

    popt1, pcov1 = curve_fit( gaussian1, stats['dep_bin_centers'][imin:imax+1], stats['fgdepn_hist_rest'][imin:imax+1] )
    dep_hist_fit1 = gaussian1( stats['dep_bin_centers'], *popt1)
    idcs = where(dep_hist_fit1 > nmin)
    stats['dep_hist_fit1'] = dep_hist_fit1

    if False :
        # histogram plot
        fig, ax = plt.subplots()
        ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist']    ,  color='k', label='full (%d obs.)'      % stats['n_full'], linewidth=1   )
        ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_cs2'],  color='b', label='double clear sky (%2.0f%%)' % (100*stats['n_cs2'] / float(stats['n_full']) ) )
        ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_rest'], '-', color='r', label='rest (%2.0f%%)' % (100*stats['n_rest'] / float(stats['n_full']) ) )
        ax.semilogy( stats['dep_bin_centers'][idcs], dep_hist_fit[idcs],  '--', color='#999999', label='w=%4.2f fit'%abs(popt[1]) )
        ax.semilogy( stats['dep_bin_centers'][idcs], dep_hist_fit1[idcs], ':', color='k', label='w=1 fit' )
        ax.legend(frameon=False, fontsize=10)
        fig.suptitle('RAD REFL FG departures for '+ name )
        ax.set_xlabel('H(x)-y / sqrt(e_o**2+spread**2)')
        ax.set_ylabel('#cases')
        ax.grid()
        ax.set_ylim((1,stats['fgdepn_hist'].max()))
        fig.savefig(prefix+'fg_dep_components_hist.'+file_type)
        plt.close(fig)

    # PDF plot
    fig, ax = plt.subplots()
    ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist']     /nrm,  color='k', label='FG full (%d obs.)'      % stats['n_full'], linewidth=1   )
    ax.semilogy( stats['dep_bin_centers'], stats['anadepn_hist']    /nrm,  color='#006600', label='AN', linewidth=1   )
    if use_ana_nl :
        ax.semilogy( stats['dep_bin_centers'], stats['anadepn_nl_hist'] /nrm,  color='#00cc00', label='AN nl. + infl.', linewidth=1 )
    ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_cs2'] /nrm,  color='b', label='FG double clear sky (%2.0f%%)' % (100*stats['n_cs2'] / float(stats['n_full']) ) )
    ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_rest']/nrm,  '-', color='r', label='FG rest (%2.0f%%)' % (100*stats['n_rest'] / float(stats['n_full']) ) )
    #ax.semilogy( stats['dep_bin_centers'][idcs], dep_hist_fit[idcs]/nrm, '--', color='#999999', label='w=%4.2f fit'%abs(popt[1]) )
    #
    # Das ist vermutlich Quatsch:
    #ax.semilogy( stats['dep_bin_centers'], dep_hist_fit1/nrm, ':', color='k', label='w=1 fit' )
    # Und das nicht:
    ax.semilogy( stats['dep_bin_centers'], (stats['n_rest'] / float(stats['n_full'])) * exp(-0.5*stats['dep_bin_centers']**2)/sqrt(2*pi), ':', color='k', label='gaussian dist. * {:.0f}%'.format( 100*stats['n_rest']/float(stats['n_full']) ))
    #
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle('RAD REFL FG departures for '+ name )
    ax.set_xlabel('H(x)-y / sqrt(e_o**2+spread**2)')
    ax.set_ylabel('PDF')
    ax.grid()
    ax.set_ylim((1e-4,3))
    fig.savefig(prefix+'fg_dep_components.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'fg_dep_components.'+file_type)

    # PDF plot (linear scale)
    fig, ax = plt.subplots()
    ax.plot( stats['dep_bin_centers'], stats['fgdepn_hist']     /nrm,  color='k', label='full (%d obs.)'      % stats['n_full'], linewidth=1   )
    ax.plot( stats['dep_bin_centers'], stats['fgdepn_hist_cs2'] /nrm,  color='b', label='double clear sky (%2.0f%%)' % (100*stats['n_cs2'] / float(stats['n_full']) ) )
    ax.plot( stats['dep_bin_centers'], stats['fgdepn_hist_rest']/nrm,  '-', color='r', label='rest (%2.0f%%)' % (100*stats['n_rest'] / float(stats['n_full']) ) )
    #ax.plot( stats['dep_bin_centers'], dep_hist_fit1/nrm, ':', color='k', label='w=1 fit' )
    #ax.plot( stats['dep_bin_centers'],                                             exp(-0.5*stats['dep_bin_centers']**2)/sqrt(2*pi), ':', color='k', label='gaussian dist.' )
    ax.plot( stats['dep_bin_centers'], (stats['n_rest'] / float(stats['n_full'])) * exp(-0.5*stats['dep_bin_centers']**2)/sqrt(2*pi), ':', color='k', label='gaussian dist. * {:.0f}%'.format(100*stats['n_rest'] / float(stats['n_full'])) )
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle('RAD REFL FG departures for '+ name )
    ax.set_xlabel('H(x)-y / sqrt(e_o**2+spread**2)')
    ax.set_ylabel('PDF')
    ax.grid()
    #ax.set_ylim((1e-4,3))
    fig.savefig(prefix+'fg_dep_components_linscl.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'fg_dep_components_linscl.'+file_type)

    # Similar to Harnisch+2016 Fig. 7 with reflectance instead of cloud impact
    fig, ax = plt.subplots()
    ax.plot( stats['refl_bin_centers'], stats['fgmeandep_std_refl'], 'k',   label='std(FG mean departures)' )
    ax.plot( stats['refl_bin_centers'], stats['meanfgspread_refl'],  '--b', label='mean spread'  )
    #ax.plot( stats['refl_bin_centers'], sqrt( maximum( fgdep_std_refl**2 - fg_std_refl**2, 0 ) ), ':r', label='e_o estimate'  )
    ax.legend(loc='upper left')
    fig.savefig(prefix+'fg_dep_std_refl.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'fg_dep_std_refl.'+file_type)

    # Desroziers error estimate as function of reflectance
    fig, ax = plt.subplots()
    ax.plot( stats['refl_bin_centers'], stats['e_desroz_refl'], 'k',   label='desroziers est.' )
    if nonlinear :
        ax.plot( stats['refl_bin_centers'], stats['e_desroz_nl_refl'], 'k', alpha=0.5, label='desroziers est. (nl)' )
    ax.plot( stats['refl_bin_centers'], stats['e_o_refl'],      'b', label='assumed error'  )
    ax.plot( stats['refl_bin_centers'], stats['n_refl']/float(10*stats['n_refl'].max()),      'r', label='num. fraction / 10'  )

    ax.plot( stats['rsym_bin_centers'], stats['e_desroz_rsym'], 'g',   label='denoised desroziers est.' )
    if nonlinear :
        ax.plot( stats['rsym_bin_centers'], stats['e_desroz_nl_rsym'], 'g', alpha=0.5, label='denoised desroziers est. (nl)' )

    #x = linspace(0,1,100)
    #ax.plot( x, 0.07*exp( -((x-0.5)/0.3)**2 ), '--g' )

    ax.text( 0.95, 0.01, 'fraction active: %5.3f (%d of %d)' % (stats['n_refl_tot']/float(stats['n_refl_max']),
                                                                stats['n_refl_tot'], stats['n_refl_max']), ha='right' )
    ax.set_ylim((0,0.31))
    ax.set_xlim((0,1))
    ax.grid()
    ax.legend(loc='upper left', frameon=False, fontsize=10, title=name )
    fig.savefig(prefix+'desroz_refl.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'desroz_refl.'+file_type)

    # Desroziers spread estimate as function of reflectance
    fig, ax = plt.subplots()
    ax.plot( stats['refl_bin_centers'], stats['spread_desroz_refl'], 'k',   label='desroziers est.' )
    if nonlinear :
        ax.plot( stats['refl_bin_centers'], stats['spread_desroz_nl_refl'], 'k', alpha=0.5, label='desroziers est. (nl)' )
    ax.plot( stats['refl_bin_centers'], stats['spread_refl'],      'b', label='actual FG spread'  )
    ax.plot( stats['refl_bin_centers'], stats['n_refl']/float(10*stats['n_refl'].max()),      'r', label='num. fraction / 10'  )
    ax.text( 0.95, 0.01, 'fraction active: %5.3f (%d of %d)' % (stats['n_refl_tot']/float(stats['n_refl_max']),
                                                                stats['n_refl_tot'], stats['n_refl_max']), ha='right' )
    ax.set_ylim((0,0.31))
    ax.set_xlim((0,1))
    ax.grid()
    ax.legend(loc='upper left', frameon=False, fontsize=10, title=name )
    fig.savefig(prefix+'spread_desroz_refl.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'spread_desroz_refl.'+file_type)

    # reflectance histograms
    fig, ax = plt.subplots()
    ax.plot( stats['refl_bin_centers'], stats['obs_refl_hist'], 'k', label='O' )
    ax.plot( stats['refl_bin_centers'], stats['fgens_refl_hist'], 'b', label='B' )
    ax.plot( stats['refl_bin_centers'], stats['anaens_refl_hist'], color='#990000', label='A (est.)' )
    if use_ana_nl :
        ax.plot( stats['refl_bin_centers'], stats['anaens_nl_refl_hist'], 'r', label='A (nl.+infl.)' )
    ax.grid()
    ax.legend(loc='upper right', frameon=False, fontsize=10, title=name )
    fig.savefig(prefix+'refl_hist.'+file_type)
    plt.close(fig)
    if verbose : print(prefix+'refl_hist.'+file_type)

    if nonlinear :
        # nonlinearity error as function symmetric reflectance
        fig, ax = plt.subplots()
        ax.plot( stats['refl_bin_centers'], stats['anamean_meannlerr_refl'], 'k', label='mean' )
        ax.plot( stats['refl_bin_centers'], stats['anamean_maxnlerr_refl'], '--k', label='max' )
        ax.grid()
        ax.legend(loc='upper left', frameon=False, fontsize=10, title=name )
        fig.savefig(prefix+'anamean_nlerr_refl.'+file_type)
        plt.close(fig)
        if verbose : print(prefix+'anamean_nlerr_refl.'+file_type)

    # error reduction as function of O-B
    
    #print '========>>> ', stats['n_symdif'].shape, stats['anameannl_err_sd'].shape, stats['rdif_bins'].shape, stats['rsym_bins'].shape
    
    n_dif = stats['n_symdif'].sum(axis=0)
    idcs = where( n_dif >= 5 )

    if len(idcs[0]) < 2 :
        print('NO N_DIF >= 5!')
    else :
        print('There are {} of {} cases with n_dif >= 5'.format(len(idcs[0]),n_dif.size))

        fge = (stats['fgmean_err_sd']*stats['n_symdif']).sum(axis=0)
        fge[idcs] /= n_dif[idcs]

        ale = (stats['anamean_err_sd']*stats['n_symdif']).sum(axis=0)
        ale[idcs] /= n_dif[idcs]

        if use_ana_nl :
            ane = (stats['anameannl_err_sd']*stats['n_symdif']).sum(axis=0)
            ane[idcs] /= n_dif[idcs]

        rdif_centers = 0.5*( stats['rdif_bins'][1:] + stats['rdif_bins'][:-1] )


        #tPlot, axes = plt.subplots(
        #            nrows=4, ncols=1, sharex=True, sharey=False,
        #            gridspec_kw={'height_ratios':[2,2,1,1]}
        #            )
        
        #tPlot.suptitle('node', fontsize=20)


        fig, ax = plt.subplots( figsize=(10,13), nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios':[3,1]} )
        m = 0.8
        c = '#009999'
        ax[0].plot( (-m,m), (0,0), c, linewidth=1 )
        ax[0].plot( (0,0), (-m,m), c, linewidth=0.5 )
        ax[0].plot( (-m,m), (-m,m), c, linewidth=1 )
        ax[0].plot( (-m,m), (-m*0.5,m*0.5), c, linewidth=0.5 )
        ax[0].plot( (-m,m), (-m*0.75,m*0.75), c, linewidth=0.5 )
        ax[0].text( -m*0.95, -m*0.95, 'create clouds', color=c)
        ax[0].text( m*0.95, -m*0.95, 'destroy clouds', color=c, ha='right')
        #ax[0].plot( rdif_centers[idcs], fge[idcs], 'k', label='B-O' )
        
        #ax[0].plot( rdif_centers[idcs], ale[idcs], '#990000', label='A-O lin.' )
        ax[0].plot( rdif_centers[idcs], stats['anamean_err_df'][idcs], '#990099', label='A-O lin.', linewidth=2 )

        #print('>>>>', stats['anamean_err_df'][idcs].min(), stats['anamean_err_df'][idcs].mean(), stats['anamean_err_df'][idcs].max())
        #print('>>>>', stats['anamean_err_df'][idcs].shape, rdif_centers[idcs].shape)
        ax[0].scatter( rdif_centers[idcs]-0.005, stats['anamean_err_df'][idcs], c=['#990099'] ) #, edgecolor='' )
        
        if use_ana_nl :
            #ax[0].plot( rdif_centers[idcs], ane[idcs], 'r', label='A-O nl.' )
            ax[0].plot( rdif_centers[idcs], stats['anameannl_err_df'][idcs], 'r', label='A-O nl.', linewidth=2 )
            ax[0].scatter( rdif_centers[idcs]+0.005, stats['anameannl_err_df'][idcs], c='r', edgecolor='' )


        for i in range(rdif_centers.size) :
            if n_dif[i] >= 5 :
                ax[0].plot( [rdif_centers[i]-0.005]*2, stats['anamean_err_df'][i]+r_[-0.5,0.5]*stats['anamean_err_std_df'][i], '#990099' )
                if use_ana_nl :
                    ax[0].plot( [rdif_centers[i]+0.005]*2, stats['anameannl_err_df'][i]+r_[-0.5,0.5]*stats['anameannl_err_std_df'][i], 'r' )
        
        ax[0].set_xlim((-0.8,0.8))
        ax[0].set_ylim((-0.8,0.8))
        ax[0].grid()
        ax[0].legend()
        ax[0].set_ylabel('X-O')
        ax[1].plot( rdif_centers, n_dif )
        ax[1].set_xlabel('#obs.')
        ax[1].set_xlabel('B-O')
        fig.savefig(prefix+'_error_reduction_sd.'+file_type)
        plt.close(fig)
        if verbose : print(prefix+'_error_reduction_sd.'+file_type)

    # error reduction as function of B-O
    idcs = where( (n_dif >= 5) & (abs(rdif_centers)>1e-3) )
    if len(idcs[0]) < 2 :
        print('NO N_DIF >= 5 AND RDIF_CENTERS>0')
    else :
        fig, ax = plt.subplots( figsize=(10,13), nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios':[3,1]} )
        m = 0.8
        c = '#009999'
        #ax[0].plot( (-m,m), (0,0), c, linewidth=1 )
        #ax[0].plot( (0,0), (-m,m), c, linewidth=0.5 )
        #ax[0].plot( (-m,m), (-m,m), c, linewidth=1 )
        #ax[0].plot( (-m,m), (-m*0.5,m*0.5), c, linewidth=0.5 )
        #ax[0].plot( (-m,m), (-m*0.75,m*0.75), c, linewidth=0.5 )
        ax[0].text( -m*0.95, 0.05, 'create clouds', color=c)
        ax[0].text( m*0.95, 0.05, 'destroy clouds', color=c, ha='right')

        print('SHAPES S S S ', rdif_centers[idcs].shape, stats['err_red_lin_df'][idcs].shape, rdif_centers.shape, stats['err_red_lin_df'].shape)
        ax[0].plot( rdif_centers[idcs], stats['err_red_lin_df'][idcs], '#990099', label='lin.', linewidth=2 )
        ax[0].scatter( rdif_centers[idcs]-0.005, stats['err_red_lin_df'][idcs], c=['#990099'] ) #, edgecolor='' )

        if use_ana_nl :
            ax[0].plot( rdif_centers[idcs], stats['err_red_nl_df'][idcs], 'r', label='nl.', linewidth=2 )
            ax[0].scatter( rdif_centers[idcs]+0.005, stats['err_red_nl_df'][idcs], c='r', edgecolor='' )
        
        for i in range(rdif_centers.size) :
            if n_dif[i] >= 5 :
                ax[0].plot( [rdif_centers[i]-0.005]*2, stats['err_red_lin_df'][i]+r_[-0.5,0.5]*stats['err_red_lin_std_df'][i], '#990099' )
                if use_ana_nl :
                    ax[0].plot( [rdif_centers[i]+0.005]*2, stats['err_red_nl_df'][i]+r_[-0.5,0.5]*stats['err_red_nl_std_df'][i], 'r' )

        ax[0].set_xlim((-0.8,0.8))
        ax[0].set_ylim((-0.1,0.8))
        ax[0].grid()
        ax[0].legend()
        ax[0].set_ylabel('error reduction')
        ax[1].plot( rdif_centers, n_dif )
        ax[1].set_xlabel('#obs.')
        ax[1].set_xlabel('B-O')
        fig.savefig(prefix+'_error_reduction2_sd.'+file_type)
        plt.close(fig)

    # error reduction as function of (B+O,B-O)
    fig, ax = plt.subplots()
    im = ax.imshow( stats['err_red_lin_sd'], origin='lower', interpolation='nearest', aspect=2.0,
                    extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                             stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                    vmin=-0.2, vmax=1.0, cmap='gist_stern' ) 
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'err_red_lin_sd.'+file_type)
    plt.close(fig)

    if use_ana_nl :
        fig, ax = plt.subplots()
        im = ax.imshow( stats['err_red_nl_sd'], origin='lower', interpolation='nearest', aspect=2.0,
                        extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                        vmin=-0.2, vmax=1.0, cmap='gist_stern' )
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+'err_red_nl_sd.'+file_type)
        plt.close(fig)
    
    # mean desroziers error estimate as a function of O-B and O+B
    nmin = 5
    for quan in ['e','spread'] :
        e_desroz_sd     = stats[quan+'_desroz_sd'] + 0
        std_e_desroz_sd = stats['std_'+quan+'_desroz_sd'] + 0
        e_desroz_sd[     where( stats['n_symdif'] < nmin ) ] = nan
        std_e_desroz_sd[ where( stats['n_symdif'] < nmin ) ] = nan

        fig, ax = plt.subplots()
        if True :
            im = ax.imshow( e_desroz_sd, origin='lower', interpolation='nearest', aspect=2.0,
                            extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                     stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                            vmin=0, vmax=0.5, cmap='gist_stern' ) # RdBu
        else :
            im = ax.contourf( stats['rdif_bin_centers'], stats['rsym_bin_centers'], e_desroz_sd,
                              levels=linspace(-0.1,0.1,11) )
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+quan+'_desroz_sd.'+file_type)
        plt.close(fig)

        e_assumed = stats['meaneo_sd'] if quan == 'e' else  stats['meanfgspread_sd']


        fig, ax = plt.subplots()
        im = ax.imshow( e_assumed, origin='lower', interpolation='nearest', aspect=2.0,
                            extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                     stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                            vmin=0, vmax=0.5, cmap='gist_stern' ) # RdBu
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+quan+'_actual_sd.'+file_type)
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow( e_desroz_sd/e_assumed, origin='lower', interpolation='nearest', aspect=2.0,
                            extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                     stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                            vmin=0, vmax=2.0, cmap='gist_stern' ) # RdBu
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+quan+'_desroz_normalized_sd.'+file_type)
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow( e_desroz_sd-e_assumed, origin='lower', interpolation='nearest', aspect=2.0,
                            extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                     stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                            vmin=-0.2, vmax=0.2, cmap='RdBu' )
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+quan+'_desroz_diff_sd.'+file_type)
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow( (e_desroz_sd-e_assumed)/e_assumed, origin='lower', interpolation='nearest', aspect=2.0,
                            extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                     stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                            vmin=-0.2, vmax=0.2, cmap='RdBu' )
        ax.set_ylabel('(B+O)/2')
        ax.set_xlabel('B-O')
        plt.colorbar(im)
        fig.savefig(prefix+quan+'_desroz_reldiff_sd.'+file_type)
        plt.close(fig)


    # number of cases
    fig, ax = plt.subplots()
    im = ax.imshow( stats['n_symdif'], origin='lower', interpolation='nearest', aspect=2.0,
                    extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                             stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                    vmin=0, cmap='terrain' )
    ax.plot( (stats['n_symdif']*stats['rdif_bin_centers']).sum(axis=1)/(stats['n_symdif'].sum(axis=1)+1e-5), stats['rsym_bin_centers'], 'r', alpha=0.5, linewidth=2 )
    ax.grid()
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'n_symdif.'+file_type)
    plt.close(fig)

    # 'strict' statistics
    fig, ax = plt.subplots()
    im = ax.imshow( stats['e_desroz_sd_strict'], origin='lower', interpolation='nearest', aspect=2.0,
                    extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                             stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                    vmin=0, vmax=0.5, cmap='gist_stern' ) # RdBu
    ax.grid()
    ax.set_ylabel('(B+O)/2')
    ax.set_xlabel('B-O')
    plt.colorbar(im)
    fig.savefig(prefix+'e_desroz_sd_strict.'+file_type)
    plt.close(fig)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    for i in range(len(stats['rsym_bin_centers'])) :
        ax[0].scatter( stats['rdif_bin_centers'], stats['e_desroz_sd_strict'][i,:], color=colseq(i,len(stats['rsym_bins'])) )
    x = linspace(-0.4,0.4,50)
    ax[0].plot( x, 0.1 + 0.25*(x/0.4)**2, ':k' )
    ax[0].set_xlabel('B-O')
    ax[0].set_xlim((-0.5,0.5))
    ax[0].set_ylim((0,0.4))
    for i in range(len(stats['rdif_bin_centers'])) :
        ax[1].scatter( stats['rsym_bin_centers'], stats['e_desroz_sd_strict'][:,i], color=colseq(i,len(stats['rdif_bins'])) )

    x = linspace(0.0,0.4,50)
    ax[1].plot( x, x, ':k' )
    ax[1].set_xlabel('(B+O)/2')
    ax[1].set_ylim((0,0.4))
    fig.savefig(prefix+'e_desroz_sd_strict_curves.'+file_type)
    plt.close(fig)



    # curves
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    for i in range(len(stats['rsym_bin_centers'])) :
        ax[0].plot( stats['rdif_bin_centers'], e_desroz_sd[i,:], color=colseq(i,len(stats['rsym_bins'])) )
        ax[0].fill_between( stats['rdif_bin_centers'],
                             e_desroz_sd[i,:]-0.5*std_e_desroz_sd[i,:],
                             e_desroz_sd[i,:]+0.5*std_e_desroz_sd[i,:], color=colseq(i,len(stats['rsym_bins'])), alpha=0.25 )
    x = linspace(-0.4,0.4,50)
    ax[0].plot( x, 0.1 + 0.25*(x/0.4)**2, ':k' )
    ax[0].set_xlabel('B-O')
    ax[0].set_xlim((-0.5,0.5))
    ax[0].set_ylim((0,0.4))
    for i in range(len(stats['rdif_bin_centers'])) :
        ax[1].plot( stats['rsym_bin_centers'], e_desroz_sd[:,i], color=colseq(i,len(stats['rdif_bins'])) )
        ax[1].fill_between( stats['rsym_bin_centers'],
                             e_desroz_sd[:,i]-0.5*std_e_desroz_sd[:,i],
                             e_desroz_sd[:,i]+0.5*std_e_desroz_sd[:,i], color=colseq(i,len(stats['rdif_bins'])), alpha=0.25 )

    x = linspace(0.0,0.4,50)
    ax[1].plot( x, x, ':k' )
    ax[1].set_xlabel('(B+O)/2')
    ax[1].set_ylim((0,0.4))
    fig.savefig(prefix+'e_desroz_sd_curves.'+file_type)
    plt.close(fig)


    if nonlinear :
        # nonlinearity error of analysis ensemble mean as a function of O-B and O+B
        for quan in ['anameanerr_nl_sd', 'anameanbias_nl_sd'] :
            fig, ax = plt.subplots()
            if True :
                im = ax.imshow( stats[quan], origin='lower', interpolation='nearest', aspect=2.0,
                                extent=[ stats['rdif_bins'].min(), stats['rdif_bins'].max(),
                                         stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                                vmin=-0.05 if 'bias' in quan else 0, vmax=0.05, cmap='RdBu_r' if 'bias' in quan else 'terrain' )
            else :
                im = ax.contourf( stats['rdif_bin_centers'], stats['rsym_bin_centers'], stats[quan],
                                  levels=linspace(-0.1,0.1,11) )
            ax.set_ylabel('(B+O)/2')
            ax.set_xlabel('B-O')
            plt.colorbar(im)
            fig.savefig(prefix+quan+'.'+file_type)
            plt.close(fig)

        # nonlinearity error of analysis ensemble mean as a function of |O-B|-spread and O+B
        for quan in ['anameanerr_nl_sn', 'anameanbias_nl_sn'] :
            fig, ax = plt.subplots()
            if True :
                im = ax.imshow( stats[quan], origin='lower', interpolation='nearest', aspect=1.0,
                                extent=[ stats['rnld_bins'].min(), stats['rnld_bins'].max(),
                                         stats['rsym_bins'].min(), stats['rsym_bins'].max() ],
                                vmin=-0.05 if 'bias' in quan else 0, vmax=0.05, cmap='RdBu_r' if 'bias' in quan else 'terrain' )
            else :
                im = ax.contourf( stats['rnld_bin_centers'], stats['rsym_bin_centers'], stats[quan],
                                  levels=linspace(-0.1,0.1,11) )
            ax.set_ylabel('(B+O)/2')
            ax.set_xlabel('|B-O|-spread')
            plt.colorbar(im)
            fig.savefig(prefix+quan+'.'+file_type)
            plt.close(fig)

        #fig, ax = plt.subplots()
        #abs(stats['anameanbias_nl_sd'])

def gaussian( x, a, b ) :
    return a*exp( -(x/b)**2 )

def gaussian1( x, a ) :
    return a*exp( -x**2 )


#-----------------------------------------------------------------------------------------------------------------------
def compare_cumulative_departure_statistics( xps, cstats, output_path='', file_type='pdf', colors=None, names=None ) :

    from kendapy.colseq import colseq


    # Desroziers e_o estimate comparison ...............................................................................

    rmax=0.8
    emax=0.32
    fig, ax = plt.subplots(figsize=(4,3))
    for ixp, xp in enumerate(xps) :
        stats = cstats[ixp]
        print( xp.exp_dir, cstats[ixp].keys() )

        lbl = xp.exp_dir if names is None else names[ixp]
        ax.plot( stats['refl_bin_centers'], stats['e_desroz_refl'], color=colseq(ixp,len(xps),no_green=True), label=lbl )
        try :
            ax.plot( (0,rmax), float(xp.settings['VISOP_ERROR'])*ones(2), '--', color=colseq(ixp,len(xps),no_green=True) )
        except ValueError:
            print( "VISOP_ERROR is not a float -> I won't plot it..." )

    ax.plot( (0,emax), (0,emax), ':k', linewidth=0.5 )
    ax.set_xlim((0,rmax))
    ax.set_ylim((0,emax))
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$(R_{\rm O}+R_{\rm B})/2$')
    ax.set_ylabel(r'$\sigma_{\rm O}$')
    fname = '_'.join([xp.exp_dir for xp in xps])
    fig.savefig('desroziers_comparison_{}.{}'.format(fname,file_type), bbox_inches='tight')
    plt.close(fig)


    # FG departure comparison ..........................................................................................

    fig, ax = plt.subplots(figsize=(4,3))
    for ixp, xp in enumerate(xps) :
        col  = colseq(ixp,len(xps),no_green=True) if colors is None else colors[ixp]
        name = xp.exp_dir                           if names  is None else names[ixp]
        stats = cstats[ixp]
        print( xp.exp_dir, cstats[ixp].keys() )
        nrm = stats['delta_dep'] * stats['n_full'] * stats['n_ens']
        ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist']     /nrm,  color=col, label='%s (%d obs.)' % (name,stats['n_full']), linewidth=1 )
        #ax.semilogy( stats['dep_bin_centers'], stats['anadepn_hist']    /nrm,  color='#006600', label='ana est.', linewidth=1   )
        #ax.semilogy( stats['dep_bin_centers'], stats['anadepn_nl_hist'] /nrm,  color='#00cc00', label='ana nl. + infl.', linewidth=1 )
        #ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_cs2'] /nrm,  color='b', label='double clear sky (%2.0f%%)' % (100*stats['n_cs2'] / float(stats['n_full']) ) )
        #ax.semilogy( stats['dep_bin_centers'], stats['fgdepn_hist_rest']/nrm,  '-', color='r', label='rest (%2.0f%%)' % (100*stats['n_rest'] / float(stats['n_full']) ) )
        #ax.semilogy( stats['dep_bin_centers'][idcs], dep_hist_fit[idcs]/nrm, '--', color='#999999', label='w=%4.2f fit'%abs(popt[1]) )

        if ixp == len(xps)-1 :

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
            popt, pcov = curve_fit( gaussian, stats['dep_bin_centers'][imin:imax+1], stats['fgdepn_hist_rest'][imin:imax+1], bounds=([1e3,0.1],[1e12,1]) )
            dep_hist_fit = gaussian( stats['dep_bin_centers'], *popt)
            print('gaussian fit parameters : ', popt)

            popt1, pcov1 = curve_fit( gaussian1, stats['dep_bin_centers'][imin:imax+1], stats['fgdepn_hist_rest'][imin:imax+1] )
            dep_hist_fit1 = gaussian1( stats['dep_bin_centers'], *popt1)
            idcs = where(dep_hist_fit1 > nmin)

            ax.semilogy( stats['dep_bin_centers'], dep_hist_fit1/nrm, ':', color='k', label='gaussian fit') # for '+name )

    ax.legend(frameon=False, fontsize=10, title='First guess departures for '+xp.exp_dir)
    ax.set_xlabel(r'(O-B) / $\sqrt{\sigma_{\rm O}^2+\sigma_{\rm B}^2}$')
    ax.set_ylabel('PDF')
    ax.grid()
    ax.set_ylim((1e-4,3))
    ax.set_xlim((-3,3))
    fname = '_'.join([xp.exp_dir for xp in xps])
    fig.savefig('fg_dep_comparison_{}.{}'.format(fname,file_type),  bbox_inches='tight')
    plt.close(fig)







#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = define_parser()
    args = parser.parse_args()

    # process all log files
    cstats = []
    xps    = []
    for xp_path in args.xp_path :

        print('processing '+xp_path)

        #xp = Experiment(xp_path)
        xp = BacyExp(xp_path)
        xp.info()
        #print('experiment %s : %s #members, first fcst start time %s, last analysis time %s' % ( \
        #       xp.settings['exp'], xp.settings['N_ENS'], xp.fcst_start_times[0], xp.veri_times[-1] ))
        xps.append(xp)
        print()

        # set some default values

        if args.output_path != '' :
            #if args.output_path != 'auto' :
            output_path = args.output_path+'/'
            #else :
            #    output_path = xp.settings['PLOT_DIR']+'/departures/'
            if not os.path.exists(output_path) :
                os.system('mkdir '+output_path)
        else :
            output_path = ''

        #if (args.start_time != '') or (args.end_time != '') or (args.delta_time != '') :
        #    start_time = args.start_time if args.start_time != '' else xp.veri_times[0]
        #    end_time   = args.end_time   if args.end_time   != '' else xp.veri_times[-1]
        #    delta_time = args.delta_time if args.delta_time != '' else int(xp.settings['ASSINT'])
        #    times = time_range( start_time, end_time, delta_time  )
        #else :
        #    times = xp.veri_times[:-2]

        times = xp.valid_times['an']
        times = [ t for t in times if (args.start_time is None or (str2t(t) >= str2t(args.start_time))) and \
                                      (args.end_time   is None or (str2t(t) <= str2t(args.end_time))) ]

        start_daily = to_timedelta(args.start_daily)
        end_daily   = to_timedelta(args.end_daily)
        times = [ t for t in times if  (to_datetime(t) >= midnight(to_datetime(t)) + start_daily) and \
                                       (to_datetime(t) <= midnight(to_datetime(t)) + end_daily) ]
        print('--> times ', times)                                            

        #deconstruct_departures( xp, times=times, error_model=args.error_model,
        #                        epsilon=args.e_o if args.e_o > 0 else None,
        #                        output_path=output_path, file_type=args.file_type )
        cstats.append( plot_cumulative_departure_statistics( xp, times=times, error_model=args.error_model, use_fg_spread=args.use_fg_spread,
                                output_path=output_path, file_type=args.file_type, retrieve_only_data=args.compare ) )

    if args.compare :
        compare_cumulative_departure_statistics( xps, cstats, output_path=output_path, file_type=args.file_type,
                                                 colors=None if args.colors=='' else args.colors.split(','),
                                                 names=None if args.names=='' else args.names.split(','))



