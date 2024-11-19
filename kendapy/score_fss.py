#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . S C O R E _ F S S
#  compute Fractions (skill) score and related quantities
#
#  Almost completely adapted from
#  Faggian, Roux, Steinle, Ebert (2015) "Fast calculation of the fractions skill score"
#  MAUSAM, 66, 3, 457-466
#
#"""
#.. module:: score_fss
#:platform: Unix
#:synopsis: Compute the fraction skill score (2D).
#.. moduleauthor:: Nathan Faggian <n.faggian@bom.gov.au>
#"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from scipy import signal

def compute_integral_table(field) :
    return field.cumsum(1).cumsum(0)

def fourier_filter(field, n) :
    return signal.fftconvolve(field, np.ones((n, n)))

def integral_filter(field, n, table=None, periodic=False) :
    """
    Fast summed area table version of the sliding accumulator.
    :param field: nd-array of binary hits/misses.
    :param n: window size.
    :param periodic: if True, assume periodic boundary conditions
    """
    w = n // 2
    if w < 1. :
        return field.astype(np.int64)
    if table is None:
        table = compute_integral_table(field)

    r, c = np.mgrid[ 0:field.shape[0], 0:field.shape[1] ]
    r = r.astype(np.int)
    c = c.astype(np.int)
    w = np.int(w)
    integral_table = np.zeros(field.shape).astype(np.int64)

    if periodic :
        rshift = [ -field.shape[0], 0, field.shape[0] ]
        cshift = [ -field.shape[1], 0, field.shape[1] ]
    else :
        rshift = [0]
        cshift = [0]

    for rs in rshift :
        for cs in cshift :
            r0, c0 = (np.clip(r+rs - w, 0, field.shape[0] - 1), np.clip(c+cs - w, 0, field.shape[1] - 1))
            r1, c1 = (np.clip(r+rs + w, 0, field.shape[0] - 1), np.clip(c+cs + w, 0, field.shape[1] - 1))
            integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape))
            integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape))
            integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape))
            integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape))

    return integral_table

def fourier_fss(fcst, obs, threshold, window) :
    """
    Compute the fraction skill score using convolution.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = fourier_filter( fcst > threshold, window)
    ohat = fourier_filter( obs  > threshold, window)
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def fss(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None, periodic=False):
    """
    Compute the fraction skill score using summed area tables .
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = integral_filter( fcst > threshold, window, fcst_cache, periodic=periodic )
    ohat = integral_filter( obs  > threshold, window, obs_cache, periodic=periodic  )

    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def fss_frame(fcst, obs, windows, levels, periodic=False):
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: list, dataframes of the FSS: numerator,denominator and score.
    """
    num_data, den_data, fss_data = [], [], []
    for level in levels:
        ftable = compute_integral_table( fcst > level )
        otable = compute_integral_table( obs  > level )
        _data = [fss(fcst, obs, level, w, ftable, otable, periodic=periodic) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])

    return ( pd.DataFrame(num_data, index=levels, columns=windows),
             pd.DataFrame(den_data, index=levels, columns=windows),
             pd.DataFrame(fss_data, index=levels, columns=windows))

### added by L.Scheck 2018.1 : #########################################################################################

def fss_dict(fcst, obs, windows, levels, believable_scale=False, target=False, periodic=False ) :
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: dictionary containting nd-array of the FSS: numerator,denominator and score (dim 0: levels, dim 1: windows).
    """
    num_data, den_data, fss_data = [], [], []
    for level in levels:
        ftable = compute_integral_table( fcst > level )
        otable = compute_integral_table( obs  > level )
        _data = [fss(fcst, obs, level, w, ftable, otable, periodic=periodic) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])

    res = { 'num':np.array(num_data), 'den':np.array(den_data), 'fss':np.array(fss_data), 'levels':levels, 'windows':windows }

    if target or believable_scale :
        fss_random,  fss_target = fss_random_target( obs, levels )
        res.update({ 'fss_random':fss_random, 'fss_target':fss_target })

    if believable_scale :
        belscl = compute_belscl( res['fss'], levels, windows, fss_target)
        res.update({ 'belscl':belscl })

    return res

def fss_ens_dict( ens, obs, windows, levels, believable_scale=False, target=False, periodic=False, ens_mean=False ) :
    """
    Apply fss_dict to all members of an ensemble, compute total fss.
    :param ens: nd-array, ensemble of forecast fields. Dimension 0 = ensemble dimension.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: dictionary containting nd-array of the FSS: numerator,denominator and score + data for individual members.
    """

    #print 'fss_en_dict input : ', ens.shape, obs.shape, windows, levels

    fss_members = []
    for m in range(ens.shape[0]) :
        fss_members.append( fss_dict( ens[m,...], obs, windows, levels, periodic=periodic ) )

    if ens_mean :
        fss = 1. - np.array( [ f['fss'] for f in fss_members ] ).sum(axis=0)
        res = {  'fss':fss, 'levels':levels, 'windows':windows, 'members':fss_members }

    else :
        num =  np.array( [ f['num'] for f in fss_members ] ).sum(axis=0) # sum over ensemble dimension
        den =  np.array( [ f['den'] for f in fss_members ] ).sum(axis=0)
        fss = 1. - num/den
        res = { 'num':num, 'den':den, 'fss':fss, 'levels':levels, 'windows':windows, 'members':fss_members }

    if target or believable_scale :
        fss_random,  fss_target = fss_random_target( obs, levels )
        res.update({ 'fss_random':fss_random, 'fss_target':fss_target })

    if believable_scale :
        belscl = compute_belscl(fss, levels, windows, fss_target)
        res.update({ 'belscl':belscl })

    return res

def fss_random_target( obs, levels ) :
    if np.isscalar(levels) :
        fss_random = np.count_nonzero( obs > levels ) / float(obs.size)
        fss_target = 0.5 + 0.5*fss_random
    else :
        fss_random = []
        fss_target = []
        for l in levels :
            fss_random.append( np.count_nonzero( obs > l ) / float(obs.size) )
            fss_target.append( 0.5 + 0.5*fss_random[-1] )
        fss_random = np.array(fss_random)
        fss_target = np.array(fss_target)
    return fss_random,  fss_target

def compute_belscl( fss, levels, windows, fss_target ) :
    """Compute believable scale, i.e. the scale where the fss exceeds the target fss"""

    belscl = np.zeros(len(levels))
    for ilev in range(len(levels)) :
        for iwin in range(len(windows)) :
            belscl[ilev] = -1
            if fss[ilev,iwin] > fss_target[ilev] :
                if iwin > 0 :
                    # determine crossing point using linear interpolation
                    sigma = (fss[ilev,iwin] - fss[ilev,iwin-1])/(windows[iwin]-windows[iwin-1])
                    belscl[ilev] = windows[iwin] - (fss[ilev,iwin] - fss_target[ilev])/sigma
                else :
                    belscl[ilev] = windows[iwin]
                break
    return belscl
