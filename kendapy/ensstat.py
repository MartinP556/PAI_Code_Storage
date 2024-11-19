#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . E N S S T A T
#  compute ensemble statistics
#
#  2017.4 L.Scheck 

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.ma as ma

def desroz( dob, doa, diagonal_only=False ) :
    """Compute Desroziers (2005) estimate for R matrix
       dob : innovation = background departures (obs - fg)
       doa : analysis departures (obs - ana)
       diagonal_only == True --> return only diagonal elements
    """

    #print 'desroz input shapes ', dob.shape, doa.shape

    if diagonal_only :
        retval = (dob*doa).mean(axis=0)

    else :
        n_ens, n_obs = dob.shape
        retval = np.zeros((n_obs,n_obs))
        for i in range(n_ens) :
            retval += np.outer(dob[i,:]*doa[i,:])

    return retval

def rankhist( obs, ens ) :
    """Compute rank histogram"""

    n_ens, n_obs = ens.shape

    # put ensemble and observations into one numpy array
    ensobs = np.zeros((n_ens+1,n_obs))
    ensobs[0,:]  = obs
    ensobs[1:,:] = ens

    # combine numpy functions to compute the rank histogram
    bins = np.arange(1,n_ens+3)-0.5
    rankhist, edges = np.histogram( np.argmin( np.argsort(ensobs,axis=0),axis=0)+1, bins=bins )

    return rankhist


def brier_score( obs, ens, thres, components=False, n_prob=10 ) :
    """Computes Brier score for observations obs, ensemble of model equivalents ens and threshold thres."""

    # make obs binary
    ob = np.zeros(obs.shape, dtype=obs.dtype)
    ob[ np.where( obs > thres ) ] = 1

    # compute probabilities from ensemble
    eb = np.zeros( ens.shape, dtype=ens.dtype )
    eb[ np.where( ens > thres ) ] = 1
    pb = eb.mean(axis=0) # ensemble dimension = first dimension

    # compute brier score
    bs = ((pb - ob)**2).mean()

    if components :

        # uncertainty
        obmean = ob.mean()
        unc = obmean * ( 1.0 - obmean )

        # define probability classes
        n_class      = np.zeros(n_prob)
        pb_class     = np.zeros(n_prob)
        obmean_class = np.zeros(n_prob)
        for i in range(n_prob) :
            pbmin = float(i  )/n_prob
            pbmax = float(i+1)/n_prob
            pb_class[i] = 0.5*( pbmin + pbmax )

            # select observations falling into class i
            if i == 0 :
                idcs = np.where( (pb >= pbmin) & (pb <= pbmax) )
            else :
                idcs = np.where( (pb > pbmin) & (pb <= pbmax) )
            n_class[i] = len(idcs[0])

            # compute mean occurance rate for class i
            if n_class[i] > 0 :
                obmean_class[i] = ob[idcs].mean()
            else :
                obmean_class[i] = 0

        n_tot = n_class.sum()

        #reliability
        rel = ( n_class * (pb_class - obmean_class)**2 ).sum() / n_tot

        # resolution
        res =  ( n_class * (obmean_class - obmean)**2 ).sum() / n_tot

        # skill score
        bss = (res-rel)/unc

        #print 'n_class = ', n_class, n_tot, ob.size
        #print 'pb_class = ', pb_class
        #print 'obmean_class = ', obmean_class
        #print 'components = ', res, rel, unc
        #print 'score = ', bs, rel - res + unc, bss

        return {'bs':bs, 'unc':unc, 'rel':rel, 'res':res, 'bss':bss, 'n_tot':n_tot,
                'n_prob':n_prob, 'n_class':n_class, 'obmean':obmean, 'obmean_class':obmean_class, 'pb_class':pb_class }

    else :
        return bs

def brier_score_masked( obs, ens, thres, components=False, n_prob=10 ) :
    """Computes Brier score for observations obs, ensemble of model equivalents ens and threshold thres.
       The observation array may be a masked arrays and masked entries are ignored in the computations."""

    # make obs binary
    ob = ma.masked_array( np.zeros(obs.shape, dtype=obs.dtype ) )
    ob[ ma.where(obs>thres) ] = 1
    if ma.is_masked(obs) :
        ob.mask = obs.mask

    # compute probabilities from ensemble
    eb = ma.masked_array( np.zeros(ens.shape, dtype=ens.dtype ) )
    eb[ ma.where(ens>thres) ] = 1
    if ma.is_masked(ens) :
        eb.mask = ens.mask
    pb = eb.mean(axis=0)

    bs = ((pb - ob)**2).mean()

    if components :
        # uncertainty
        obmean = ob.mean()
        unc = obmean * ( 1.0 - obmean )

        n_class      = np.zeros(n_prob)
        pb_class     = np.zeros(n_prob)
        obmean_class = np.zeros(n_prob)
        for i in range(n_prob) :
            pbmin = float(i  )/n_prob
            pbmax = float(i+1)/n_prob
            pb_class[i] = 0.5*( pbmin + pbmax )

            if i == 0 :
                idcs = np.where( (pb >= pbmin) & (pb <= pbmax) & (pb.mask == False) )
            else :
                idcs = np.where( (pb > pbmin) & (pb <= pbmax) & (pb.mask == False) )
            n_class[i] = len(idcs[0])

            if n_class[i] > 0 :
                obmean_class[i] = ob[idcs].mean()
            else :
                obmean_class[i] = 0

        n_tot = n_class.sum()

        #reliability
        rel = ( n_class * (pb_class - obmean_class)**2 ).sum() / n_tot

        # resolution
        res =  ( n_class * (obmean_class - obmean)**2 ).sum() / n_tot

        # skill score
        bss = (res-rel)/unc

        #print 'n_class = ', n_class, n_class.sum(), obs.size
        #print 'pb_class = ', pb_class
        #print 'obmean_class = ', obmean_class
        #print 'components = ', res, rel, unc
        #print 'score = ', bs, rel - res + unc, bss

        return {'bs':bs, 'unc':unc, 'rel':rel, 'res':res, 'bss':bss, 'n_tot':n_tot,
                'n_prob':n_prob, 'n_class':n_class, 'obmean':obmean, 'obmean_class':obmean_class, 'pb_class':pb_class }

    else :
        return bs

def accumulate_brier_scores( scores ) :
    """Compute total brier score for total data set from a list of Brier score results for parts of the total data set"""

    n_tot   = scores[0]['n_tot']
    bs      = scores[0]['bs']*scores[0]['n_tot']

    n_class = scores[0]['n_class']
    obmean  = scores[0]['obmean']*scores[0]['n_tot']
    obmean_class = scores[0]['obmean_class']*scores[0]['n_class']

    for score in scores[1:] :

        n_tot   += score['n_tot']
        bs      += score['bs']*score['n_tot']

        n_class += score['n_class']
        obmean  += score['obmean']*score['n_tot']
        obmean_class += score['obmean_class']*score['n_class']

    bs           /= n_tot
    obmean       /= n_tot
    obmean_class /= n_class

    n_prob   = scores[0]['n_prob']
    pb_class = scores[0]['pb_class']

    # uncertainty
    unc = obmean * ( 1.0 - obmean )

    #reliability
    rel = ( n_class * (pb_class - obmean_class)**2 ).sum() / n_tot

    # resolution
    res =  ( n_class * (obmean_class - obmean)**2 ).sum() / n_tot

    # skill score
    bss = (res-rel)/unc

    return {'bs':bs, 'unc':unc, 'rel':rel, 'res':res, 'bss':bss, 'n_tot':n_tot,
            'n_prob':n_prob, 'n_class':n_class, 'obmean':obmean, 'obmean_class':obmean_class, 'pb_class':pb_class }

