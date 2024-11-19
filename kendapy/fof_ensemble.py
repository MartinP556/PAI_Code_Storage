#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . F O F _ E N S E M B L E
#  represents an ensemble of fof files, allows to compute statistics for common subset of observations
#
#  2018.1 L.Scheck

from __future__ import absolute_import, division, print_function
import numpy as np
import os, sys, getpass, subprocess, argparse, time, re, gc, pickle

class FofEns(object):
    """class representing an ensemble of FOF files"""

    #...................................................................................................................
    def __init__( self, fofs, ref=None, varname=None, obstype=None, timerange=None, harmonize=False, verbose=False ) :
        """
        Initialize fof ensemble, find common subset of observations
        (optionally exclude all observations that are not contained in a reference Ekf object)

        :param fofs:     list of Ekf objects representing fof files
        :param ref:      reference Ekf object
        :param verbose:  Be more verbose
        """

        self.fofs  = fofs
        self.ref   = ref
        self.n_ens = len(fofs)

        for fof in fofs :
            fof.set_filter( add=True, varname=varname, obstype=obstype, time=timerange )

        if harmonize : #or (not varname is None) or (not obstype is None) or (not timerange is None) :
            self.harmonize( varname=varname, obstype=obstype, timerange=timerange )

    #...................................................................................................................
    def harmonize( self, varname=None, obstype=None, timerange=None, params=None ) :
        """
        Filter observations according to variable name, observation type and time range,
        remove all observation that are not present in all members (and the reference observation set, if specified).
        
        :param varname: variable name 
        :param obstype: observation type
        :param timerange: time range (t_first,t_last) in minutes
        :param params: parameters to extract from the fof files (default: ['obs','time','plevel','lat','lon'] ) 
        :return: 
        """

        if params is None :
            params = ['obs','time','plevel','lat','lon']

        # compute identifiers for reference observation set (if available)
        if not self.ref is None :
            self.ref.set_filter( add=True, varname=varname, obstype=obstype, time=timerange )

            # account for different reference times in ref and fof
            trr = self.ref.attr['verification_ref_time']
            trf = self.fofs[0].attr['verification_ref_time']
            # we assume the ref reference time is later (analysis is performed after start of forecasts)
            if trr < trf : # oops, there was midnight...
                trr += 2400
            time_shift = (trr//100 - trf//100)*60 + trr%100 - trf%100
            ids = self.identify( self.ref, time_shift=time_shift )
            print(('found %d observations in the reference set...' % len(ids)))
            n_complete = self.n_ens+1
        else :
            ids = None
            n_complete = self.n_ens

        # count observations in each ensemble member
        for fof in self.fofs :
            fof.set_filter( add=True, varname=varname, obstype=obstype, time=timerange )
            ids = self.identify( fof, ids )

        # remove observation ids that have not been found in all members and there reference
        to_be_removed = []
        for id in ids :
            if ids[id] != n_complete :
                to_be_removed.append(id)
        for id in to_be_removed :
            ids.pop(id)
        print(('removed %d observations that are not available in all members...' % len(to_be_removed)))

        n_obs = len(ids)
        print(('Now %d observations are left...' % n_obs))

        # create result arrays
        res = {'n_obs':n_obs}
        for p in params :
            res[p] = np.zeros(n_obs)
        res['meq'] = np.zeros((self.n_ens,n_obs))

        # compute result index for each id
        idx = {}
        jj = 0
        for id in ids :
            idx[id] = jj
            jj += 1

        # fill result arrays with data from fofs for the observations in ids
        for i in range(self.n_ens) :
            for j in range((self.fofs[i].obs()).size) :
                # compute identifier
                time   = self.fofs[i].obs(param='time')
                level  = self.fofs[i].obs(param='level')
                lat    = self.fofs[i].obs(param='lat')
                lon    = self.fofs[i].obs(param='lon')
                ot     = self.fofs[i].obs(param='obstype')
                ct     = self.fofs[i].obs(param='codetype')
                varno  = self.fofs[i].obs(param='varno')
                #obs    = self.fofs[i].obs(param='obs')
                id = "%d,%d,%d,%d,%f,%f,%f" % ( varno[j], ot[j], ct[j], time[j], level[j], lat[j], lon[j] )
                jj = idx[id]
                for p in params :
                    res[p][jj] = self.fofs[i].obs(param=p)[j]
                res['meq'][i,jj] = self.fofs[i].veri('fg')[j]

        self.res = res
        self.n_obs = n_obs

    #...................................................................................................................
    def identify( self, ekf, ids=None, time_shift=0, n_target=0, verbose=True ):

        n_obs  = ekf.obs().size
        time   = ekf.obs(param='time')
        level  = ekf.obs(param='level')
        lat    = ekf.obs(param='lat')
        lon    = ekf.obs(param='lon')
        ot     = ekf.obs(param='obstype')
        ct     = ekf.obs(param='codetype')
        varno  = ekf.obs(param='varno')
        #obs    = ekf.obs(param='obs')

        if ids is None :
            # create new identifier count dictionary, avoid non-unique identifiers
            ids = {}
            if n_target > 0 :
                n_target_ = n_target
            else :
                n_target_ = 1
            ignore_unknown=False
            if verbose : print(('fof_ensemble/identify: starting with %d observations...' % n_obs))
        else :
            # use existing identifier count dictionary (which should not contain non-unique identifiers)
            if n_target > 0 :
                n_target_ = n_target
            else :
                n_target_ = 0
            ignore_unknown=True
            if verbose : print(('fof_ensemble/identify: starting with %d obs. and %d ref. obs.' % (n_obs,len(ids))))

        if verbose : print(('fof_ensemble/identify: target n = %d, unknown observations will be %s...' % (n_target_,
                                                                        'ignored' if ignore_unknown else 'included')))

        # compute identifier for each observation, count how often each identifier is encountered
        n_ignored = 0
        for j in range(n_obs) :
            #id = "%d,%f,%f,%f" % ( time[j], -1.0 if plevel.mask[j] else plevel[j], lat[j], lon[j] )
            #id = "%d,%f,%f,%f" % ( time[j]+time_shift, level[j], lat[j], lon[j] )
            #id = "%d,%d,%d,%d,%f,%f,%f,%f" % ( varno[j], ot[j], ct[j], time[j]+time_shift, level[j], lat[j], lon[j], obs[j] )
            id = "%d,%d,%d,%d,%f,%f,%f" % ( varno[j], ot[j], ct[j], time[j]+time_shift, level[j], lat[j], lon[j] )

            if id in ids :
                ids[id] += 1
            else :
                if not ignore_unknown :
                    ids[id] = 1
                else :
                    n_ignored += 1
                    print(('ignoring unknown observations with identifier %s...' % id))
        if ignore_unknown :
            if verbose : print(('fof_ensemble/identify: ignored %d unknown observations -> %d left...' % (n_ignored,len(ids))))

        if n_target_ != 0 : # remove observations with non-unique identifiers
            to_be_removed = []
            n_removed = 0
            n_too_often = 0
            n_too_seldom = 0
            for id in ids :
                if ids[id] != n_target_ :
                    if ids[id] > n_target_ :
                        n_too_often += ids[id]
                    else :
                        n_too_seldom += ids[id]
                    to_be_removed.append(id)
                    n_removed += ids[id]
            for id in to_be_removed :
                ids.pop(id)
            if verbose : print(('fof_ensemble/identify: removed %d obs. with %d (%d+%d) identifiers that had the wrong number of matches (n!=%d)...' % \
                               ( n_removed, len(to_be_removed), n_too_seldom, n_too_often, n_target_ )))
        else :
            print('not removing anything...')

        if verbose : print(('fof_ensemble/identify: ending up with %d observations...' % len(ids)))

        return ids


    #...................................................................................................................
    def statistics( self, obstype, varname, timerange ) :

        obs0 = self.fofs[0].obs(obstype=obstype,varname=varname,time=timerange)
        if len(obs0) == 0 :
            return {'n_obs':0}

        # retrieve observations and model equivalents for all members
        obs   = [ fof.obs(                obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]
        lat   = [ fof.obs( param='lat',   obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]
        lon   = [ fof.obs( param='lon',   obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]
        time  = [ fof.obs( param='time',  obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]
        level = [ fof.obs( param='level', obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]
        meq   = [ fof.veri('fg',          obstype=obstype,varname=varname,time=timerange) for fof in self.fofs ]

        nobs = np.array([o.size for o in obs])
        nmeq = np.array([m.size for m in meq])

        if False : # for debugging
            i = 0
            f = self.fofs[0]
            print(('FOF ', f.fname))
            for j in [137,138] :
                print(j, end=' ')
                for p in ['lat','lon','time','level','plevel'] :
                    print(p,'=',f.obs( param=p, obstype=obstype,varname=varname,time=timerange)[j], ' ', end=' ')
                print()

        # determine subset of common observations
        cobs, cmeq = common_subset( obs, meq, lat, lon, time, level )
        #, clat, clon, ctime, clevel

        print(('%s / %s observations in t=%s : N_max=%d, N_min=%d, N_common=%d' % (obstype, varname, timerange, nobs.max(), nobs.min(), cobs.size)))

        if len(cobs) == 0 :
            return {'n_obs':0}

        cmeq_ensmean = cmeq.mean(axis=0)
        stats = { 'rmse'  :np.sqrt( ((cobs-cmeq_ensmean)**2).mean() ),
                  'spread':np.sqrt( (cmeq.std(axis=0,ddof=1)**2).mean() ),
                  'bias'  :(cmeq_ensmean-cobs).mean(),
                  'n_obs' : cobs.size }

        stats.update({ 'obs':cobs, 'fgmean':cmeq_ensmean })

        return stats

def common_subset( obs, meq, lat, lon, time, level ) :
    """Determine common subset of observations in an ensemble"""

    n_ens = len(obs)

    og = {}
    for i in range(n_ens) :
        oh = {}
        for j in range(len(obs[i])) :
            oid = "t%d_lvl%f_lat%f_lon%f" % ( time[i][j], level[i][j], lat[i][j], lon[i][j] )
            if oid in oh :
                if og[oid] > -1 :
                    print(('WARNING: hash key not unique : ', oid, i, j, oh[oid], ' -> all observations with this key will be ignored.'))
                    og[oid] = -1
            else :
                oh[oid] = j
            if oid in og :
                if og[oid] > -1 :
                    og[oid] += 1
            else :
                og[oid] = 1
    cobs = {}
    cmeq = {}
    for i in range(n_ens) :
        for j in range(len(obs[i])) :
            oid = "t%d_lvl%f_lat%f_lon%f" % ( time[i][j], level[i][j], lat[i][j], lon[i][j] )
            if og[oid] == n_ens :
                if not oid in cmeq :
                    cmeq[oid] = np.zeros(n_ens)
                cmeq[oid][i] = meq[i][j]
                cobs[oid]    = obs[i][j]

    comobs = np.zeros(         len(cobs) )
    commeq = np.zeros(( n_ens, len(cobs) ))
    j = 0
    for oid in cobs :
        comobs[j]   = cobs[oid]
        for i in range(n_ens) :
            commeq[i,j] = cmeq[oid][i]
        j += 1

    return comobs, commeq


#-------------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------------
#-------------------------------------------------------------------------------------

    from kendapy.experiment import Experiment
    import argparse
    parser = argparse.ArgumentParser(description='Parse EKF files generated by KENDA experiment')

    parser.add_argument( '-S', '--state-filter',    dest='state_filter',    default='active', help='observation state filter [active|passive|valid(default)]' )
    parser.add_argument( '-O', '--obstype-filter',  dest='obstype_filter',  default='all',    help='observation type filter' )
    parser.add_argument( '-C', '--codetype-filter', dest='codetype_filter', default='all',    help='code type filter' )
    parser.add_argument( '-A', '--area-filter',     dest='area_filter',     default='all',    help='area filter' )
    parser.add_argument( '-T', '--time-filter',     dest='time_filter',     default='all',    help='time filter' )
    parser.add_argument( '-V', '--varname', dest='varname', help='variable name', default='T' )
    parser.add_argument( '-t', '--time', dest='time', help='print observation time information', action='store_true' )
    parser.add_argument( 'logfile', metavar='logfile', help='log file name[s]', nargs='*' )
    args = parser.parse_args()

    for logfile in args.logfile :
        print(logfile)
        xp = Experiment(logfile)

        fofs = [ xp.get_fof( xp.lfcst_start_times[0], state_filter=args.state_filter, memidx=m, lfcst=True ) for m in range(1,xp.n_ens+1) ]
        fofens = FofEns( fofs, verbose=True )

        #print fofens.statistics('AIREP','T','121-180')
        print((fofens.statistics(args.obstype_filter,args.varname,args.time_filter)))



