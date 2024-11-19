#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . M O D E L _ S T A T E
#  class representing COSMO model state (wraps around enstools.io)
#
#  2018.8 L.Scheck

from __future__ import absolute_import, division, print_function
from enstools import io
import numpy as np

class ModelState(object):
    """class representing a COSMO model state"""

    # level types used in Grib1 and Grib2 files + level type classes:
    #                M = model levels
    #                P = pressure l.
    #                Z = geometrical height
    #                S = surface
    #                0 = mean sea level
    #                L = special level, e.g. cloud top
    #                X = satellite or other vertically integrating obs.
    level_types = { 'M':['hybrid','hybridLayer','generalVertical','generalVerticalLayer'],
                    'P':['isobaricInhPa'],
                    'Z':['depthBelowLand','depthBelowLandLayer','heightAboveSea','heightAboveGround'],
                    'S':['surface'],
                    '0':['meanSea'],
                    'L':['isothermZero','cloudTop','cloudBase','nominalTop','lakeBottom','thermocline','mixedLayer',
                         'entireLake','atmML'],
                    'U':['unknown'],
                    'X':['TOA','meanLayer','entireAtmosphere','isobaricLayer'] }

    # variables that can be computed or at least approximated
    computable_quantities = ['PHL','PML','HML','RLAT','RLON','RHO','dz','TQV','TQC','TQI','RELHUM']

    #-------------------------------------------------------------------------------------------------------------------
    def __init__( self, fname, def_lev_type='M', constfile=None, verbose=True ) :
        """
        Initialize ModelState object

        :param fname:        model output file name
        :param def_lev_type: default level type
        :param verbose:      Be more verbose
        """

        if verbose :
            print()

        # open file using enstools
        self.Dataset = io.read(fname)

        # optional additional file containing constant arrays (like HHL) that are not written to all output files
        # will be opened only on demand...
        self.constfile = constfile
        self.DatasetConst = None

        # set up dictionary with synonyms
        self.synonyms = {}
        for vname in self.Dataset.data_vars :
            for lt in self.level_types :
                for ln in self.level_types[lt] :
                    if vname.endswith('_'+ln) :
                        basevname = vname[:-len(ln)-1]
                        self.create_synonym( vname, basevname+'__'+lt ) # e.g. T__M -> T_hybrid
                        if lt == def_lev_type :
                            self.create_synonym( vname, basevname )     # e.g. T    -> T_hybrid

        if verbose :
            print('available variables: ')
            print(self.Dataset)
            print('synonyms:')
            for s in self.synonyms :
                print( '  - ', s, ' = ', self.synonyms[s] )

    #-------------------------------------------------------------------------------------------------------------------
    def create_synonym( self, vname, syn ):
        if syn in self.Dataset.data_vars :
            print('WARNING: could not create synonym {} for {} -- variable with same name exists already!'.format(syn,vname))
        elif syn in self.synonyms :
            print('WARNING: could not create synonym {} for {} -- synonym exists already and points to {}!'.format(syn,vname,self.synonym[syn]))
        elif syn in self.Dataset.coords :
            print('WARNING: could not create synonym {} for {} -- coordinate with same name exists already!'.format(syn,vname))
        else :
            self.synonyms[syn] = vname

    #-------------------------------------------------------------------------------------------------------------------
    def __getitem__( self, vname ):
        return self.get( vname )

    #-------------------------------------------------------------------------------------------------------------------
    def __getattr__( self, vname ):
        return self.get( vname )

    #-------------------------------------------------------------------------------------------------------------------
    def get( self, vname ) :

        ret = None

        if (vname in self.Dataset.data_vars) or (vname in self.Dataset.coords) :
            ret = self.Dataset[vname]

        elif vname in self.synonyms :
            ret = self.Dataset[ self.synonyms[vname] ]

        elif vname in self.computable_quantities :
            ret = self.compute( vname )

        else :
            if not self.constfile is None :
                if self.DatasetConst is None :
                    self.DatasetConst = io.read( self.constfile )
                if vname in self.DatasetConst.data_vars :
                    ret = self.DatasetConst[vname]

        return ret

    #-------------------------------------------------------------------------------------------------------------------
    def compute(self, vname ) :

        print("ModelState.compute hast NOT YET BEEN TESTED AND PROBABLY DOES NOT WORK !!!")

        if vname == 'HML' :
            res = self.adjust_levels(self['HHL'], like=self['HHL'][:,:,1:])
            #meta = {'name': 'geometric model level height above sea level', 'units': 'm'}

        elif vname == 'PHL' : # pressure in model layers
            z = self['HHL']
            res = self.pref(z) + self.adjust_levels( self['PP'], like=self['HHL'] )
            #meta = { 'name':'pressure on half model levels', 'units':'Pa'}

        elif vname == 'P' : # pressure at model levels
            z = self['HML']
            res = self.pref(z) + self['PP']
            #meta = {'name': 'pressure on full model levels', 'units':'Pa'}

        elif vname == 'RHO' :
            Rd = 287.058 # J/(kg·K)
            Rv = 461.495 # J/(kg·K)
            res = self['P'] / ( Rd*self['T']*(1+(Rv/Rd-1)*self['QV']-self['QC']-self['QI']))
            # meta = {'name': 'density', 'units':'kg/m3'}

        elif vname == 'dz' :
            res = self['HHL'][...,:-1,:,:] - self['HHL'][...,1:,:,:]
            #meta = {'name': 'level spacing', 'units':'m'}

        elif vname == 'TQV' :
            res = (self['QV'] *self['RHO']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
            #meta = {'name': 'Total Column-Integrated Water vapour', 'units':'kg m-2'}

        elif vname == 'TQC' :
            res = (self['QC'] *self['RHO']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
            #meta = {'name': 'Total Column-Integrated Cloud Water', 'units':'kg m-2'}

        elif vname == 'TQI' :
            res = (self['QI'] *self['RHO']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
            #meta = {'name': 'Total Column-Integrated Cloud Ice', 'units':'kg m-2'}

        elif vname == 'RELHUM' :
            # from data_constants.f90
            b1       =   610.78
            b2w      =    17.2693882
            b3       =   273.16
            b4w      =    35.86
            r_d      =   287.05 # gas constant for dry air
            r_v      =   461.51 # gas constant for water vapor
            rdv      = r_d / r_v
            o_m_rdv  = 1.0 - rdv
            # from pp_utilities.f90/calrelhum
            zpvs = b1*np.exp( b2w*(self['T']-b3) / (self['T']-b4w) )
            zqvs = rdv*zpvs / (self['P'] - o_m_rdv*zpvs)
            # Set minimum value of relhum to 0.01 (was 0 before)
            res = np.maximum( self['QV']/zqvs * 100, 0.01 )
            #meta = {'name': 'Relative Humidity', 'units':'%'}

        else :
            raise ValueError("ModelState.compute: I don't know how to compute "+vname)

        return res

    #-------------------------------------------------------------------------------------------------------------------
    def pref( self, z ) :
        """Compute reference pressure for given height field z"""

        # from data_constants.f90 :
        r_d  = 287.05
        g    = 9.80665
        # from COSMO User's Guide, Section 3.1
        psl  = 100000.0 # Pa
        tsl  = 288.15   # K
        beta = 42       # K
        return psl * np.exp( -(tsl/beta) * (1 - np.sqrt( 1 - (2*beta*g*z)/(r_d*(tsl**2)) ) ) )
