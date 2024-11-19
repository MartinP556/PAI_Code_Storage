#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . E K F
#  read & filter EKF files
#
#  2017.4 L.Scheck 

from __future__ import absolute_import, division, print_function
from numpy import *
import numpy as np
import os, sys, getpass, subprocess, argparse, time, re, gc, pickle
import netCDF4

# ---------------------------------------------- tables from documentation ---------------------------------------------
# (see http://www2.cosmo-model.org/content/model/documentation/core/cosmoFeedbackFileDefinition.pdf )

tables = {    'obstypes': { 1:'SYNOP'  ,
                            2:'AIREP'  ,
                            3:'SATOB'  ,
                            4:'DRIBU'  ,
                            5:'TEMP'   ,
                            6:'PILOT'  ,
                            7:'SATEM'  ,
                            8:'PAOB'   , 
                            9:'SCATT'  ,
                            10:'RAD'   ,
                            11:'GPSRO' ,
                            12:'GPSGB' ,
                            13:'RADAR' },
              'codetypes' : { -1:  'UNKNOWN-1',
                              0:   'UNKNOWN0',
                              145: 'ACARS',
                              11:  'SRSCD',
                              14:  'ATSCD',
                              21:  'AHSCD',
                              24:  'ATSHS',
                              140: 'METAR',
                              110: 'GPS',
                              141: 'AIRCD',
                              41:  'CODAR',
                              144: 'AMDAR',
                              146: 'MODES',
                              87:  'CLPRD',
                              88:  'STBCD',
                              90:  'AMV',
                              165: 'DRBCD',
                              64:  'TESAC',
                              35:  'LDTCD',
                              36:  'SHTCD',
                              135: 'TDROP',
                              37:  'TMPMB',
                              32:  'LDPCD',
                              33:  'SHPCD',
                              38:  'PLTMB',
                              210: 'ATOVS',
                              132: 'WP_EU',
                              133: 'RA_EU',
                              134: 'WP_JP',
                              136: 'PR_US',
                              137: 'RAVAD',
                              218: 'SEVIR',
                              123: 'ASCAT',
                              122: 'QSCAT',
                              216: 'AIRS',
                              217: 'IASI',
                              231: 'TEMPD',
                              109: 'UNKNOWN'},
              'r_states' : { 0:  'ACCEPTED',
                             1:  'ACTIVE',
                             3:  'MERGED',
                             5:  'PASSIVE',
                             7:  'REJECTED',
                             9:  'PAS REJ',
                             11: 'OBS ONLY',
                             13: 'DISMISS'},
              'r_flags' : { 2:  'SUSP LOCT',
                            3:  'TIME',
                            4:  'AREA',
                            8:  'PRACTICE',
                            9:  'DATASET',
                            1:  'BLACKLIST',
                            5:  'HEIGHT',
                            6:  'SURF',
                            7:  'CLOUD',
                            16: 'GROSS',
                            0:  'OBSTYPE',
                            10: 'REDUNDANT',
                            11: 'FLIGHTTRACK',
                            12: 'MERGE',
                            13: 'THIN',
                            14: 'RULE',
                            17: 'NO BIASCOR',
                            15: 'OBS ERR',
                            19: 'NO OBS',
                            18: 'FG',
                            21: 'FG LB',
                            20: 'OPERATOR',
                            32: 'NONE'},
              'varnames' : { 0:'NUM',
                             3:'U',
                             4:'V',
                             8:'W',
                             1:'Z',
                             57:'DZ',
                             9:'PWC',
                             28:'TRH',
                             29:'RH',
                             58:'RH2M',
                             2:'T',
                             59:'TD',
                             39:'T2M',
                             40:'TD2M',
                             11:'TS',
                             30:'PTEND',
                             60:'W1',
                             61:'WW',
                             62:'VV',
                             63:'CH',
                             64:'CM',
                             65:'CL',
                             66:'NHcbh',
                             67:'NL',
                             93:'NM',
                             94:'NH',
                             69:'C',
                             70:'NS',
                             71:'SDEPTH',
                             72:'E',
                             79:'TRTR',
                             80:'RR',
                             81:'JJ',
                             87:'GCLG',
                             91:'N',
                             92:'SFALL',
                             95:'ICLG',
                             110:'PS',
                             111:'DD',
                             112:'FF',
                             118:'REFL',
                             119:'RAWBT',
                             120:'RADIANCE',
                             41:'U10M',
                             42:'V10M',
                             7:'Q',
                             56:'VT',
                             155:'VN',
                             156:'HEIGHT',
                             157:'FLEV',
                             192:'RREFL',
                             193:'RADVEL',
                             128:'PDELAY',
                             162:'BENDANG',
                             252:'IMPPAR',
                             248:'REFR',
                             245:'ZPD',
                             246:'ZWD',
                             247:'SPD',
                             240:'VGUST',
                             242:'GUST',
                             251:'P',
                             243:'TMIN',
                             244:'UNKNOWN',
                             236:'RAD_DI',
                             237:'RAD_GL',
                             238:'RAD_DF',
                             239:'RAD_LW' },
              'fullvarnames' : { 0:'NUM ordinal (channel) number',
                                 3:'U m/s u-component of wind',
                                 4:'V m/s v-component of wind',
                                 8:'W m/s vertical velocity',
                                 1:'Z (m/s)**2 geopotential',
                                 57:'DZ (m/s)**2 thickness',
                                 9:'PWC kg/m**2 precipitable water content',
                                 28:'TRH 0..1 transformed relative humidity',
                                 29:'RH 0..1 relative humidity',
                                 58:'RH2M 0..1 2 metre relative humidity',
                                 2:'T K upper air temperature',
                                 59:'TD K upper air dew point',
                                 39:'T2M K 2 metre temperature',
                                 40:'TD2M K 2 metre dew point',
                                 11:'TS K surface temperature',
                                 30:'PTEND Pa/3h pressure tendency',
                                 60:'W1 WMO 020004 past weather',
                                 61:'WW WMO 020003 present weather',
                                 62:'VV m visibility',
                                 63:'CH WMO 020012 type of high clouds',
                                 64:'CM WMO 020012 type of middle clouds',
                                 65:'CL WMO 020012 type of low clouds',
                                 66:'NH m cloud base height',
                                 67:'NL WMO 020011 low cloud amount',
                                 93:'NM WMO 020011 medium cloud amount',
                                 94:'NH WMO 020011 high cloud amount',
                                 69:'C WMO 500 additional cloud group type',
                                 70:'NS WMO 2700 additional cloud group amount',
                                 71:'SDEPTH m snow depth',
                                 72:'E WMO 020062 state of ground',
                                 79:'TRTR h time period of information',
                                 80:'RR kg/m**2 precipitation amount',
                                 81:'JJ K maximum temperature',
                                 87:'GCLG Table 6 general cloud group',
                                 91:'N WMO 020011 total cloud amount',
                                 92:'SFALL m 6h snow fall',
                                 95:'individual cloud layer group',
                                 110:'PS Pa surface (station) pressure',
                                 111:'DD degree wind direction',
                                 112:'FF m/s wind force',
                                 118:'REFL 0..1 reflectivity',
                                 119:'RAWBT K brightness temperature',
                                 120:'RADIANCE W/sr/m**3 radiance',
                                 41:'U10M m/s 10m u-component of wind',
                                 42:'V10M m/s 10m v-component of wind',
                                 7:'Q kg/kg specific humidity',
                                 56:'VT K virtual temperature',
                                 155:'VN CTH m cloud top height',
                                 156:'HEIGHT m height',
                                 157:'FLEV m nominal flight level',
                                 192:'RREFL Db radar reflectivity',
                                 193:'RADVEL m/s radial velocity',
                                 128:'PDELAY m atmospheric path delay',
                                 162:'BENDANG rad bending angle',
                                 252:'IMPPAR m impact parameter',
                                 248:'REFR refractivity',
                                 245:'ZPD zenith path delay',
                                 246:'ZWD zenith wet delay',
                                 247:'SPD slant path delay',
                                 242:'GUST m/s wind gust',
                                 251:'P Pa pressure',
                                 243:'TMIN K minimum temperature' },
              'veri_run_types' : { 0 :'FORECAST',   
                                   1 :'FIRSTGUESS', 
                                   2 :'PREL ANA',   
                                   3 :'ANALYSIS',   
                                   4 :'INIT ANA',   
                                   5 :'LIN ANA' },
              'veri_run_classes' : { 0 : 'HAUPT',
                                     1 : 'VOR',  
                                     2 : 'ASS',  
                                     3 : 'TEST' },
              'veri_ens_member_names' : {  0:'ENS MEAN',    
                                           -1:'DETERM',      
                                           -2:'ENS SPREAD',  
                                           -3:'BG ERROR',    
                                           -4:'TALAGRAND',   
                                           -5:'VQC WEIGHT',  
                                           -6:'MEMBER',      
                                           -7:'ENS MEAN OBS' }
              }

########################################################################################################################

class Ekf(object):
    """class representing a EKF file"""

    # --------------------------------------- tables from documentation ------------------------------------------------
    # copy global tables (may be extended during runtime if unknown variables are found)
    table = tables

    #-------------------------------------------------------------------------------------------------------------------
    def __init__( self, fname, test_tables=False, convert_to_nonrotated=True, verbose=False, repair_ibody=True, **filter ) :
        """
        Initialize Ekf object

        :param fname:      ekf file name
        :param filter:     filter dictionary or string
        :param verbose:    Be more verbose
        """

        self.verbose = verbose

        # open ekf file and read global attributes .....................................................................
        self.fname = fname
        if verbose :
            print()
            print(('-- ' * 25))
            print(('-- Ekf : Opening %s...' % fname))
        ncf = netCDF4.Dataset( fname, 'r')

        self.attr = {}
        if verbose : print('-- global attributes :')
        for k in ncf.ncattrs() :
            self.attr[k] = getattr( ncf, k )
            if verbose : print(("--   %25s | %s" % (k,self.attr[k])))

        if ('verification_ref_date' in self.attr) and ('verification_ref_time' in self.attr) :
            self.ref_time = str(self.attr['verification_ref_date']) + str(self.attr['verification_ref_time']) + '00'

        if 'file_version_number' in self.attr :
            if float(self.attr['file_version_number']) < 1.015 :
                if verbose : print( 'Warning: This file still contains rotated velocity coordinates...' )
                self.rotated = True
            else :
                self.rotated = False

        self.n_hdr = self.attr['n_hdr']
        self.n_body = self.attr['n_body']
        if verbose :
            print(("-- Found n_body = %d obs in n_hdr = %d reports..." % (self.n_body,self.n_hdr)))

        # read data ....................................................................................................
        self.data = {}
        self.hdr_vars = []
        self.body_vars = []
        self.veri_vars = []
        self.radar_vars = []

        if verbose : print('-- available variables :')
        for vname in list(ncf.variables.keys()) :
            vdims = ncf.variables[vname].dimensions
            if verbose :
                print( "-- {:25s}".format(vname), vdims, ncf.variables[vname].size )
                if False :
                    if len(vdims) == 1 :
                        print( "-- {:25s}".format(vname), vdims, ncf.variables[vname].size, ncf.variables[vname][0], ', ',\
                               ncf.variables[vname][0], ', ... ', ncf.variables[vname][-1])
                    else :
                        if vdims[1] == 'char10' :
                            print((("-- %25s"%vname), vdims, ncf.variables[vname].size, '"',\
                                  ncf.variables[vname][0,:].tostring().decode(), '", ... "', ncf.variables[vname][-1,:].tostring().decode()))

            # classify & read variable
            if vdims[0] == 'd_hdr'  :
                self.hdr_vars.append(vname)
                self.data[vname] = ncf.variables[vname][:self.n_hdr,...]
                # in some fof files there are > n_hdr elements in the first dimension -> take only the first n_hdr

            elif vdims[0] == 'd_body' :
                self.body_vars.append(vname)
                self.data[vname] = ncf.variables[vname][:self.n_body,...]
                # in some fof files there are > n_body elements in the first dimension -> take only the first n_body

            elif vdims[0] == 'd_veri' :
                self.veri_vars.append(vname)
                self.data[vname] = ncf.variables[vname][...]

            elif vdims[0] == 'd_radar' :
                self.radar_vars.append(vname)
                self.data[vname] = ncf.variables[vname][...]

            else :
                print(('WARNING: Found unkown netCDF variable %s in %s.' % (vname,fname)))
                self.data[vname] = ncf.variables[vname][...]

        if verbose :
            print(('-- header       variables : ', self.hdr_vars))
            print(('-- body         variables : ', self.body_vars))
            print(('-- verification variables : ', self.veri_vars))
            print(('-- radar        variables : ', self.radar_vars))
        # close ekf file
        ncf.close()

        # i_body : FOF INDICES ARE IN C CONVENTION, EKF INDICES ARE IN FORTRAN CONVENTION
        # !!!! FIXED ON 2018-FEB-7 !!!!
        if ('ekf' in self.fname.split('/')[-1]) or  ('ver' in self.fname.split('/')[-1]):
            # adjust indices from Fortran -> C convention
            self.data['i_body'] -= 1

        if not self.data['i_body'].min() == 0 :
            print('>>> WARNING <<<  Minimum i_body should be 0 and is', self.data['i_body'].min() )
            print('                 Maximum i_body = {}, n_body = {}'.format(self.data['i_body'].max(), self.n_body))
            if repair_ibody :
                print('!!! --> REDUCING ALL i_body VALUES BY ', self.data['i_body'].min() )
                self.data['i_body'] -= self.data['i_body'].min()
            
            if False :
                print()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                for i in range(10) :
                    print( 'report #{} : i_body = {}-{}'.format(i, self.data['i_body'][i], self.data['i_body'][i]+self.data['l_body'][i]-1 ) )
                    print( '             index_x={}, index_y={}'.format( self.data['index_x'][i], self.data['index_y'][i]  ))
                    print( '             levels = ', self.data['level'][ self.data['i_body'][i] : self.data['i_body'][i]+self.data['l_body'][i] ])
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print()

        # add link from body entries back to corresponding report
        self.data['i_hdr'] = zeros(self.n_body,dtype=int)
        for i in range(self.n_hdr) :
            self.data['i_hdr'][ self.data['i_body'][i] : self.data['i_body'][i]+self.data['l_body'][i]] = i
        self.body_vars.append('i_hdr')
        # test
        #for i in range(self.n_body) :
        #    i_hdr = self.data['i_hdr'][i]
        #    print self.data['i_body'][i_hdr], i, self.data['i_body'][i_hdr]+self.data['l_body'][i_hdr]


        if self.rotated and convert_to_nonrotated :
            pass # implement me...
            #if verbose : print 'converting all velocity components from rotated to non-rotated coordinates...'
            #pollat, pollon = self.attr['pole_lat_lon']
            #for i in range(self.n_hdr) :
            #    for j in range( self.data['i_body'][i], self.data['i_body'][i]+self.data['l_body'][i] ) :
            #u, v =  uvrot2uv( urot, vrot, rlat, rlon, pollat=pollat, pollon=pollon )
            #self.rotated = False


        # check which observation types and variables are available ....................................................
        self.all_obstypes =  [ self.table['obstypes'][i]  for i in set(self.data['obstype']) ]
        self.all_codetypes = [ self.table['codetypes'][i] for i in set(self.data['codetype']) ]
        self.all_varnos   = set(self.data['varno'])
        self.all_varnames = []
        for i in self.all_varnos :
            if not i in list(self.table['varnames'].keys ()) :
                print(('WARNING: Encountered unknown variable number %d !' % i))
                self.table['varnames'][i] = 'V'+str(i)
        self.all_varnames = [self.table['varnames'][i] for i in self.all_varnos]

        # invert all tables (this is done here, because unknown V??? variables may have been added)
        self.invert_tables()

        # which combinations of obstype / var do occur?
        self.all_ov_combinations = {}
        for ot in self.all_obstypes :
            self.all_ov_combinations[ot] = {}
            ots = self.data['obstype'][ self.data['i_hdr'] ]
            idcs = where( ots == self.table['obstypes_inv'][ot] )
            for vn in self.all_varnames :
                vns = self.data['varno'][idcs]
                idcs2 = where( vns == self.table['varnames_inv'][vn] )
                m = len( idcs2[0] )
                if m > 0 :
                    self.all_ov_combinations[ot][vn] = m
                    #print "%s/%s " %  (ot, vn),

        if verbose :
            for ot in list(self.all_ov_combinations.keys()) :
                print('-- variables available for obstype %s : ' % ot, end=' ')
                for vn in self.all_ov_combinations[ot] :
                    print("%s " % vn, end=' ')
                print()
           

        # check which verification runs are available ..................................................................

        if verbose :
            print('--')
            print(('-- verification runs '+('-'*120)))
            print(("--   %2s |     %12s |     %5s | %20s | %8s | %5s | %10s | description" % \
                  ('i', 'type', 'class','ens. member','exp_id', 'time', 'model' )))

        self.i_det          = -1
        self.i_anamean      = -1
        self.i_fgmean       = -1
        self.i_fgspread     = -1
        self.i_fgdet        = -1
        self.i_anadet       = -1
        self.i_fgens_first  = -1
        self.i_fgens_last   = -1
        self.i_anaens_first = -1
        self.i_anaens_last  = -1

        self.i_maindet = {}

        for i in range(self.data['veri_run_type'].size) :

            if self.data['veri_ens_member'][i] > 0 :
                mem = "%d" % self.data['veri_ens_member'][i]
            else :
                mem = ("%d = " % self.data['veri_ens_member'][i]) + self.table['veri_ens_member_names'][self.data['veri_ens_member'][i]]

            veri_model =  self.data['veri_model'][i,:].tostring().decode().strip()
            veri_desc  =  self.data['veri_description'][i,:].tostring().decode().strip()

            if verbose :
                print(("--   %2d | %1d = %12s | %1d = %5s | %20s | %8d | %5d | %10s | " % (i,
                      self.data['veri_run_type'][i], self.table['veri_run_types'][self.data['veri_run_type'][i]],
                      self.data['veri_run_class'][i], self.table['veri_run_classes'][self.data['veri_run_class'][i]],
                      mem, self.data['veri_exp_id'][i], self.data['veri_forecast_time'][i], veri_model ), veri_desc))

            # analysis ensemble mean
            if self.data['veri_run_type'][i] == 3 and self.data['veri_ens_member'][i] == -7 and self.i_anamean < 0 :
                self.i_anamean = i   

            # first guess ensemble mean and spread
            if self.data['veri_run_type'][i] == 1 and self.data['veri_ens_member'][i] == -7 and self.i_fgmean < 0 :
                self.i_fgmean = i   
            if self.data['veri_run_type'][i] == 1 and self.data['veri_ens_member'][i] == -2 and self.i_fgspread < 0 :
                self.i_fgspread = i   

            # deterministic run first guess and analysis
            if self.data['veri_run_type'][i] == 1 and self.data['veri_ens_member'][i] == -1 and self.i_fgdet < 0 :
                self.i_fgdet = i   
            if self.data['veri_run_type'][i] == 2 and self.data['veri_ens_member'][i] == -1 and self.i_anadet < 0 :
                self.i_anadet = i   

            # deterministic run
            if self.data['veri_run_type'][i] == 0 and self.data['veri_ens_member'][i] == -1 and self.i_det < 0 :
                self.i_det = i   

            # first guess ensemble members
            if self.data['veri_run_type'][i] == 1 and self.data['veri_ens_member'][i] == 1 and self.i_fgens_first < 0 :
                self.i_fgens_first = i
            if self.data['veri_run_type'][i] == 1 and self.data['veri_ens_member'][i] >= 1 :
                self.i_fgens_last = i

            # analysis ensemble members
            if self.data['veri_run_type'][i] == 5 and self.data['veri_ens_member'][i] == 1 and self.i_anaens_first < 0 :
                self.i_anaens_first = i
            if self.data['veri_run_type'][i] == 5 and self.data['veri_ens_member'][i] >= 1 :
                self.i_anaens_last = i

            # deterministic main forecast
            if self.data['veri_run_type'][i] == 0 and self.data['veri_run_class'][i] == 0 and self.data['veri_ens_member'][i] == -1 :
                inidate = ''.join(( b.decode('UTF-8') for b in self.data['veri_initial_date'][i,:])) + '00'
                #leadtime = self.data['veri_forecast_time'][i]
                #print('found deterministic main forecast for initial date ', inidate)
                self.i_maindet[inidate] = i  

        if verbose : print(('------------------'+('-'*120)))

        if verbose :
            if self.i_anamean  < 0 : print(("WARNING: cannot find analysis ensemble mean in observation space in "+self.fname))
            if self.i_fgmean   < 0 : print(("WARNING: cannot find first guess ensemble mean in observation space in "+self.fname))
            if self.i_fgspread < 0 : print(("WARNING: cannot find first guess ensemble spread in observation space in "+self.fname))

            if self.i_fgens_first  < 0 : print(("WARNING: cannot find first first guess ensemble member in "+self.fname))
            if self.i_anaens_first < 0 : print(("WARNING: cannot find first analysis ensemble member in "+self.fname))

        if self.i_fgens_first < 0 or self.i_fgens_last < 0 :
            self.nens = -1
        else :
            self.nens = 1 + self.i_fgens_last - self.i_fgens_first
            if verbose:
                if self.nens != 1 + self.i_anaens_last - self.i_anaens_first :
                    print("WARNING: number of ensemble members different in first guess and analysis!")
                print(('Found %d ensemble members...' % self.nens))
                print()

        if verbose :
            if self.i_fgdet  < 0 : print(("WARNING: cannot find deterministic ensemble member first guess in "+self.fname))
            if self.i_anadet < 0 : print(("WARNING: cannot find deterministic ensemble member analysis in "+self.fname))
            if self.i_det    < 0 : print(("WARNING: cannot find deterministic run in "+self.fname))
                
        # set filter ...................................................................................................
        self.filtered_idcs = {}
        self.clear_filter()
        self.set_filter( **filter )
        self.oids = {}

        # print observation status statistics ..........................................................................
        if verbose :
            self.print_state_statistics( apply_filter=False )
            self.print_state_statistics( apply_filter=True, continue_output=True )
            print()

        self.stat = {}

    #-------------------------------------------------------------------------------------------------------------------
    def uvrot2uv( urot, vrot, rlat, rlon, pollat=-40.0, pollon=10 ) :
        """
        Adapted from dace_code/basic/utilities.f90

        input parameters:
        urot, vrot,       ! wind components in the rotated grid
        rlat, rlon,       ! latitude and longitude in the true geographical system
        pollat, pollon    ! latitude and longitude of the north pole of the
                          ! rotated grid

        returns u, v      ! wind components in the true geographical system
        """

        zrpi18 = 57.2957795
        zpir18 = 0.0174532925

        zsinpol = np.sin(pollat * zpir18)
        zcospol = np.cos(pollat * zpir18)
        zlonp   = (pollon-rlon) * zpir18
        zlat    =         rlat  * zpir18

        zarg1   = zcospol*np.sin(zlonp)
        zarg2   = zsinpol*np.cos(zlat) - zcospol*np.sin(zlat)*np.cos(zlonp)
        znorm   = 1.0/np.sqrt(zarg1**2 + zarg2**2)

        u       =   urot*zarg2*znorm + vrot*zarg1*znorm
        v       = - urot*zarg1*znorm + vrot*zarg2*znorm

        return u, v

    #-------------------------------------------------------------------------------------------------------------------
    def invert_tables( self, test_tables=False ) :
        """For each table in Ekf.table{} generate a 'inverted' table in which keys and values are swapped"""

        for tab in list(self.table.keys()) :
            if tab.endswith('_inv') : continue
            self.table[tab+'_inv'] = {y.strip():x for x,y in list(self.table[tab].items())}
            if test_tables : # test bijectivity
                for v in list(self.table[tab].keys()) :
                    print(('>>> ', v, self.table[tab][v], self.table[tab+'_inv'][self.table[tab][v]]))
                    if self.table[tab+'_inv'][self.table[tab][v]] != v :
                        raise ValueError('Table %s has non-unique key %s' % (tab,v))

    #-------------------------------------------------------------------------------------------------------------------
    def print_state_statistics( self, apply_filter=True, continue_output=False ) :

        if apply_filter :
            fidcs = self.idcs
        else :
            fidcs = list(range(self.n_body))

        r_state = self.data['r_state'][ self.data['i_hdr'] ][fidcs]
        r_check = self.data['r_check'][ self.data['i_hdr'] ][fidcs]
        state   = self.data['state'][fidcs]
        check   = self.data['check'][fidcs]

        header = "%12s  %12s  |  %12s  %12s  | %8s " % ('R_STATE','R_CHECK','STATE','CHECK','#CASES')
        for vn in self.all_varnames :
            header += ("%6s " % vn[-6:])
        divider = '-' * len(header)

        if not continue_output :
            print()
            print('-- state and flags statistics ', end=' ')
            if apply_filter :
                print('(after filtering) :')
            else :
                print('(without filter) :')
            print(divider)
            print(header)
            print(divider)
        else :
            if apply_filter :
                print("-- and after filtering :")
                print(divider)

        for rst in list(self.table['r_states'].keys()) :
            for rck in list(self.table['r_flags'].keys()) :
                for st in list(self.table['r_states'].keys()) :
                    for ck in list(self.table['r_flags'].keys()) :
                        idcs = where( (r_state==rst) & (r_check==rck) & (state==st) & (check==ck) )
                        n = len(idcs[0])
                        if n > 0 :
                            print("%12s  %12s  |  %12s  %12s  | %8d" % (self.table['r_states'][rst],
                                                                        self.table['r_flags'][rck],
                                                                        self.table['r_states'][st],
                                                                        self.table['r_flags'][ck],n), end=' ')
                            vns = self.data['varno'][fidcs][idcs]
                            for vn in self.all_varnames :
                                m = len( where( vns == self.table['varnames_inv'][vn ])[0] )
                                print('%5d '%m, end=' ')
                            print()
        print(divider)

    #-------------------------------------------------------------------------------------------------------------------
    def filter_indices( self, filter=None, add=True, return_name=False, verbose=False,
                        state_filter=None, obstype_filter=None, codetype_filter=None, area_filter=None,
                        time_filter=None, level_filter=None, pressure_filter=None, varname_filter=None, report_filter=None,
                        station_filter=None ) :
        """
        Determine indices of the observations that fulfill the specified requirements.            
        :param filter_: None, string or dictionary specifying the filters 
        :param add:     If True, use the settings in self.filter for each filter property not specified in filter_
        :return:        Indices (in [0,n_body-1]) of the observations, filter name 
        """

        if filter is None :
            ffilter = {}
        elif isinstance(filter, dict) :
            ffilter = filter
        else :
            ffilter = self.filtername2dict( filter )

        if verbose : print(( 'filter_indices : input filter = ', ffilter ))

        # settings for specific filters ('*_filter') override settings from the 'filter' argument
        if not state_filter    is None : ffilter['state']    = state_filter
        if not obstype_filter  is None : ffilter['obstype']  = obstype_filter
        if not codetype_filter is None : ffilter['codetype'] = codetype_filter
        if not area_filter     is None : ffilter['area']     = area_filter
        if not time_filter     is None : ffilter['time']     = time_filter
        if not pressure_filter is None : ffilter['pressure'] = pressure_filter
        if not level_filter    is None : ffilter['level']    = level_filter
        if not varname_filter  is None : ffilter['varname']  = varname_filter
        if not report_filter   is None : ffilter['report']   = report_filter
        if not station_filter  is None : ffilter['station']  = station_filter

        if verbose : print('filter_indices : modified filter = ', ffilter)

        if add :
            #print 'ADD input filter   : ', ffilter
            #print 'ADD default filter : ', self.filter
            for f in self.filter :
                if not f in ffilter :
                    ffilter[f] = self.filter[f]

        if verbose : print('filter_indices : filter with default values = ', ffilter)

        if 'name' in ffilter :
            filtername = ffilter['name']
        else :
            # Create an unique filter description string
            filtername = self.dict2filtername( ffilter, remove_empty=True )

        if filtername in self.filtered_idcs : # cached results are available for this filter name ......................

            idcs = self.filtered_idcs[filtername]

        else : # filter name is not yet known, observation indices still have to be determined .........................

            use_obs = zeros( self.n_body, dtype=bool )

            # state filter .............................................................................................
            if ('state' in ffilter) \
                and (not ffilter['state'] is None) and (ffilter['state'] != 'all') and (ffilter['state'] != '') :

                state = ffilter['state']

                rsi = self.table['r_states_inv']
                rfi = self.table['r_flags_inv']

                # get observation states and checks
                o_state = self.data['state']
                o_check = self.data['check']

                # expand report states, checks and obstypes to same size as observation states
                r_state   = self.data['r_state'][ self.data['i_hdr'] ]
                r_check   = self.data['r_check'][ self.data['i_hdr'] ]
                r_obstype = self.data['obstype'][ self.data['i_hdr'] ]

                if   state == 'active' :  # only active obs.
                    use_obs[ where( ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True

                elif state == 'obs_active' :  # only active obs., report state does not matter
                    use_obs[ where( ( o_state==rsi['ACTIVE'] ) ) ] = True

                elif state == 'report_active' :  # report state active, obs. state does not matter
                    use_obs[ where( ( r_state==rsi['ACTIVE'] ) ) ] = True

                elif state == 'passive' : # only passive obs.
                    #use_obs[ where( (o_state==rsi['PASSIVE']) & ( (check==rfi['NONE']) | (check==rfi['OBSTYPE']) ) ) ] = True
                    #use_obs[ where(  (r_state==rsi['PASSIVE']) | (o_state==rsi['PASSIVE']) ) ] = True
                    use_obs[ where( ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True

                elif state == 'valid' :   # active and passive obs.
                    #use_obs[ where( ( (r_state==rsi['ACTIVE']) | (r_state==rsi['ACCEPTED']) ) & (o_state==rsi['ACTIVE']) ) ] = True
                    #use_obs[ where( (o_state==rsi['PASSIVE']) & ( (check==rfi['NONE']) | (check==rfi['OBSTYPE']) ) ) ] = True
                    #use_obs[ where(  (r_state==rsi['PASSIVE']) | (o_state==rsi['PASSIVE']) ) ] = True
                    use_obs[ where( ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True
                    use_obs[ where( ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True

                elif state == 'notrej' : # not rejected
                    use_obs[ where( ( r_state == rsi['ACTIVE']  ) & ( o_state <= rsi['PASSIVE'] ) ) ] = True
                    use_obs[ where( ( r_state == rsi['PASSIVE'] ) & ( o_state <= rsi['PASSIVE'] ) ) ] = True
                    # <= PASSIVE means not in ['REJECTED', 'PAS REJ', 'OBS ONLY', 'DISMISS']

                elif state == 'obs_only' : # "observed only" state, all report state are allowed
                    use_obs[ where( o_state==rsi['OBS ONLY'] ) ] = True

                elif '/' in state : # combination of report and observation state
                    r_st, o_st = state.split('/')
                    use_obs[ where( ( r_state==rsi[r_st.upper().replace('_',' ')] ) & ( o_state==rsi[o_st.upper().replace('_',' ')] ) ) ] = True

                elif state.startswith('default:') : # complex rule
                    # of the form "default:<state>,<OBSTYPE1>:<state2>[,<OBSTYPE1>:<state2>]..."

                    toks = state.split(',')
                    defstate = toks[0].split(':')[1]
                    # print 'default rule: ', defstate
                    if defstate == 'active' :
                        use_obs[ where( ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True
                    elif defstate == 'passive' :
                        use_obs[ where( ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True
                    elif defstate == 'valid' :
                        use_obs[ where( ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True
                        use_obs[ where( ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True
                    else :
                        raise ValueError( 'ERROR : Unknown state filter %s.' % defstate )

                    for tok in toks[1:] :
                        ot, st = tok.split(':')
                        # print 'special rule: %s --> %s' % (ot,st)
                        use_obs[ where( r_obstype==self.table['obstypes_inv'][ot] ) ] = False
                        if st == 'active' :
                            use_obs[ where( (r_obstype==self.table['obstypes_inv'][ot]) & ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True
                        elif st == 'passive' :
                            use_obs[ where( (r_obstype==self.table['obstypes_inv'][ot]) & ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True
                        elif st == 'valid' :
                            use_obs[ where( (r_obstype==self.table['obstypes_inv'][ot]) & ( r_state==rsi['ACTIVE'] ) & ( o_state==rsi['ACTIVE'] ) ) ] = True
                            use_obs[ where( (r_obstype==self.table['obstypes_inv'][ot]) & ( o_state==rsi['PASSIVE'] ) & ( (o_check==rfi['NONE']) ) & ( r_state==rsi['PASSIVE'] ) & ( (r_check==rfi['NONE']) ) ) ] = True
                        else :
                            raise ValueError( 'ERROR : Unknown state filter %s.' % st )

                else :
                    raise ValueError( 'ERROR : Unknown state filter %s.' % state )
            else :
                use_obs[:] = True

            # obstype filter ...........................................................................................
            if ('obstype' in ffilter) \
                and (not ffilter['obstype'] is None) and (ffilter['obstype'] != 'all') and (ffilter['obstype'] != '') :

                ot = self.data['obstype'][ self.data['i_hdr'] ] # expand to n_body size
                # remove entries with non-matching obstype
                use_obs[ where(ot != self.table['obstypes_inv'][ffilter['obstype']]) ] = False

            # codetype filter ..........................................................................................
            if ('codetype' in ffilter) \
                and (not ffilter['codetype'] is None) and (ffilter['codetype'] != 'all') and (ffilter['codetype'] != '') :

                ct = self.data['codetype'][ self.data['i_hdr'] ] # expand to n_body size
                # remove entries with non-matching codetype
                use_obs[ where(ct != self.table['codetypes_inv'][ffilter['codetype']]) ] = False

            # varname filter ...........................................................................................
            if ('varname' in ffilter) \
                and (not ffilter['varname'] is None) and (ffilter['varname'] != 'all') and (ffilter['varname'] != '') :

                # remove entries with non-matching variable name
                use_obs[ where( self.data['varno'] != self.table['varnames_inv'][ffilter['varname']]) ] = False

            # area filter ..............................................................................................
            if ('area' in ffilter) \
                and (not ffilter['area'] is None) and (ffilter['area'] != 'all') and (ffilter['area'] != '') :
                from kendapy.area import is_in_area
                outside = invert( is_in_area( self.data['lat'][ self.data['i_hdr'] ],
                                              self.data['lon'][ self.data['i_hdr'] ], ffilter['area'] ) )
                use_obs[ where(outside) ] = False

            # time filter ..............................................................................................
            if ('time' in ffilter) \
                and (not ffilter['time'] is None) and (ffilter['time'] != 'all') and (ffilter['time'] != '') :

                # time is assumed to be a string of the form "<mfirst>[,<mlast>]",
                # where <mfirst> and <mlast> are in minutes and integer

                time = ffilter['time']
                if ',' in time :
                    mfirst, mlast = list(map( int, time.split(',') ))
                else :
                    mfirst = mlast = int(time)
                otime = self.data['time'][ self.data['i_hdr'] ]
                use_obs[ where( (otime < mfirst) | (otime > mlast) ) ] = False

            # level filter ..............................................................................................
            if ('level' in ffilter) \
                and (not ffilter['level'] is None) and (ffilter['level'] != 'all') and (ffilter['level'] != '') :

                # level is assumed to be a string of the form "<mfirst>[,<mlast>]",
                # where <mfirst> and <mlast> are integer values

                level = ffilter['level']
                if ',' in level :
                    mfirst, mlast = list(map( int, level.split(',') ))
                else :
                    mfirst = mlast = int(level)
                olevel = array(self.data['level'],dtype=int)
                use_obs[ where( (olevel < mfirst) | (olevel > mlast) ) ] = False

            # pressure filter ..........................................................................................
            if ('pressure' in ffilter) \
                and (not ffilter['pressure'] is None) and (ffilter['pressure'] != 'all') and (ffilter['pressure'] != '') :

                # pressure is assumed to be a string of the form "<plow>[,<phigh>]",
                # where <plow> and <phigh> are in hPa and integer

                pressure = ffilter['pressure']
                if ',' in pressure :
                    plow, phigh = list(map( int, pressure.split(',') ))
                else :
                    plow = phigh = int(pressure)
                if plow > phigh :
                    plow, phigh = phigh, plow
                opressure = self.data['plevel'] / 100 # [hPa]  plevel is a body field

                if ma.is_masked(opressure) :
                    # observations with masked pressure level (e.g. SYNOP/PS) should not be removed...
                    pidcs = where( (opressure.mask==False) & ((opressure < plow) | (opressure > phigh)) )
                else :
                    pidcs = where(                           ((opressure < plow) | (opressure > phigh)) )

                use_obs[ pidcs ] = False

            # station filter ..........................................................................................
            if ('station' in ffilter) \
                and (not ffilter['station'] is None) and (ffilter['station'] != 'all') and (ffilter['station'] != '') :

                # convert 10-Byte arrays to strings
                statid_hdr = self.statids(all=True) #[ self.data['statid'][i,:].tostring().decode("ascii").strip() for i in range(self.n_hdr)]
                # remove observations with non-matching station id
                the_station = ffilter['station'].strip()
                for i in range(self.n_body) :
                    if statid_hdr[self.data['i_hdr'][i]] != the_station :
                        use_obs[i] = False

            # report filter ..........................................................................................
            if ('report' in ffilter) \
                and (not ffilter['report'] is None) and (ffilter['report'] != 'all') and (ffilter['report'] != '') :

                # remove entries with non-matching report id
                use_obs[ where( self.data['i_hdr'] != int(ffilter['report']) ) ] = False

            # select indices that passed all filters and store them in a cache dictionary
            idcs = where( use_obs == True )
            self.filtered_idcs[filtername] = idcs

            # ..........................................................................................................

        if return_name :
            return idcs, filtername
        else :
            return idcs

    #-------------------------------------------------------------------------------------------------------------------
    def statids(self,all=False,**filter):
        if all :
            return [ self.data['statid'][i,:].tostring().decode("ascii").strip() for i in range(self.n_hdr)]
        else :
            return [ self.data['statid'][self.data['i_hdr'][i],:].tostring().decode("ascii").strip() for i in self.filter_indices( **filter )[0] ]


    #-------------------------------------------------------------------------------------------------------------------
    def filtername2dict( self, filtername, debug=False ) :
        #print '>>>%s<<<', filtername
        if filtername is None or filtername == '' :
            d = {}
        else :
            d = dict( [x.split('=') for x in filtername.strip().split()] )
        return d

    #-------------------------------------------------------------------------------------------------------------------
    def dict2filtername( self, d, remove_empty=True ) :
        """
        Convert filter dictionary to filter string.
        Optionally remove all empty filters -> string will be unique.
        :param d: dictionary with filter settings
        :param remove_empty: If True, do not include empty filter settings in the string
        :return: filter name string
        """

        if remove_empty :
            dd = {}
            for f in d :
                if (not d[f] is None) and (d[f] != '') and (d[f] != 'all') :
                    dd[f] = d[f]
        else :
            dd = d

        return ' '.join([ '%s=%s'%(v,dd[v]) for v in sorted(dd) ])

    #-------------------------------------------------------------------------------------------------------------------
    def clear_filter( self ) :
        """Remove all filters"""
        self.replace_filter( filter={} )

    #-------------------------------------------------------------------------------------------------------------------
    def add_filter( self, **filter ) :
        self.set_filter( add=True, **filter )

    #-------------------------------------------------------------------------------------------------------------------
    def replace_filter( self, **filter ) :
        self.set_filter( add=False, **filter )

    #-------------------------------------------------------------------------------------------------------------------
    def set_filter( self, add=True, **filter ) :
        """
        Set filter
        :param filter: dictionary, string or None 
        """

        self.idcs, self.filtername = self.filter_indices( add=add, return_name=True, **filter )
        self.filter = self.filtername2dict( self.filtername )

        self.obstypes  = [ self.table['obstypes'][i]  for i in set(self.data['obstype'][self.data['i_hdr']][self.idcs]) ]
        self.codetypes = [ self.table['codetypes'][i] for i in set(self.data['codetype'][self.data['i_hdr']][self.idcs]) ]
        self.varnames  = [ self.table['varnames'][i]  for i in set(self.data['varno'][self.idcs]) ]

        if self.verbose :
            if add :
                print(('-- set_filter : adding ', filter if not filter is None else 'None'))
            print(('-- set_filter : filter is now %s -> using %d of %d observations' \
                                % ( self.filtername, len(self.idcs[0]), self.n_body )))
            print(('-- obstypes  left after filtering : ', self.obstypes))
            print(('-- codetypes left after filtering : ', self.codetypes))
            print(('-- variables left after filtering : ', self.varnames))

    #-------------------------------------------------------------------------------------------------------------------
    def get_filter( self ) :
        return self.filter

    #-------------------------------------------------------------------------------------------------------------------
    def identify( self, allowed=None, time_shift=0, filtername=None ) :
        """
        Compute unique id for all observations passing the current filter settings
        
        :param allowed: dictionary with allowed observation ids
                        If allowed is None, a uniqueness-test weill be performed and all non-unique observations
                        will be ignored
        :return: dictionary with observation ids as keys and observation indices as values
                 If allowed != None, additionally a list of missing observation ids will be returned 
        """

        idcs_2bremoved = []
        oids_2bremoved = []

        nobs_in = self.idcs[0].size
        if nobs_in == 0 :
            return None, None

        oids = {}
        for k in range(nobs_in) :
            i = self.idcs[0][k]           # body index
            j = self.data['i_hdr'][i]  # header index

            # compute observation identifier
            try :
                oid = "%d,%d,%d,%d,%09.5f,%09.5f,%d,%f" % (
                    self.data['obstype'][j],
                    self.data['codetype'][j],
                    self.data['instype'][j] if not (type(self.data['instype'][j]) == np.ma.core.MaskedConstant) else 0,
                    self.data['time'][j]+time_shift,
                    self.data['lat'][j],
                    self.data['lon'][j],
                    self.data['varno'][i],
                    self.data['level'][i] )
            except :
                print('FAILED TO CONSTRUCT IDENTIFIER FROM')
                print( self.data['obstype'][j], self.data['codetype'][j],
                       self.data['instype'][j], self.data['time'][j]+time_shift,
                       self.data['lat'][j], self.data['lon'][j],
                       self.data['varno'][i], self.data['level'][i] )
                #print( type(self.data['instype'][j]) )
                sys.exit(-1)

            if allowed is None : # no restrictions

                if oid in oids : # non-unique
                    i2 = oids[oid]
                    j2 = self.data['i_hdr'][i2]
                    print('- '*40)
                    print('WARNING: ENCOUNTERED NON-UNIQUE OBS ID ', oid)
                    for hp in self.hdr_vars : #['obstype','codetype','instype','time','lat','lon'] :
                        if hp != 'statid' and self.data[hp][j] != self.data[hp][j2] :
                            print( hp, self.data[hp][j], self.data[hp][j2] )
                    for bp in self.body_vars :
                        if self.data[bp][i] != self.data[bp][i2] :
                            print( bp, self.data[bp][i], self.data[bp][i2] )
                    print('- '*40)
                    #sys.exit(0)
                    oids_2bremoved.append(oid)

                else : # unique (so far)
                    oids[oid] = i

            else : # ignore unknown ids

                if oid in allowed :
                    oids[oid] = i
                else :
                    oids_2bremoved.append(oid)

        # remove non-unique observation ids
        if allowed is None :
            for oid in oids_2bremoved :
                if oid in oids :
                    oids.pop(oid)
                else :
                    print('WARNING: ID TO BE REMOVED IS ALREADY GONE : ',oid)

        # create & activate new filter with the specified name (without the unknown or non-unique observation)
        if not filtername is None :
            self.idcs = (array(list(oids.values())),)
            self.filtername = filtername
            self.filter['name'] = filtername
            self.filtered_idcs[filtername] = self.idcs
            self.oids[filtername] = oids

        return oids, oids_2bremoved

    #-------------------------------------------------------------------------------------------------------------------
    def activate_exsisting_filter( self, filtername ) :
        """Activate existing named filter (e.g. generated by identify)"""
        if filtername in self.filtered_idcs :
            self.idcs = self.filtered_idcs[filtername]
            self.filtername = filtername
            self.filter['name'] = filtername
        else :
            raise ValueError('Filter does not exist : '+filtername)

    #-------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def filter_common( ekfs, filtername='common', verbose=True ) :
        """Select a common subset of observations in all elements of the list of Ekf objects provided as input."""

        n_left = []
        oids = None

        # forward pass
        for iekf, ekf in enumerate(ekfs) :
            oids, oids_2bremoved = ekf.identify( allowed=oids, filtername=filtername )
            if oids is None :
                break
            n_left.append(len(oids))
            if verbose : print( '{}>{} '.format(iekf, len(oids)), end='')

        # backward pass                       * here '>2' was in use until 2021/6/7
        if (not (oids is None)) and (len(ekfs)>1) and (n_left[0] > n_left[-1]) : # if we did not loose any observations, the second step is not necessary..
            for iekf, ekf in enumerate(ekfs[::-1]) :
                oids, oids_2bremoved = ekf.identify( allowed=oids, filtername=filtername )
                if verbose : print( '{}<{} '.format(iekf, len(oids)), end='')
        if verbose : print()

        return oids


    #-------------------------------------------------------------------------------------------------------------------
    def n_obs( self, **filter ) :
        """Return number of filtered observations"""

        idcs = self.filter_indices( **filter )
        return len(idcs[0])


    #-------------------------------------------------------------------------------------------------------------------
    def obs( self, param='obs', **filter ) :
        """Retrieve observation data"""

        idcs = self.filter_indices( **filter )

        if param in self.body_vars :
            #print param + ' is body var', self.data[param].size
            retval = self.data[param][idcs]

        elif param in self.hdr_vars :
            #print param + ' is hdr var', self.data[param].size, self.data['i_hdr'].size
            retval = self.data[param][self.data['i_hdr']][idcs]

        else :
            raise ValueError( 'ERROR: Neither a header nor a body variable : '+param )

        return retval


    #-------------------------------------------------------------------------------------------------------------------
    def n_rep(self, **filter ) :
        """Return number of reports"""
        return len(self.reports(**filter))


    #-------------------------------------------------------------------------------------------------------------------
    def reports(self, **filter) :
        return list(set(self.obs(param='i_hdr',**filter)))


    #-------------------------------------------------------------------------------------------------------------------
    def veri_available( self, veri ) :

        r = None

        # return True if veri si available
        if veri == 'anamean'  : r = self.i_anamean  >= 0
        if veri == 'fgmean'   : r = self.i_fgmean   >= 0
        if veri == 'fgspread' : r = self.i_fgspread >= 0
        if veri == 'fgdet'    : r = self.i_fgdet    >= 0
        if veri == 'anadet'   : r = self.i_anadet   >= 0
        if veri == 'det'      : r = self.i_det      >= 0
        if veri == 'fg'       : r = (self.i_fgens_last >= 0) and ((self.i_fgens_first == self.i_fgens_last) or (self.i_fgens_first < 0))
        if veri == 'fgens'    : r = (self.i_fgens_first >= 0) and (self.i_fgens_last > 0)
        if veri == 'anaens'   : r = self.i_anaens_last >= 0

        # special case deterministic main forecasts: return initial times
        if veri == 'maindet' : r = sorted(list(self.i_maindet.keys()))

        if r is None :
            raise ValueError('Unknown veri type '+veri )

        return r

    #-------------------------------------------------------------------------------------------------------------------
    def veri( self, veriname, inidate=None, leadtime=None, **filter ) :
        """Retrieve verification data"""

        idcs = self.filter_indices( **filter )

        ens = False
        if veriname == 'anamean' :
            i_veri = self.i_anamean
        elif veriname == 'fgmean' :
            i_veri = self.i_fgmean
        elif veriname == 'fgspread' :
            i_veri = self.i_fgspread
        elif veriname == 'fgdet' :
            i_veri = self.i_fgdet
        elif veriname == 'anadet' :
            i_veri = self.i_anadet
        elif veriname == 'det' :
            i_veri = self.i_det
        elif veriname == 'anaens'   :
            i_veri_first = self.i_anaens_first
            i_veri_last  = self.i_anaens_last
            ens = True
        elif veriname == 'fgens'   :
            i_veri_first = self.i_fgens_first
            i_veri_last  = self.i_fgens_last
            ens = True
        elif veriname == 'fg' : # for FOF files -- there is only one member
            i_veri = self.i_fgens_first
        elif veriname == 'maindet' :
            if not inidate is None :
                i_veri = self.i_maindet[inidate]
            elif not leadtime is None :
                from kendapy.bacy_utils import t2str, to_datetime, to_timedelta
                inidate = t2str( to_datetime(self.ref_time) - to_timedelta(leadtime) )
                i_veri = self.i_maindet[inidate]
            else :
                raise ValueError('Initialization or lead time must be specified for main forecast')    
        else :
            raise ValueError('Unknown verification type %s' % veriname)

        if ens :
            #print '>>> veri ', veriname, idcs[0][:5], i_veri_first, i_veri_first+self.nens-1
            result = zeros(( self.nens, len(idcs[0]) ))

            if veriname == 'anaens' :
                ensmean =  self.data['veri_data'][self.i_anamean,:][idcs]
            else :
                ensmean = 0.0

            for i in range(self.nens) :
                result[i,:] = self.data['veri_data'][i_veri_first+i,:][idcs] + ensmean
        else :
            result = self.data['veri_data'][i_veri,:][idcs]

        return result

    # abbreviations for veri
    def anamean(  self, **filter ) : return self.veri( 'anamean',  **filter )
    def anaens(   self, **filter ) : return self.veri( 'anaens',   **filter )
    def anadet(   self, **filter ) : return self.veri( 'anadet',   **filter )
    def det(      self, **filter ) : return self.veri( 'det',      **filter )
    def fgmean(   self, **filter ) : return self.veri( 'fgmean',   **filter )
    def fgens(    self, **filter ) : return self.veri( 'fgens',    **filter )
    def fgdet(    self, **filter ) : return self.veri( 'fgdet',    **filter )
    def fg(       self, **filter ) : return self.veri( 'fg',       **filter )
    def maindet(  self, **filter ) : return self.veri( 'maindet',  **filter )

    def spread( self, veriname, **filter ) :

        if veriname in ['fg','fgens'] :
            if self.veri_available('fgspread') :
                sprd = self.veri( 'fgspread', **filter )
            elif self.veri_available('fgens') :
                sprd = self.fgens(**filter).std(axis=0,ddof=1)
            else :
                raise IOError('Cannot determine first guess spread -- neither fgspread nor fgens are available')

        elif veriname in ['ana','anaens'] :
            #if self.veri_available('anaspread') :
            #    sprd = self.veri( 'anaspread', **filter )
            if self.veri_available('anaens') :
                sprd = self.anaens(**filter).std(axis=0,ddof=1)
            else :
                raise IOError('Cannot determine analysis spread -- neither anaspread nor anaens are available')

        else :
            raise ValueError( 'Unknown verification type '+veriname )

        return sprd

    def fgspread(  self, **filter ) : return self.spread( 'fg', **filter )
    def anaspread( self, **filter ) : return self.spread( 'ana', **filter )

    #-------------------------------------------------------------------------------------------------------------------
    def departures( self, veriname, normalize=False, e_o=None, **filter ) :

        #idcs = self.filter_indices( **filter )
        obs = self.obs( **filter )
        ens = self.veri( veriname, **filter )

        if veriname.endswith('mean') or veriname.endswith('det') : # departures of ensemble mean
            if normalize : # divide departures by expectation value
                if e_o is None :
                    e_o = self.obs( param='e_o', **filter )
                dep = (ens - obs) / sqrt( e_o**2 + self.spread( veriname.replace('mean','').replace('det',''), **filter )**2 )
            else :
                dep = ens - obs

        else : # departures of members
            newshape = [1]+list(obs.shape)
            if normalize : # divide departures by expectation value
                if e_o is None :
                    e_o = self.obs( param='e_o', **filter )
                dep = (ens - obs.reshape(newshape)) / sqrt( e_o.reshape(newshape)**2 + self.spread( veriname, **filter )**2 )
            else :
                dep = ens - obs.reshape(newshape)

        return dep

    # abbreviations for departures
    def anadep(      self, **filter ) : return self.departures( 'anaens', **filter )
    def fgdep(       self, **filter ) : return self.departures( 'fgens',  **filter )
    def anadep_norm( self, **filter ) : return self.departures( 'anaens', normalize=True, **filter )
    def fgdep_norm(  self, **filter ) : return self.departures( 'fgens',  normalize=True, **filter )

    def anameandep(      self, **filter ) : return self.departures( 'anamean', **filter )
    def fgmeandep(       self, **filter ) : return self.departures( 'fgmean',  **filter )
    def anameandep_norm( self, **filter ) : return self.departures( 'anamean', normalize=True, **filter )
    def fgmeandep_norm(  self, **filter ) : return self.departures( 'fgmean',  normalize=True, **filter )

    def anadetdep(      self, **filter ) : return self.departures( 'anadet', **filter )
    def fgdetdep(       self, **filter ) : return self.departures( 'fgdet',  **filter )
    def anadetdep_norm( self, **filter ) : return self.departures( 'anadet', normalize=True, **filter )
    def fgdetdep_norm(  self, **filter ) : return self.departures( 'fgdet',  normalize=True, **filter )

    #-------------------------------------------------------------------------------------------------------------------
    def increments( self, veritype, ana=None, **filter ) :
        """
        veritype : mean|det|ens
        """
        fg  = self.veri( 'fg' +veritype, **filter )
        if ana is None :
            ana_ = self.veri( 'ana'+veritype, **filter )
        else :
            ana_ = ana
        return ana_ - fg

    #-------------------------------------------------------------------------------------------------------------------
    def statistics( self, recompute=False, desroz=True, rankhist=True, verbose=False, **filter ) :

        import kendapy.ensstat as ensstat

        if verbose : print('computing statistics...')

        idcs, filtername = self.filter_indices( return_name=True, **filter )

        if verbose : print(('Ekf.statistics : processing %d observations...' % len(idcs[0])))
        if verbose : print(('Ekf.statistics : filter keywords = ', filter))

        if filtername in list(self.stat.keys()) and not recompute :
            stat = self.stat[filtername]

        else :
            stat = {'n_obs':len(idcs[0])}

            # get index of last observation time
            obstimes = self.obs(param='time',**filter)
            tcrit = percentile(obstimes,99)
            lastobs = where(obstimes>=tcrit-1e-6)

            for veri in ['anamean','fgmean'] :
                if self.veri_available(veri) : #veri in var[vname].keys() :
                    stat[veri] = {}
                    stat[veri]['rmse'] = sqrt( ((self.veri(veri,**filter) - self.obs(**filter))**2).mean() )
                    stat[veri]['bias'] =       ( self.veri(veri,**filter) - self.obs(**filter)    ).mean()
                    if verbose : print(('    %8s :   RMSE %f   BIAS %f' % ( veri, stat[veri]['rmse'], stat[veri]['bias'] )))

                    # get rmse for the last observation time (i.e. for RAD the last satellite image)
                    stat[veri]['rmse_last'] = sqrt( ((self.veri(veri,**filter)[lastobs] - self.obs(**filter)[lastobs])**2).mean() )
                    stat[veri]['bias_last'] =       ( self.veri(veri,**filter)[lastobs] - self.obs(**filter)[lastobs]    ).mean()
                    #print '>>>>>>>>>> last obs. ', obstimes.min(), obstimes.max(), tcrit, len(lastobs[0]), len(obstimes), stat[veri]['rmse'], stat[veri]['rmse_last']

                #else :
                #    print 'NOT AVAILABLE IN ', filter, ' ::: ', veri
                #    sys.exit(-1)

            # Definition of spread : See http://journals.ametsoc.org/doi/full/10.1175/JHM-D-14-0008.1
            #                        --> shoud be (1) square root of mean variance, not (2) mean square root of variance
            #                        However fgspread from LETKF output agrees with (2)

            veri = 'fgspread'
            if self.veri_available(veri) :
                stat[veri] = {}
                #stat[veri]['meanw'] =       self.veri(veri,**filter).mean()      # mean standard deviation (wrong def.)
                stat[veri]['mean']  = sqrt((self.veri(veri,**filter)**2).mean()) # square root of mean variance (correct)
                # spread for last observation time
                stat[veri]['mean_last']  = sqrt((self.veri(veri,**filter)[lastobs]**2).mean()) # square root of mean variance (correct)
                if verbose : print(('    %8s :   MEAN %f' % ( veri, stat[veri]['mean'] )))

            for veri in ['anaens','fgens'] :
                if self.veri_available(veri) :
                    stat[veri] = {}
                    #stat[veri]['spreadw'] = self.veri(veri,**filter).std(axis=0,ddof=1).mean()             # mean standard deviation (wrong def.)
                    stat[veri]['spread'] = sqrt( (self.veri(veri,**filter).std(axis=0,ddof=1)**2).mean() )  # square root of mean variance (correct)
                    #stat[veri]['spread2d'] = self.veri(veri,**filter).std(axis=0,ddof=1)                    # spread for each pixel
                    stat[veri]['spread_last'] = sqrt( (self.veri(veri,**filter).std(axis=0,ddof=1)[lastobs]**2).mean() )
                    #stat[veri]['spreadskill'] = stat[veri]['spread'] / stat[veri.replace('ens','mean')]['rmse']
                    #print '>>>>>>>>>>> spreadskill shapes ',  stat[veri]['spread'].shape, stat[veri.replace('ens','mean')]['rmse'].shape, self.obs(param='e_o',**filter).shape
                    stat[veri]['spreadskill'] = sqrt( (self.veri(veri,**filter).std(axis=0,ddof=1)**2 + self.obs(param='e_o',**filter)**2).mean() ) / sqrt( ((self.fgmean(**filter)-self.obs(**filter))**2).mean() )
                    stat[veri]['bias']   = (self.veri(veri,**filter) - self.obs(**filter).reshape((1,self.obs(**filter).size))).mean()
                    stat[veri]['min']    = self.veri(veri,**filter).min()
                    stat[veri]['mean']   = self.veri(veri,**filter).mean()
                    stat[veri]['max']    = self.veri(veri,**filter).max()
                    if verbose : print(('    %8s : SPREAD %f  BIAS %f  MIN %f  MEAN %f  MAX %f' % ( veri, stat[veri]['spread'], stat[veri]['bias'],
                                                                                  stat[veri]['min'], stat[veri]['mean'], stat[veri]['max'])))
                    #print '    %8s : SPREAD %f (according to wrong definition)' % ( veri, stat[veri]['spreadw'] )

            veri = 'obs'
            stat[veri] = {}
            stat[veri]['minerr']   = self.obs(param='e_o',**filter).min()
            stat[veri]['meanerr']  = self.obs(param='e_o',**filter).mean()
            stat[veri]['maxerr']   = self.obs(param='e_o',**filter).max()
            stat[veri]['meanbcor'] = self.obs(param='bcor',**filter).mean()
            stat[veri]['min']      = self.obs(**filter).min()
            stat[veri]['mean']     = self.obs(**filter).mean()
            stat[veri]['max']      = self.obs(**filter).max()

            if verbose : print(('    %8s : MIN %f  MEAN %f  MAX %f' % ( veri, stat[veri]['min'], stat[veri]['mean'], stat[veri]['max'])))
            if verbose : print(('    %8s : MIN %f  MEAN %f  MAX %f  MEAN_BC %f' % ( 'obserr',
                                                          stat[veri]['minerr'], stat[veri]['meanerr'], stat[veri]['maxerr'],
                                                          stat[veri]['meanbcor'])))

            if desroz :
                e2 = ensstat.desroz( self.obs(**filter).reshape((1,self.obs(**filter).size)) - self.fgens(**filter),
                                          self.obs(**filter).reshape((1,self.obs(**filter).size)) - self.anaens(**filter),
                                          diagonal_only=True )
                if e2.min() < 0 :
                    print('WARNING: Negative numbers encountered in desroziers calculation -> set to zero!')
                    e2[ where( e2 < 0 ) ] = 0
                e = sqrt( e2 )
                stat['desroz'] = {}
                stat['desroz']['e_o'] = e
                stat['desroz']['e_o_mean'] = e.mean()
                stat['desroz']['spreadskill'] = sqrt( self.fgspread(**filter)**2 + e2 ) / sqrt( ((self.fgmean(**filter)-self.obs(**filter))**2).mean() )
                stat['desroz']['spreadskill_mean'] = stat['desroz']['spreadskill'].mean()
                idcs = where(e > 1e-10)
                if verbose : print('%12s : MIN %f  MEAN %f  MAX %f' % ( 'desroz', e[idcs].min(), e[idcs].mean(), e[idcs].max() ))

            if rankhist :
                for veri in ['anaens','fgens'] :
                    stat[veri]['rankhist'] = ensstat.rankhist( self.obs(**filter), self.veri(veri,**filter) )
                    if verbose : print((('  %s rank hist. : '%veri), stat[veri]['rankhist']))
            
            self.stat[filtername] = stat

        return stat

    #-------------------------------------------------------------------------------------------------------------------
    def find_reports_containing( self, varname ) :
        f = self.get_filter()
        self.add_filter(filter='varname=%s'%varname)
        ireps = sorted(list(set(self.obs(param='i_hdr'))))
        self.replace_filter(filter=f)
        return ireps


    # def get_reports( self, varnames=['T','U','V','RH'] ) :
    #     """Return all reports containing all of the specified variables"""
    #
    #     self.replace_filter(filter='state=all varname=%s'%varnames[0])
    #     ireps = sorted(list(set(self.obs(param='i_hdr'))))
    #
    #     rpts = []
    #     for irep in ireps :
    #         self.replace_filter(filter='state=all varname=%s report=%d'%(varnames[0],irep))
    #         print irep, self.obs(param='lat')[0], self.obs(param='lon')[0], self.obs(param='time')[0]
    #         rpt = {'i_hdr':irep, 'lat':self.obs(param='lat')[0], 'lon':self.obs(param='lon')[0], 'time':self.obs(param='time')[0]}
    #         for varname in varnames :
    #             self.replace_filter(filter='state=all varname=%s report=%d'%(varname,irep))
    #             rpt[varname] = { 'plevel':self.obs(param='plevel'), 'obs':self.obs(), 'fgmean':self.fgmean(), 'anamean':self.anamean() }
    #         rpts.append(rpt)
    #
    #     return rpts


#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": # -------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

    import argparse
    parser = argparse.ArgumentParser(description='Parse EKF files generated by KENDA experiment')

    parser.add_argument( '-F', '--filter',          dest='filter',          default='state=active', help='filter string (default: "state=active")' )
    parser.add_argument( '-S', '--state-filter',    dest='state_filter',    default='none', help='observation state filter [active|passive|valid(default)]' )
    parser.add_argument( '-O', '--obstype-filter',  dest='obstype_filter',  default='none', help='observation type filter' )
    parser.add_argument( '-C', '--codetype-filter', dest='codetype_filter', default='none', help='code type filter' )
    parser.add_argument( '-A', '--area-filter',     dest='area_filter',     default='none', help='area filter' )
    parser.add_argument( '-T', '--time-filter',     dest='time_filter',     default='none', help='time filter' )
    parser.add_argument( '-L', '--level-filter',    dest='level_filter',    default='none', help='level filter' )
    parser.add_argument( '-P', '--pressure-filter', dest='pressure_filter', default='none', help='pressure filter' )
    parser.add_argument( '-V', '--varname-filter',  dest='varname_filter',  default='none', help='variable name filter' )
    parser.add_argument( '-s', '--var-statistics',  dest='varstatistics', help='compute basic statistics for each variable', action='store_true' )
    parser.add_argument(       '--departure-hist',  dest='departure_hist', help='plot departure histogram', action='store_true' )
    parser.add_argument(       '--temp-locations',  dest='temp_locations', help='print TEMP locations for use in visop_i2o', action='store_true' )
    parser.add_argument( '-t', '--time',            dest='time', help='print observation time information', action='store_true' )
    parser.add_argument(       '--nobs',            dest='nobs', help='print number of observations', action='store_true' )
    parser.add_argument(       '--list-reports',    dest='list_reports', help='list reports', action='store_true' )
    parser.add_argument(       '--obs-idx',         dest='obs_idx', help='dump all information on observation with this body index', default=None, type=int )
    parser.add_argument(       '--main',            dest='main', help='print information on main forecasts', action='store_true' )
    parser.add_argument( 'ekffile', metavar='ekffile', help='ekf file name[s]', nargs='*' )
    args = parser.parse_args()

    header = True
    for f in args.ekffile :
        if not args.nobs :
            print("========================= %s ======================" % f)

        ekf = Ekf( f, filter=args.filter,
                      state_filter    = None if args.state_filter    == 'none' else args.state_filter,
                      obstype_filter  = None if args.obstype_filter  == 'none' else args.obstype_filter,
                      codetype_filter = None if args.codetype_filter == 'none' else args.codetype_filter,
                      area_filter     = None if args.area_filter     == 'none' else args.area_filter,
                      time_filter     = None if args.time_filter     == 'none' else args.time_filter,
                      level_filter    = None if args.level_filter    == 'none' else args.level_filter,
                      pressure_filter = None if args.pressure_filter == 'none' else args.pressure_filter,
                      varname_filter  = None if args.varname_filter  == 'none' else args.varname_filter,
                      verbose=False if args.nobs else True )

        if args.list_reports :
            print("REPORTS:")
            for i in ekf.reports() :
                print(' #{} station={}, lon={}, lat={}, time={}'.format( i, ekf.statids(all=True)[i], ekf.data['lon'][i], ekf.data['lat'][i], ekf.data['time'][i]) )
                print('    standard ', ekf.data['i_body'][i], ekf.data['l_body'][i])
                if 'i_spec' in ekf.data :
                    print('    radar     ', ekf.data['i_spec'][i], ekf.data['l_spec'][i])

        if args.varstatistics :
            for vn in ekf.varnames :
                print(('>>> VARIABLE %s :' % vn))
                ekf.statistics(varname_filter=vn,verbose=True)

        if args.departure_hist : # generate histogram plot of normlized departures

            # Using the departures of the ensemble mean is probably the correct way,
            # but using all the individual departures of the ensemble members would
            # provide better statistics.
            use_mean = True

            for vn in ekf.varnames :
                print('plotting normalized departures for variable '+vn+'...')
                vnf={'varname_filter':vn}
                dep_bins = linspace(-3.0,3.0,31)
                dep_bin_centers = 0.5*(dep_bins[1:]+dep_bins[:-1])
                delta_dep = dep_bins[1] - dep_bins[0]

                if use_mean :
                    fgdepn_hist  = histogram( ekf.fgmeandep_norm(**vnf), bins=dep_bins )[0]
                    anadepn_hist = histogram( ekf.anameandep_norm(**vnf), bins=dep_bins )[0]
                else :
                    fgdepn_hist  = histogram( ekf.fgdep_norm(**vnf), bins=dep_bins )[0]
                    anadepn_hist = histogram( ekf.anadep_norm(**vnf), bins=dep_bins )[0]

                if ekf.veri_available( 'fgdet' ) :
                    fgdetdepn_hist  = histogram( ekf.fgdetdep_norm(**vnf), bins=dep_bins )[0]
                if ekf.veri_available( 'anadet' ) :
                    anadetdepn_hist = histogram( ekf.anadetdep_norm(**vnf), bins=dep_bins )[0]


                gauss_hist   = np.exp(-0.5*dep_bin_centers**2)
                gauss_hist *= anadepn_hist.sum()/gauss_hist.sum()

                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(figsize=(6,6))
                ax.semilogy( dep_bin_centers, fgdepn_hist, color='b', label='FG' )
                ax.semilogy( dep_bin_centers, anadepn_hist, color='r', label='ANA' )

                if ekf.veri_available( 'fgdet' ) :
                    ax.semilogy( dep_bin_centers, fgdetdepn_hist, '--', color='b', label='FG DET' )
                if ekf.veri_available( 'anadet' ) :                    
                    ax.semilogy( dep_bin_centers, anadetdepn_hist, '--', color='r', label='ANA DET' )

                ax.semilogy( dep_bin_centers, gauss_hist, ':k', label='normal dist.' )
                ax.grid()
                ax.legend(title=vn)
                ax.set_xlabel('normalized departure')
                ax.set_ylabel('#cases')

                if use_mean :
                    fig.savefig('fg_ana_mean_dep_hist_'+vn+'.png')
                else :
                    fig.savefig('fg_ana_dep_hist_'+vn+'.png')

        if args.time :
            t = ekf.obs(param='time')
            print(('number of observations : ', t.size))
            print(('min/mean/max time : ', t.min(), t.mean(), t.max()))
            for tt in set(t) :
                print(('time ', tt, ' occurs ', len(where(t==tt)[0]), 'times..'))

        #plvl = ekf.obs(param='plevel')
        #print 'min/mean/max plevel = ', plvl.min(), plvl.mean(), plvl.max()

        if args.nobs :
            if header :
                for varname in ekf.all_varnames :
                    print('%8s  ' % varname[-8:], end=' ')
                print()
                header=False
            for varname in ekf.all_varnames :
                nobs = len(ekf.obs(varname_filter=varname,state_filter='active'))
                print('%8d  ' % nobs, end=' ')
            print()

        if args.temp_locations :
            reps = ekf.find_reports_containing('T')
            filter = ekf.get_filter()
            l=''
            for r in reps :
                ekf.add_filter(filter='report=%d'%r)
                if l != '' : l += ','
                l+= '%.2f/%.2f' % ( ekf.obs(param='lat')[0], ekf.obs(param='lon')[0] )
                ekf.replace_filter(filter=filter)
            print('latlon'+l)

        #h_loc = ekf.obs(param='h_loc')
        #print 'h_loc min/mean/max', h_loc.min(), h_loc.mean(), h_loc.max()
        #v_loc = ekf.obs(param='v_loc')
        #print 'v_loc min/mean/max', v_loc.min(), v_loc.mean(), v_loc.max()

        if not args.obs_idx is None :
            print('OBSERVATION #{}:'.format(args.obs_idx))
            for p in ekf.body_vars :
                print('body {:20s} : '.format(p), ekf.obs(param=p)[args.obs_idx] ) 
            for p in ekf.hdr_vars :
                print('hdr  {:20s} : '.format(p), ekf.obs(param=p)[args.obs_idx] )
            for p in [ 'anamean', 'fgmean', 'fgspread', 'fgdet', 'anadet', 'det', 'fgens', 'anaens', 'fg' ] :
                if ekf.veri_available(p) :
                    print('veri {:20s} : '.format(p), end='' )
                    if ekf.veri(p).ndim == 1 :
                        print( ekf.veri(p)[args.obs_idx] )
                    else :
                        print( ekf.veri(p)[...,args.obs_idx] )

        if args.main :
            inidates = ekf.veri_available('maindet') # list of available deterministic main forecast initializhation times
            for inid in inidates :
                print('main forecast started at {}: '.format(inid))
                for vn in ekf.varnames :                
                    vnf={'varname_filter':vn}
                    obs = ekf.obs(**vnf)
                    #mdl = ekf.veri('maindet',inidate=inid,**vnf)
                    mdl = ekf.maindet(inidate=inid,**vnf)
                    # example for a call with a leadtime argument
                    # mdl = ekf.veri('maindet',leadtime='2:00',**vnf)
                    rmse = np.sqrt( ((obs-mdl)**2).mean() )
                    bias = (mdl-obs).mean()
                    print('-- {:12s}: mean obs {:10.3e}, mean model {:10.3e}, rmse = {:10.3e}, bias = {:10.3e}'.format(vn,obs.mean(),mdl.mean(),rmse,bias))


