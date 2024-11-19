#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . E X P E R I M E N T
#  postprocess KENDA experiments
#
#  2016.10 L.Scheck 

from __future__ import absolute_import, division, print_function
from numpy import *
import os, sys, getpass, subprocess, argparse, time, re, gc, pickle, glob, copy, datetime, socket
from kendapy.time14 import Time14, time_range
from kendapy.ekf import Ekf, tables
from kendapy.bacy_exp import BacyExp

class Experiment(object):
    """class representing a KENDA experiment"""

    def __init__( self, value, verbose=False, dwd2lmu=False, sofia=False, runtimes=False ) :
        """Initialize object using information from the log file"""

        self.verbose = verbose

        if not os.path.exists(value) :
            raise ValueError( value, ' DOES NOT EXIST!' )
        
        #print( 'EXISTS',  value+'/modules', os.path.exists(value+'/modules'))
        if os.path.exists(value+'/modules') : # new bacy (2020)
            self.exp_type = 'bacy1'
        elif os.path.exists(value+'/letkf_config.py') : # idealised COSMO setup
            self.exp_type = 'sofia'
            sofia = True
        else : # old bacy (before 2020)
            self.exp_type = 'bacy' 

        if verbose :
            print("experiment type : "+self.exp_type)

        #...............................................................................................................

        if self.exp_type == 'sofia' :

            self.basedir = value
            self.settings = self.get_sofia_settings( value, verbose=verbose )
            self.lfcst_settings = None
            self.visop_active = True
            self.lfcst = False
            self.sfc_output = False

        if self.exp_type == 'bacy1' : # new bacy

            # gather information from config files and data/feedback directories           
            self.bacy_exp = BacyExp( value )
            config = self.bacy_exp.config

            settings = {}
            settings['exp']      = config['CY_EXPID']
            settings['EXPID']    = config['CY_EXPID']
            settings['N_ENS']    = int(config['BA_ENSIZE_ASS'])
            settings['ASSINT']   = int(config['BA_DELT_ASS'])
            settings['DATE_INI'] = self.bacy_exp.valid_times['fg'][0]  #sorted(self.bacy_exp.filetree['cycles'].keys())[0]
            settings['DATE_END'] = self.bacy_exp.valid_times['fg'][-1] #sorted(self.bacy_exp.filetree['cycles'].keys())[-1]
            settings['DATA_DIR'] = os.path.join( value, 'data' )
            settings['FEED_DIR'] = os.path.join( value, 'feedback' )
            settings['PLOT_DIR'] = settings['DATA_DIR']
            settings['VISOP_ASSIM_INTERVAL'] = 60
            settings['VISOP_OUTPUT_INTERVAL'] = 60

            self.settings = settings

            self.sfc_output = False
            self.visop_active = False
            
        else : # old bacy
            self.logfile = value
            
            self.sfc_output = True

            #if '/project/meteo' in self.logfile : dwd2lmu = True
            if socket.getfqdn().endswith('uni-muenchen.de') : dwd2lmu = True

            # read log file
            if verbose :
                print('parsing %s ...' % value)
            self.settings = self.parse_logfile( value, verbose=verbose, dwd2lmu=dwd2lmu, runtimes=runtimes )

            # add VISOP config file settings, if available
            if 'VISOP' in list(self.settings.keys()) and (self.settings['VISOP'].lower() in ['true','T'] ) :
                fname = self.settings['TOP_DIR'] +'/config/config_visop.sh'
                if verbose :
                    print()
                    print('parsing %s ...' % fname)
                if os.path.exists(fname) :
                    self.settings = self.parse_logfile( fname, verbose=verbose, dwd2lmu=dwd2lmu,
                                                        settings=self.settings, strip_comments=True, runtimes=False )
                else :
                    if verbose : print("Warning: Cannot find "+fname)
                self.visop_active = True
            else :
                self.visop_active = False

            # gather information on long (not cycling) forecasts
            fname = self.settings['TOP_DIR'] +'/config/config_forecast.sh'
            if verbose :
                print()
                print('parsing %s ...' % fname)
            if os.path.exists(fname) :
                self.lfcst_settings = self.parse_logfile( fname, verbose=verbose, dwd2lmu=dwd2lmu, strip_comments=True,
                                                    settings=copy.copy(self.settings), overwrite=True, runtimes=False )

                self.lfcst_start_times = time_range( self.lfcst_settings['DATE_INI'], self.lfcst_settings['DATE_END'], self.lfcst_settings['FCINT'] )
                self.lfcst_end_times = [str(Time14(t)+Time14(self.lfcst_settings['FCTIME'])) for t in self.lfcst_start_times]

                self.lfcst_sfc_output_times = [str(Time14(str(t))) for t in range( 0, int(self.lfcst_settings['FCTIME'])+1, int(3600*float(self.lfcst_settings['SFC_OUTPUT_INTERVAL'])) )]
                # FCTIME is in seconds, SFC_OUTPUT_INTERVAL in hours, lfcst_sfc_output_times are ddhhmmss strings
                self.lfcst_sfc_output_times_min = [Time14(t).daymin() for t in self.lfcst_sfc_output_times]

                if self.visop_active :
                    self.lfcst_visop_output_times = [str(Time14(str(t))) for t in range(0,int(self.lfcst_settings['FCTIME'])+1,int(self.settings['VISOP_OUTPUT_INTERVAL'])*60)]
                    self.lfcst_visop_output_times_min = [Time14(t).daymin() for t in self.lfcst_visop_output_times]

                if verbose :
                    print('long fcsts : Between %s and %s, interval %s seconds, duration %s seconds' % ( \
                        self.lfcst_settings['DATE_INI'], self.lfcst_settings['DATE_END'],
                        self.lfcst_settings['FCINT'], self.lfcst_settings['FCTIME'] ))
                    print('             start times : ', self.lfcst_start_times)
                    print('             end   times : ', self.lfcst_end_times)
                    print('         sfc times [min] : ', self.lfcst_sfc_output_times_min)
                    if self.visop_active :
                        print('       visop times [min] : ', self.lfcst_visop_output_times_min)
                self.lfcst = True
                if not 'VISOP_ASSIM_INTERVAL' in list(self.settings.keys()) :
                    if not 'VISOP_ASSIM_INTERVAL' in list(self.lfcst_settings.keys()) :
                        self.settings['VISOP_ASSIM_INTERVAL'] = '15'
                    else :
                        self.settings['VISOP_ASSIM_INTERVAL'] = self.lfcst_settings['VISOP_ASSIM_INTERVAL']

            else :
                if not 'VISOP_ASSIM_INTERVAL' in list(self.settings.keys()) :
                    self.settings['VISOP_ASSIM_INTERVAL'] = '15'
                if verbose : print("Warning: Cannot find "+fname)
                self.lfcst_settings = None
                self.lfcst = False
        #...............................................................................................................

        self.expid = self.settings['EXPID']

        # number of ensemble members
        self.n_ens = int(self.settings['N_ENS'])

        # cache dictionaries
        self.ekfs  = {}
        self.cosmo = {}
        self.visop = {}

        # fcst start/end/output times
        self.fcst_start_times = time_range( self.settings['DATE_INI'], self.settings['DATE_END'], self.settings['ASSINT'] )
        self.veri_times = time_range( str( Time14(self.settings['DATE_INI']) + Time14(self.settings['ASSINT']) ),
                                      str( Time14(self.settings['DATE_END']) + Time14(self.settings['ASSINT']) ),
                                      self.settings['ASSINT'] )
        self.fcst_end_times = self.veri_times
        self.all_times = self.fcst_start_times + [self.veri_times[-1]]

        if self.sfc_output :
            self.sfc_output_times = [str(Time14(str(t))) for t in range( 0, int(self.settings['ASSINT'])+1, int(3600*float(self.settings['SFC_OUTPUT_INTERVAL'])) )]
            # ASSINT is in seconds, SFC_OUTPUT_INTERVAL in hours, sfc_output_times are ddhhmmss strings
            self.sfc_output_times_min = [Time14(t).daymin() for t in self.sfc_output_times]

        if self.visop_active :
            self.visop_output_times = [str(Time14(str(t))) for t in range(0,int(self.settings['ASSINT'])+1,int(self.settings['VISOP_OUTPUT_INTERVAL'])*60)]
            self.visop_output_times_min = [Time14(t).daymin() for t in self.visop_output_times]

        if verbose :
            print()
            print('EXPERIMENT SETTINGS ------------------')
            print('experiment type = ', self.exp_type)
            for k, v in self.settings.items() :
                print("%20s = %s" % (k,v))
            print()
            if not self.lfcst_settings is None :
                print()
                print('FORECAST SETTINGS ------------------')
                for k, v in self.lfcst_settings.items() :
                    if not k in self.settings or self.settings[k] != self.lfcst_settings[k] :
                        print("%20s = %s" % (k,v))
                print()
            print('--------------------------------------')
            print()
            print('--> forecasts started at ', self.fcst_start_times)
            print('    analyses  valid   at ', self.veri_times)
            if self.sfc_output :
                print('         sfc times [min] : ', self.sfc_output_times_min)
            if self.visop_active :
                print('       visop times [min] : ', self.visop_output_times_min)
            if not self.lfcst_settings is None :
                print('--> long forecasts started at ', self.lfcst_start_times)

    #-------------------------------------------------------------------------------------------------------------------
    def get_fcst_start_times( self, tmin=None, tmax=None, lfcst=False ) :

        if self.exp_type == 'bacy1' :
            all_times = self.bacy_exp.valid_times['fg']
        else :
            all_times = self.lfcst_start_times if lfcst else self.fcst_start_times

        t_min = tmin if tmin != '' else None
        t_max = tmax if tmax != '' else None
        if t_min is None and t_max is None :
            return all_times
        else :
            times = []
            for t in all_times :
                valid = True
                if not t_min is None :
                    if Time14(t) < Time14(t_min) :
                        valid = False
                if not t_max is None :
                    if Time14(t) > Time14(t_max) :
                        valid = False
                if valid :
                    times.append(t)
            return times

    #--------------------------------------------------------------------------------
    def get_sofia_settings( self, basedir, verbose=False ) :

        settings = {}

        #for d in ['output','data','data/ekf','letkf_config.py','letkf_template.py'] :
        #    if not os.path.exists( basedir+'/'+d ) :
        #        raise IOError(basedir+'/'+d+' does not exist')

        globs = dict()
        locs  = dict()
        exec(compile(open( basedir+'/letkf_config.py' ).read(), basedir+'/letkf_config.py', 'exec'),   {}, locs)
        exec(compile(open( basedir+'/letkf_template.py' ).read(), basedir+'/letkf_template.py', 'exec'), {}, locs)

        if False :
            print()
            print('------ information obtained from letkf_config.py and letkf_template.py in %s ------' % basedir)
            for g in locs :
                print(g, locs[g])
            print('-----------------------------------------------------------------------------------------------')
            print()

        settings['exp']      = locs['expid']
        settings['EXPID']    = locs['expid']
        settings['N_ENS']    = "%d" % locs['NMem']
        settings['ASSINT']   = "%d" % (locs['assint']*60)
        settings['DATE_INI'] = str( Time14(locs['startdate']) - Time14(settings['ASSINT']) )
        settings['DATE_END'] = str( Time14(locs['stopdate'] ) - Time14(settings['ASSINT']) )
        settings['DATA_DIR'] = locs['datadir']+'/archive_exp/exp'+locs['expid']
        settings['FEED_DIR'] = settings['DATA_DIR'] + '/data/ekf/'
        settings['PLOT_DIR'] = settings['DATA_DIR']

        if not 'VISOP_OUTPUT_INTERVAL' in list(settings.keys()) :
            settings['VISOP_OUTPUT_INTERVAL'] = '15'
        if not 'VISOP_ASSIM_INTERVAL' in list(settings.keys()) :
            settings['VISOP_ASSIM_INTERVAL'] = '15'

        #for s in settings :
        #    print s, settings[s]

        if False : # detect settings from directory/file structure
            # experiment name is directory name
            settings['EXP'] = basedir.split('/')[-1]

            # find output/laf*spread files to detect output times
            cwd = os.getcwd()
            os.chdir( basedir+'/output' )
            times = sorted( [x[3:].replace('.spread','') for x in glob.glob('laf*.spread')] )
            print('analysis time  : between ', times[0], ' and ', times[-1])

            assint = Time14(times[1]) - Time14(times[0])
            settings['ASSINT']   = str(assint.daysec())
            print('interval [sec] : ', assint.daysec())

            settings['DATE_INI'] = (Time14(times[0]) - assint).string14()
            settings['DATE_END'] = (Time14(times[-1]) - assint).string14()

        return settings

    #--------------------------------------------------------------------------------
    def parse_logfile( self, logfile, verbose=False, dwd2lmu=False, settings=None, strip_comments=False, overwrite=False, runtimes=True ) :
        """parse KENDA/bacy/run_cycle output to determine experiment settings"""

        # regular expression capturing all "key = value" lines
        # (there are also some "key = = value" lines that will also be recognized)
        p1 = re.compile('^([a-zA-Z0-9_]+)\s*=[\s=]*(.+)\s*')
        p2 = re.compile('.*\$\{([a-zA-Z0-9_]+)\}.*')

        if settings is None :
            settings = {}

        if runtimes :
            settings['runtimes'] = {}
            get_start_time = False
            get_end_time   = False

        with open( logfile, 'r') as f :
            for line in f:

                if runtimes :
                    if get_start_time :
                        # parse something like 'Sat Jul  1 07:28:30 UTC 2017'
                        #print 'trying to parse start time |%s|' % line.strip()
                        day = int(line.split()[2])
                        daytime = line.split()[3]
                        #print 'daytime = ', daytime
                        #swday, smon, mday, daytime, utc, year = line.split(' ')
                        hour, minute, second = list(map( int, daytime.split(':') ))
                        start_time = datetime.datetime( 2000, 1, day, hour, minute, second )
                        get_start_time = False
                        continue
                    elif get_end_time :
                        # parse something like 'Sat Jul  1 07:28:30 UTC 2017'
                        #print 'trying to parse end time |%s|' % line.strip()
                        day = int(line.split()[2])
                        daytime = line.split()[3]
                        #print 'daytime = ', daytime
                        #swday, smon, mday, daytime, utc, year = line.split(' ')
                        hour, minute, second = list(map( int, daytime.split(':') ))
                        end_time = datetime.datetime( 2000, 1, day, hour, minute, second )
                        get_end_time = False

                        if not code in settings['runtimes'] :
                            settings['runtimes'][code] = []
                        dt = end_time - start_time
                        #print '>>> adding run time for ', code, dt.total_seconds() #start_time, end_time,
                        settings['runtimes'][code].append( dt.total_seconds() )
                        continue
                    elif line.startswith('start time') :
                        get_start_time = True
                        code = line.split(' ')[2][:-1]
                        continue
                    elif line.startswith('end time') :
                        get_end_time = True
                        code_ = line.split(' ')[2][:-1]
                        if code_ != code :
                            print("WARNING: inconsistent start/end time structure.")
                        continue

                m1 = p1.match(line)
                if m1 :
                    k = m1.group(1)
                    v = m1.group(2).strip()
                    if not k in list(settings.keys()) or overwrite :
                        if dwd2lmu and ('cosmo_letkf' in v) :
                            pre,  post  = v.split('cosmo_letkf')
                            lpre, lpost = logfile.split('cosmo_letkf')
                            v = lpre + 'cosmo_letkf' + post
                        if strip_comments :
                            v = v.split('#')[0]
                        # expand variables
                        if '$' in v :
                            if verbose : print('>>> in  >>> ', v)
                            m2 = p2.match(v)
                            while not m2 is None :
                                vrep = m2.group(1)
                                if vrep in settings :
                                    #print 'replacing ', vrep
                                    v = v.replace('${'+vrep+'}',settings[vrep])
                                else :
                                    print('WARNING: could not replace variable ', vrep)
                                m2 = p2.match(v)
                            if verbose : print('>>> out >>> ', v)
                        settings[k] = v.strip()
                        if verbose : print('%s = %s' % (k,v))

        if dwd2lmu :
            #topdir  = settings['TOP_DIR']
            basedir = settings['DATA_DIR'].split('cosmo_letkf')[0] + 'cosmo_letkf/'
            settings['TOP_DIR'] = basedir + 'settings/'+settings['EXPID']
            settings['SCRIPT_DIR'] = basedir + 'settings/'+settings['EXPID']+'/scripts/'
            #for k in settings.keys() :
            #    if topdir in settings[k] :
            #        settings[k] = settings[k].replace(topdir,basedir)

        if not 'exp' in list(settings.keys()) :
            if 'EXPID' in list(settings.keys()) :
                settings['exp'] = settings['EXPID']
            else :
                print('warning: guessing experiment id from data directory name')
                settings['exp'] = settings['DATA_DIR'].split('/')[-1]

        if not 'PLOT_DIR' in list(settings.keys()) :
            settings['PLOT_DIR'] = settings['DATA_DIR'].replace('/data','/plots')
            if self.verbose : print('warning: inventing plot directory name %s ...' % settings['PLOT_DIR'])

        if not 'VISOP_OUTPUT_INTERVAL' in list(settings.keys()) :
            settings['VISOP_OUTPUT_INTERVAL'] = '15'

        # multiscale evaluation settings
        if not 'VISOP_EVAL_AREA' in settings :
            settings['VISOP_EVAL_AREA'] = 'LATLON:74,77,329,204_LATLONUNITS:PX'
        if not 'VISOP_EVAL_SCALES' in settings :
            settings['VISOP_EVAL_SCALES'] = '1,2,4,8,16,32,64'

        # VISOP default daily period
        if not       'VISOP_START_HOUR'   in settings : # VISOP will not be run before VISOP_START_HOUR : VISOP_START_MINUTE UTC
            settings['VISOP_START_HOUR'  ]='5'
        if not       'VISOP_START_MINUTE' in settings :
            settings['VISOP_START_MINUTE']='15'
        if not       'VISOP_STOP_HOUR'    in settings : # VISOP will not be run after  VISOP_STOP_HOUR  : VISOP_STOP_MINUTE  UTC
            settings['VISOP_STOP_HOUR'   ]='18'
        if not       'VISOP_STOP_MINUTE'  in settings :
            settings['VISOP_STOP_MINUTE' ]='0'

        # observations in this area are used e.g. in compute_error_evolution
        if not 'OBS_EVAL_AREA' in settings :
            settings['OBS_EVAL_AREA'] = 'LATLON:47.7,3.5,56.0,17.5_LATLONUNITS:DEG'

        # output interval for lff*_sfc [h]
        if not 'SFC_OUTPUT_INTERVAL' in settings :
            settings['SFC_OUTPUT_INTERVAL'] = '1.0'

        return settings

    #--------------------------------------------------------------------------------
    def description( self, template='_VIS_' ) :

        s = template.replace('_REF_','$E: A$A $I').replace('_VIS_','$E: A$A $I V${VISOP_ASSIM_INTERVAL} e${VISOP_ERROR} L$L S$S T$T D$D')

        # special cases
        if '$E' in s : s = s.replace('$E',self.expid)
        if '$A' in s : s = s.replace('$A','%d'%(int(self.settings['ASSINT'])/60))
        if '$L' in s : s = s.replace('$L','%03.0f'%float(self.settings['VISOP_HLOC']))
        if '$S' in s : s = s.replace('$S', '%d'%(int(self.settings['VISOP_SUPEROBB'].split(',')[0])*3) )
        if '$T' in s :
            if self.settings['VISOP_THINNING'].startswith('"latlon') :
                s = s.replace('$T', 'latlons' )
            else :
                s = s.replace('$T', self.settings['VISOP_THINNING'] )
        if '$D' in s :
            if self.settings['VISOP_THINNING'].startswith('"latlon') :
                l_obs = 1e6
            else :
                l_obs = int(self.settings['VISOP_SUPEROBB'].split(',')[0])*3 * float(self.settings['VISOP_THINNING'].split(',')[0])
            d = (2*pi*float(self.settings['VISOP_HLOC'])**2) * (60.0/int(self.settings['VISOP_ASSIM_INTERVAL'])) / (l_obs*l_obs)
            s = s.replace('$D','%.1f'%d)
        if '$I' in s :
            import re
            res = {'RAD':r"\s*&REPORT\s*type='RAD'\s*use='([a-zA-Z]+)'",
                   'mf':r'\s*mf\s*=\s*([0-9a-zA-Z%.]+)',
                   'rtpp':r'\s*apply_rtpp\s*=\s*(\S)',
                   'rtps':r'\s*apply_rtps\s*=\s*(\S)',
                   'adap_rho':r'\s*adap_rho\s*=\s*(\S)',
                   'adap_loc':r'\s*adap_loc\s*=\s*(\S)',
                   'lh':r'\s*lh\s*=\s*([\d.]+)'}
            rec = {}
            for r in res :
                rec[r] = re.compile(res[r])

            lst = {}
            with open( self.settings['TOP_DIR']+'/templates/namelist_letkf_template_addcovdwd', 'r' ) as f :
                nml = f.readlines()
            for line in nml :
                for r in res :
                    m = rec[r].match(line)
                    if m : lst[r] = m.group(1)

            if Time14(self.settings['RHO_DATE']) < Time14(self.veri_times[-1]) :
                lst['adap_rho'] = 'T'
            else :
                lst['adap_rho'] = 'F'

            if 'EXPID_START' in self.settings :
                expid_start = '%9s' % self.settings['EXPID_START']
            else :
                expid_start = '...vis...'
            infloc = 'addInf%s adapInf%s rtpp%s rtps%s %s ' % (lst['mf'],lst['adap_rho'],lst['rtpp'],lst['rtps'],
                                                                  '***REF***' if lst['RAD'].lower() == 'passive' else expid_start )
            if lst['adap_loc'] == 'F' :
                infloc += 'lh%03d'%int(float(lst['lh']))
            else :
                infloc += 'lhAda'
            s = s.replace('$I', infloc )

        # variables from settings or lfcst_settings
        for p in self.settings :
            si = "${%s}" % p
            if si in s :
                s = s.replace(si,self.settings[p])
        if self.lfcst :
            for p in self.lfcst_settings :
                si = "${lfcst_%s}" % p
                if si in s :
                    s = s.replace(si,self.lfcst_settings[p])
        return s

    #--------------------------------------------------------------------------------
    def list_files( self ) :

        print()
        print('>>> list of available files for cycles ', self.settings['exp'])
        for t in self.all_times :
            print()
            print('    time %s  -------------------------------------------------' % t)

            if self.exp_type != 'sofia' :
                for channel in ['VIS006','VIS008','NIR016'] :
                    if os.path.exists(self.visop_filename( t, memidx=1, channel=channel )) :
                        print('    VISOP    : ', channel)
                    #else :
                    #    print 'does not exist : ', self.visop_filename( t, memidx=1, channel=channel )

            print('    feedback : ', end=' ')
            for ot in list(tables['obstypes'].values()) :
                if os.path.exists( self.ekf_filename( t, ot ) ) :
                    print(ot, ' ', end=' ')
                #else :
                #    print 'does not exist : ', self.ekf_filename( t, ot )
            print()

            print('    data     : ', end=' ')
            for pref in ['laf','lff'] :
                print(pref + '[', end=' ')
                for suff in ['det','mean','spread'] :
                    if os.path.exists( self.cosmo_filename(t,prefix=pref,suffix=suff) ) : print(suff, ' ', end=' ')
                print('] ', end=' ')
            print()
        print()

        if not self.lfcst_settings is None :
            print()
            print('>>> list of available files for longer forecasts : ')
            for ts in self.lfcst_start_times :
                print(ts, '.'*50)
                print('SFC    ', end=' ')
                for t in self.lfcst_sfc_output_times_min :
                    n = 0
                    for m in range(self.n_ens) :
                        fname =  self.cosmo_filename( t, lfcst_time=ts, prefix='lfff', suffix='sfc', member=m+1 )
                        if os.path.exists(fname) :
                            n += 1
                    if n > 0 : print(' %s[%s]' % (t,n), end=' ')
                print()
                print('VISOP ', end=' ')
                for t in self.lfcst_visop_output_times_min :
                    n = 0
                    for m in range(self.n_ens) :
                        fname = self.visop_filename( ts, fcsttime=t, memidx=m+1, channel='VIS006', lfcst=True )
                        if os.path.exists(fname) :
                            n += 1
                    if n > 0 : print(' %s[%s]' % (t,n), end=' ')
                print()


    #-------------------------------------------------------------------------------------------------------------------
    def cosmo_filename( self, start_time, output_time=None, lfcst=False, prefix='lff', suffix='auto', member=None ) :
        """Return absolute file name of the specified COSMO output file

           start_time = fcst start time (e.g. '20160605100000')
           output_time = output time (relative to start_time) [min]
           member : member index (default=0=deterministic member)
        """

        if member is None :
            member_ = 0
        else :
            member_ = member

        fname = ''

        if self.exp_type == 'sofia' : # SOFIA ..........................................................................

            dir = self.settings['DATA_DIR']+'/output/'
            tstr = str(Time14(start_time))

        else : # BACY/COSMO ............................................................................................

            if not lfcst : # CYCLING . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

                if output_time is None :
                    if prefix.startswith('lff') :
                        tdir = str( (Time14(start_time) + Time14(self.settings['ASSINT'])).divisible( Time14(self.settings['OBSINT'])) )
                    elif prefix == 'laf' :
                        tdir = str( Time14(start_time).divisible( Time14(self.settings['OBSINT'])) )
                    else :
                        raise ValueError('Unknown cosmo file name prefix '+prefix)
                    dir = self.settings['DATA_DIR']+'/'+tdir+'/'
                    tstr = str(Time14(start_time))

                else :
                    if suffix == 'sfc' :
                        # e.g. [...]/feedback/0605.104/visop/20160605150000/lff_sfc_20160605150000_0100.001
                        otstr = str(Time14(output_time*60))[2:-2]
                        fname = "%s/visop/%s/lff_sfc_%s_%s.%03d" % (self.lfcst_settings['FEED_DIR'], start_time, start_time, otstr, member_)
                    else :
                        dir = "%s/%s/ens%03d/" % (self.settings['DATA_DIR'],start_time,member_)
                        tstr = str(Time14(output_time*60))

            else : # LONG FORECASTS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

                if member == 0 :
                    dir = "%s/%s/det/" % (self.lfcst_settings['DATA_DIR'],start_time)
                else :
                    dir = "%s/%s/ens%03d/" % (self.lfcst_settings['DATA_DIR'],start_time,member_)
                tstr = str(Time14(output_time*60))
                fname = dir + prefix + tstr + (('_'+suffix) if not suffix == 'auto' else '')

        #if suffix != '' and not suffix.startswith('.') :
        #    suffix_ = '.'+suffix
        #else :

        if fname == '' :

            if suffix == 'auto' :
                if member_ == 0 :
                    suffix_ = '.det'
                else :
                    suffix_ = ".%03d" % member_
            elif suffix == 'spread' :
                suffix_ = '.spread'
            elif suffix == 'mean' :
                suffix_ = '.mean'
            elif suffix == 'det' :
                suffix_ = '.det'
            elif suffix == 'sfc' :
                suffix_ = '_sfc'
            else :
                suffix_ = suffix

            fname = dir + prefix + tstr + suffix_

        #print('COSMO_FILENAME: ', start_time, output_time, lfcst, prefix, suffix, member, ' --> ', fname )

        return fname

    #--------------------------------------------------------------------------------
    def get_cosmo( self, start_time, lfcst=False, output_time=None, prefix='lff', suffix='', member=None, preload=[], cache=True ) :
        """read (and by default cache) COSMO output file"""

        from kendapy.cosmo_state import CosmoState

        if member is None :
            member_ = 0
        else :
            member_ = member

        cskey = prefix+'_'+str(start_time)+'_'+suffix+'_mem'+str(member_)
        if not output_time is None :
            cskey += '_t'+str(output_time)

        if not cskey in list(self.cosmo.keys()) :

            if not lfcst :
                # only the lff*det files contain all constants like RLAT, RLON...
                # --> such a file should be provided to CosmoState via the constfile parameter
                if prefix == 'lff' :
                    consttime = start_time
                else :
                    # laf files are defined at different times than lff files...
                    consttime = str( Time14(start_time)-Time14(self.settings['ASSINT']) )
                constfile = self.cosmo_filename( consttime, prefix='lff', suffix='det' )
            else :
                constfile = None

            #print( 'get_cosmo: ', start_time, output_time, lfcst, prefix, suffix, member_ )
            fname = self.cosmo_filename( start_time, lfcst=lfcst, output_time=output_time, prefix=prefix, suffix=suffix, member=member )
            cs = CosmoState( fname, preload=preload, constfile=constfile, singlestep=True )
            if cache : self.cosmo[cskey] = cs

        else :
            cs = self.cosmo[cskey]
        return cs

    #--------------------------------------------------------------------------------
    def visop_filename( self, starttime, fcsttime=-1, memidx=-1, channel='VIS006', lfcst=False ) :
        """Return absolute file name of the VISOP reflectance file for the specified
           fcst start time + <fcsttime> minutes (-1=>use last output), channel and
           member index (-1=obs,0=det)"""

        #if fcsttime == -1 :
        #    fcsttime_ = int(self.settings['ASSINT'])/60
        #else :
        #    fcsttime_ = fcsttime

        if lfcst :
            if fcsttime  == -1 :
                fcsttime_ = self.lfcst_visop_output_times_min[-1]
            else :
                fcsttime_ = fcsttime
                if not fcsttime_ in self.lfcst_visop_output_times_min :
                    raise ValueError("Invalid lfcst_visop_output_time ", fcsttime )
        else :
            fcsttimes = self.visop_fcsttimes()
            if fcsttime == -1 :
                fcsttime_ = fcsttimes[-1]
            else :
                fcsttime_ = fcsttime
                if not fcsttime in fcsttimes :
                    raise ValueError("Invalid visop fcsttime ", fcsttime )

        if fcsttime_ >= 60 :
            fcsttime_ = 100*(fcsttime_//60) + fcsttime_%60

        if memidx == -1 :
            pref = 'refl_seviri_'
            postf = '_'+channel
        else :
            pref = 'refl_visop_'
            postf = "_%s.%03d" % (channel,memidx)

        if lfcst :
            fname = self.lfcst_settings['DATA_DIR']+'/visop/'+starttime+'/'+pref+starttime+("_%04d"%fcsttime_)+postf+'.npy'
        else :
            fname = self.settings['FEED_DIR']+'/visop/'+starttime+'/'+pref+starttime+("_%04d"%fcsttime_)+postf+'.npy'

        return fname

    #----------------------------
    def visop_fcsttimes( self ) :
        """Return list of times (in minutes, relative to fcst start time)
           for which visop output is available"""

        dt_visop = int(self.settings['VISOP_OUTPUT_INTERVAL'])
        assint   = int(self.settings['ASSINT'])//60

        fcsttimes = arange(0,assint+1,dt_visop)

        return fcsttimes

    #--------------------------------------------------------------------------------
    def get_visop( self, starttime, fcsttime=-1, channel='VIS006', lfcst=False, cache=True, **kw ) :
        """Returns visop ensemble object"""

        vskey = '%s_%04d_%s' % (starttime,fcsttime,channel)
        if lfcst : vskey += '_lfcst'

        if not vskey in list(self.visop.keys()) :

            import kendapy.visop_ensemble
            vs = kendapy.visop_ensemble.VisopEnsemble( self, starttime, fcsttime=fcsttime, channel=channel, lfcst=lfcst,
                                                       verbose=self.verbose, **kw )
            if cache :
                self.visop[vskey] = vs

        else :
            vs = self.visop[vskey]
        return vs

    #--------------------------------------------------------------------------------
    def ekf_filename( self, time, obs_type ) :
        """Return absolute file name of the specified ekf file"""

        if self.exp_type == 'sofia' :
            return self.settings['FEED_DIR']+'/'+time+'/'+'ekf'+obs_type+'.nc'

        elif self.exp_type == 'bacy1' :
            return self.bacy_exp.get_filename( 'ekf', time, obs_type=obs_type )
            #print('+++++ ', self.bacy_exp.filetree['cycles'][time].keys() )
            #return self.bacy_exp.filetree['cycles'][time]['ekf'][obs_type]

        else :
            tdir = str( Time14(time).divisible( Time14(self.settings['OBSINT'])) )
            dir  = self.settings['FEED_DIR']+'/'+tdir+'/'
            return dir+'ekf'+obs_type+'_'+time+'.nc'

    #--------------------------------------------------------------------------------
    def get_ekf( self, time, obs_type, state_filter='active', area_filter='auto', varname_filter=None, cache=True,
                 verbose=False, **filter ) :
        """read and cache ekf file"""

        if (not area_filter is None) and (area_filter == 'auto') :
            if 'OBS_EVAL_AREA' in self.settings :
                area_filter_ = self.settings['OBS_EVAL_AREA']
            else :
                area_filter_ = None
        else :
            area_filter_ = area_filter

        ekfkey = time+'_O'+obs_type
        if not state_filter   is None : ekfkey += '_S'+state_filter
        if not area_filter    is None : ekfkey += '_A'+area_filter
        if not varname_filter is None : ekfkey += '_V'+varname_filter

        if (not ekfkey in list(self.ekfs.keys())) or not cache or len(filter) > 0 :
            ekf = Ekf( self.ekf_filename( time, obs_type ), verbose=verbose,
                       state_filter=state_filter, area_filter=area_filter_, varname_filter=varname_filter, **filter )
            if cache :
                self.ekfs[ekfkey] = ekf
        else :
            ekf = self.ekfs[ekfkey]

        return ekf

    #--------------------------------------------------------------------------------
    def fof_filename( self, start_time, memidx=1, lfcst=False ) :

        if lfcst :
            tdir = str( Time14(start_time).divisible( Time14(self.settings['OBSINT'])) )
            if not self.lfcst :
                raise ValueError('Long forecasts not available')
            dir  = self.lfcst_settings['DATA_DIR']+'/'+tdir
        else :
            # FIXME e.g. fof*110000 is in the *120000 folder, not in *090000!
            #            fof*100000 is in *090000
            #            fof*120000 is in *120000
            #                              *************
            tdir = str( (Time14(start_time)+Time14(3600)).divisible( Time14(self.settings['OBSINT'])) )
            dir  = self.settings['DATA_DIR']+'/'+tdir
        return '%s/fof_%s_ens%03d.nc' % (dir,start_time,memidx)

    #--------------------------------------------------------------------------------
    def get_fof( self, start_time, memidx=1, lfcst=False, verbose=False, area_filter='auto', **filter ) :
        """read and cache fof file"""

        if (not area_filter is None) and (area_filter == 'auto') :
            if 'OBS_EVAL_AREA' in self.settings :
                area_filter_ = self.settings['OBS_EVAL_AREA']
            else :
                area_filter_ = None
        else :
            area_filter_ = area_filter

        # FIXME pressure_filter=auto with lower pressure bound from LETKF namelist ?
        #pressure_filter='300,1100',

        return Ekf( self.fof_filename( start_time, memidx=memidx, lfcst=lfcst ), verbose=verbose,
                    area_filter=area_filter_, **filter )


    #--------------------------------------------------------------------------------
    def get_fofens( self, start_time, lfcst=False, ekf=None, area_filter='auto', **filter ) :
        """
        Read FOF file ensemble
        
        :param start_time:    Forecast start time
        :param lfcst:         If True, use FOFs from long forecasts 
        :param ekf:           Ekf object. If specified, observations not contained in ekf wil be ignored
        :param area_filter:   Ekf area filter, defaults to self.settings['OBS_EVAL_AREA']
        :param filter:        Ekf filter keywords
        :return:              FofEns object
        """

        import kendapy.fof_ensemble
        fofs = [ self.get_fof( start_time, memidx=m, lfcst=lfcst, area_filter=area_filter, **filter )
                 for m in range(1,self.n_ens+1) ]
        return kendapy.fof_ensemble.FofEns( fofs )

    #--------------------------------------------------------------------------------
    def get_fofens_statistics( self, time_range='all', times=None, obs_types=None, variables=None,
                               area_filter='auto', state_filter='active', lfcst=False, recompute=False ) :
        import hashlib

        if obs_types is None :
            obs_types = list(tables['obstypes'].values())
        if variables is None :
            variables = list(tables['varnames'].values())
        if times is None :
            if lfcst :
                times = self.lfcst_start_times
            else :
                times = self.fcst_start_times

        # cache results
        cachefdir = self.settings['PLOT_DIR']+'/cache/fofens/'
        if not os.path.exists(cachefdir) :
            print('generating cache file directory %s ...' % cachefdir)
            os.makedirs(cachefdir)

        cachefname = '%s/FOFENS_%s-%s_%s_%s_%s%s' % ( cachefdir, times[0], times[-1], time_range, state_filter,
                                                      area_filter, '_lfcst' if lfcst else '' )
        otvn = 'OT_'+('-'.join(obs_types)) + '-VN_' + ('-'.join(variables))
        hash_object = hashlib.md5(otvn.encode())
        otvn_hash = hash_object.hexdigest()
        #print 'USING HASH ', otvn_hash
        cachefname += otvn_hash

        if os.path.exists(cachefname) and not recompute :
            print('loading %s ...' % cachefname)
            with open( cachefname, 'rb') as f :
                fofstat = pickle.load(f)
        else :
            if recompute :
                print('ignoring potentially existing cache file %s ...' % (cachefname))
            else :
                print('cache file %s does not yet exist...' % (cachefname))

            fofstat = {}
            for start_time in times :
                if self.verbose : print('%s : reading fof ensemble...' % start_time)
                fofens = self.get_fofens( start_time, state_filter=state_filter, area_filter=area_filter, lfcst=lfcst )
                for obs_type in obs_types :
                    if obs_type == 'RADAR' :
                        print('WARNING: IGNORING RADAR! (FIXME)')
                        continue
                    for vname in variables :
                        stat =  fofens.statistics( obs_type, vname, time_range )
                        if stat['n_obs'] > 0 :
                            if self.verbose : print('=== ', obs_type, vname, stat)
                            if not obs_type in list(fofstat.keys())           : fofstat[obs_type] = {}
                            if not vname    in list(fofstat[obs_type].keys()) : fofstat[obs_type][vname] = {}
                            fofstat[obs_type][vname][start_time] = stat

            print('saving cache file to %s ...' % (cachefname))
            with open( cachefname, 'wb') as f :
                pickle.dump( fofstat, f, pickle.HIGHEST_PROTOCOL )

        return fofstat

    #--------------------------------------------------------------------------------
    def compute_error_evolution( self, times=[], obs_types=[], variables=[], verbose=False,
                                 area_filter='auto', state_filter='active', **filter ) :

        if len(obs_types) == 0 :
            obs_types = list(tables['obstypes'].values())
        if len(variables) == 0 :
            variables = list(tables['varnames'].values())
        if len(times) == 0 :
            times = self.veri_times

        if (not area_filter is None) and (area_filter == 'auto') :
            if 'OBS_EVAL_AREA' in self.settings :
                area_filter_ = self.settings['OBS_EVAL_AREA']
            else :
                area_filter_ = None
        else :
            area_filter_ = area_filter
        state_filter_ = state_filter

        if verbose : print('>>> compute_error_evolution : filters = ', area_filter_, state_filter_)

        eevo = {}
        for obs_type in obs_types : 
            if verbose : print('>>> error evolution for observation type %s...' % obs_type, end=' ')

            for t in times :
                if not os.path.exists( self.ekf_filename( t, obs_type ) ) : continue

                ekf = self.get_ekf( t, obs_type, state_filter=state_filter_, area_filter=area_filter_, **filter )
                #print 'ekf.filtername = ', ekf.filtername
                for vname in ekf.varnames :
                    if not vname    in  variables            : continue
                    if not obs_type in list(eevo.keys())           : eevo[obs_type] = {}
                    if not vname    in list(eevo[obs_type].keys()) : eevo[obs_type][vname] = {}
                    eevo[obs_type][vname][t] = ekf.statistics( varname_filter=vname )
                    if verbose : print(eevo[obs_type][vname][t]['n_obs'], end=' ')
            print()

        if verbose :
            for ot in eevo :
                for vname in eevo[ot] :
                    times = list(eevo[ot][vname].keys())
                    nobs = array([       eevo[ot][vname][t]['n_obs']
                                         for t in list(eevo[ot][vname].keys()) ]).sum()
                    rmse = sqrt( array([ eevo[ot][vname][t]['n_obs']*eevo[ot][vname][t]['fgmean']['rmse']**2
                                         for t in list(eevo[ot][vname].keys()) ]).sum() / nobs )
                    bias = array([       eevo[ot][vname][t]['n_obs']*eevo[ot][vname][t]['fgmean']['bias']
                                         for t in list(eevo[ot][vname].keys()) ]).sum() / nobs
                    print('%10s / %10s : nobs = %6d, mean fg rmse = %f, mean fg bias = %f' % (ot, vname, nobs, rmse, bias))

        return eevo

    #--------------------------------------------------------------------------------
    def compute_cumulative_statistics( self, times=[], obs_types=[], variables=[],
                                       depmin=-5,   depmax=5,    ndepbins=100,
                                       valmin=None, valmax=None, nvalbins=100, **filter_kw ) :

        if len(obs_types) == 0 :
            obs_types = list(tables['obstypes'].values())
        if len(variables) == 0 :
            variables = list(tables['varnames'].values())
        if len(times) == 0 :
            times = self.veri_times

        cstat = {}
        for obs_type in obs_types :
            print('>>> cumulative statistics for observation type %s...' % obs_type)

            for t in times :
                if not os.path.exists( self.ekf_filename( t, obs_type ) ) : continue

                ekf = self.get_ekf( t, obs_type, **filter_kw )
                #print '#obs : ', ekf.obs().size

                for vname in ekf.varnames :
                    if not vname in variables : continue

                    if not obs_type in list(cstat.keys())           : cstat[obs_type] = {}
                    if not vname    in list(cstat[obs_type].keys()) : cstat[obs_type][vname] = {}

                    # departure statistics
                    for veri in ['anaens','fgens'] :
                        dep_bins = linspace(depmin,depmax,ndepbins)
                        dep = ekf.departures( veri, normalize=True, varname_filter=vname )
                        dep_hist, dep_hist_edges = histogram( dep, bins=dep_bins )

                        if not veri+'_dep_hist' in cstat[obs_type][vname] :
                            cstat[obs_type][vname][veri+'_dep_hist'] = dep_hist
                            cstat[obs_type][vname][veri+'_dep_hist_edges'] = dep_hist_edges
                        else :
                            cstat[obs_type][vname][veri+'_dep_hist'] += dep_hist

                    # value statistics
                    for veri in  ['anaens','fgens','obs'] :

                        if (valmin is None) or (valmax is None) :
                            obs = ekf.obs(varname_filter=vname)
                            obsmin, obsmax = obs.min(), obs.max()
                            dobs = obsmax - obsmin
                            print('min/max observation values : ', obsmin, obsmax)
                        if valmin is None :
                            valmin = obsmin - 0.05*dobs
                            print('setting minimum value for value histograms to', valmin)
                        if valmax is None :
                            valmax = obsmax + 0.05*dobs
                            print('setting maximum value for value histograms to', valmax)

                        val_bins = linspace(valmin,valmax,nvalbins)
                        if veri == 'obs' :
                            val_hist, val_hist_edges = histogram( ekf.obs(varname_filter=vname), bins=val_bins )
                        else :
                            val_hist, val_hist_edges = histogram( ekf.veri(veri,varname_filter=vname), bins=val_bins )
                            val_hist /= self.n_ens
                            print('dividing by ensemble size ', self.n_ens)

                        if not veri+'_val_hist' in cstat[obs_type][vname] :
                            cstat[obs_type][vname][veri+'_val_hist'] = val_hist
                            cstat[obs_type][vname][veri+'_val_hist_edges'] = val_hist_edges
                        else :
                            cstat[obs_type][vname][veri+'_val_hist'] += val_hist

        return cstat

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Get information about KENDA experiment')
    parser.add_argument(       '--dwd2lmu',     dest='dwd2lmu',     help='convert settings',  action='store_true' )
    parser.add_argument( '-v', '--verbose',     dest='verbose',     help='be more verbose',   action='store_true' )    
    parser.add_argument( '-S', '--sofia',       dest='sofia',       help='assume results were generated by SOFIA, not BACY', action='store_true' )    
    parser.add_argument( '-E', '--evolution',   dest='evolution',   help='generate error evolution output', action='store_true' )
    parser.add_argument( '-F', '--fofstat',     dest='fofstat',     help='print fof file statsitics', action='store_true' )
    parser.add_argument( '-T', '--time-range',  dest='time_range',  help='time range for fof file statsitics', default='0-60' )
    parser.add_argument(       '--recompute',   dest='recompute',   help='do not read cache files', action='store_true' )
    parser.add_argument( '-l', '--list',        dest='list',        help='list main settings for each experiment', action='store_true' )
    parser.add_argument(       '--debug',       dest='debug',       help='debug',  action='store_true' )
    parser.add_argument( 'logfile', metavar='logfile', help='log file name (old bacy) or base directory (new bacy)', nargs='*' )
    args = parser.parse_args()

    if args.debug :
        for logfile in args.logfile :
            print( 'experiment : ', logfile )
            xp = Experiment( logfile )
            print( 'experiment type : ', xp.exp_type )
            for c in sorted(xp.bacy_exp.config) :
                print('{:25s}{}'.format(c,xp.bacy_exp.config[c]))
            print( 'forecast start times 1 : ', xp.get_fcst_start_times() )
            print( 'forecast start times 2 : ', xp.fcst_start_times )
            print( 'veri           times   : ', xp.veri_times )
        sys.exit(0)

    # process all log files

    for logfile in args.logfile :
        if not args.list :
            print("processing %s ..." % logfile)
        xp = Experiment( logfile, sofia=args.sofia, dwd2lmu=args.dwd2lmu, verbose=args.verbose )

        if args.list :
            print(xp.description())
            continue

        if not args.fofstat or args.evolution :
            xp.list_files()

        if args.evolution :
            eevo = xp.compute_error_evolution( verbose=True )

        if 'runtimes' in xp.settings :
            print()
            print('>>> min/mean/max run times [sec] : ')
            for code in list(xp.settings['runtimes'].keys()) :
                rts = array(xp.settings['runtimes'][code])
                print("-- %10s : %7.0f  /  %7.0f  /  %7.0f" % (code, rts.min(), rts.mean(), rts.max() ))

        if args.fofstat :
            fofstat = xp.get_fofens_statistics( time_range=args.time_range, lfcst=True, recompute=args.recompute )
            for obs_type in fofstat :
                for vname in fofstat[obs_type] :
                    print('%10s / %10s :: ' % (obs_type, vname), end=' ')
                    for t in fofstat[obs_type][vname] :
                        print('%15s : %5d' % (t,  fofstat[obs_type][vname][t]['n_obs']), end=' ')
                    print()

        if not xp.exp_type == 'sofia' :
            print( 'short description', xp.description('${EXPID}:A${ASSINT}/V${VISOP_ASSIM_INTERVAL}/L${VISOP_HLOC}/S${VISOP_SUPEROBB}'))
