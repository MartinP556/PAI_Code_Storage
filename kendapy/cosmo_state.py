#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . C O S M O _ S T A T E
#  class representing COSMO model state
#
#  2016.10 L.Scheck 

from __future__ import absolute_import, division, print_function
from numpy import *
import os, sys, getpass, subprocess, argparse, time, re, gc, pickle
from kendapy.cosmo_grid import rot_to_nonrot_grid, nonrot_to_rot_grid, cosmo_grid

# import grib_api or eccodes in a way that works for Python2 and Python3
using_eccodes = False
try:
    import eccodes as gribapi
    using_eccodes = True
except ImportError:
    try:
        import gribapi
    except ImportError:
        try:
            import grib_api as gribapi
        except ImportError:
            print('Found neither gribapi (versions <= 1.15) nor grib_api (versions >= 1.16)')
            sys.exit(-1)

if using_eccodes :
    grib_get_api_version        = gribapi.codes_get_api_version
    grib_new_from_file          = gribapi.codes_grib_new_from_file
    grib_keys_iterator_new      = gribapi.codes_keys_iterator_new
    grib_keys_iterator_next     = gribapi.codes_keys_iterator_next
    grib_keys_iterator_get_name = gribapi.codes_keys_iterator_get_name
    grib_get_string             = gribapi.codes_get_string
    grib_keys_iterator_delete   = gribapi.codes_keys_iterator_delete
    grib_get                    = gribapi.codes_get
    grib_release                = gribapi.codes_release
    grib_get_long               = gribapi.codes_get_long
    grib_get_values             = gribapi.codes_get_values
    GribInternalError           = gribapi.GribInternalError
else :
    grib_get_api_version        = gribapi.grib_get_api_version
    grib_new_from_file          = gribapi.grib_new_from_file
    grib_keys_iterator_new      = gribapi.grib_keys_iterator_new
    grib_keys_iterator_next     = gribapi.grib_keys_iterator_next
    grib_keys_iterator_get_name = gribapi.grib_keys_iterator_get_name
    grib_get_string             = gribapi.grib_get_string
    grib_keys_iterator_delete   = gribapi.grib_keys_iterator_delete
    grib_get                    = gribapi.grib_get
    grib_release                = gribapi.grib_release
    grib_get_long               = gribapi.grib_get_long
    grib_get_values             = gribapi.grib_get_values
    GribInternalError           = gribapi.CodesInternalError


class CaseInsensitiveDict(dict):
    """
    Case insensitive Dictionary from: http://stackoverflow.com/questions/2082152/case-insensitive-dictionary
    """
    @classmethod
    def _k(cls, key):
        return key.upper() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)

    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))

    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))

    def has_key(self, key):
        return self.__class__._k(key) in super(CaseInsensitiveDict, self)

    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)

    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)


class CosmoState(object):
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
                    'L':['isothermZero','cloudTop','cloudBase','nominalTop','lakeBottom','thermocline','mixedLayer','entireLake','atmML'],
                    'U':['unknown'],
                    'X':['TOA','meanLayer','entireAtmosphere','isobaricLayer'] }

    # time-related keys causing internal grib_api errors for some DWD grib files
    step_keys = ['stepRange','step','stepType','timeRangeIndicator','startStep','endStep','validityDate','validityTime']

    # variables that can be computed or at least approximated
    computable_quantities = ['PHL','PML','HML','RLAT','RLON','DEN','dz','TQV','TQC','TQI','TQC_DIA','TQI_DIA','RELHUM','P','AREA']

    #--------------------------------------------------------------------------------
    def __init__( self, fname, preload=None, verbose=False, constfile='', singlestep=False, translation=None,
                  default_leveltype=None, ignore_leveltype=None ) :
        """
        Initialize CosmoState object

        :param fname:      GRIB1 or GRIB2 file name
        :param preload:    Variables to be loaded from the file immediately
        :param verbose:    Be more verbose
        :param constfile:  Alternative grib file from which variables will be read that are not found in <fname>
        :param singlestep: If set to True, will prevent grib_api from accessing time step information,
                           which will suppress grib_api internal errors for DWD grib files.
                           Must not be set if more than one time step is to be accessed.
        :param default_leveltype: assume user wants variables with this leveltype (unless level type is explicitly specified)
        :param ignore_leveltype: ignore variables with this level type
        """

        if verbose :
            print(('[CosmoState] Using '+('eccodes' if using_eccodes else 'grib_api')+' version ', grib_get_api_version()))

        # check presents of files
        if not os.path.isfile(fname):
            raise IOError("file %s not found!" % fname)
        if constfile is not None and constfile != '' and not os.path.isfile(constfile):
            raise IOError("file %s not found!" % constfile)

        self.filename = fname
        if constfile is None :
            self.constfile = ''
        else :
            self.constfile = constfile

        self.quan = CaseInsensitiveDict()
        self.meta = CaseInsensitiveDict()

        # directories for the grib file and the alternative grib file
        self.vardir = None
        self.constfiledir = None

        self.singlestep = singlestep
        # if True, do not attempt to read the self.step_keys, which would cause
        # errors for some DWD grib files

        # generate list with all level types
        self.level_type_list = []
        for l in list(self.level_types.keys()) :
            self.level_type_list += self.level_types[l]
        self.default_leveltype  = default_leveltype
        self.ignore_leveltype = ignore_leveltype

        # translate variable names, if requested
        self.translation = translation # keys : grib_api/eccodes names, values : names visible to cosmo_state user
        if not self.translation is None :
            if self.translation == 'pre2019' :
                print( 'using pre2019 variable names...' )
                self.translation = { 'CLWMR':'QC', 'PRES':'P', 'Q':'QV', 'RWMR':'QR', 'SNMR':'QS', 'WZ':'W', 'CCL':'CLC', 'DEN':'RHO' }
        if not self.translation is None :
            self.translation_grib_vnames = self.translation.keys() # grib_api/eccodes variable names
            self.translation_user_vnames = self.translation.values() # user variable name
            self.translation_inv = { value : key for key, value in self.translation.items() }
        else :
            self.translation_grib_vnames = []
            self.translation_user_vnames = []
            self.translation_inv = {}

        if (not (preload is None)) and len(preload) > 0 :
            self.load_variables( preload, verbose=verbose, ignore_missing=True )

    #--------------------------------------------------------------------------------
    def translate_grib2user( self, n ) :
        if n in self.translation_grib_vnames :
            return self.translation[n]
        else :
            return n

    #--------------------------------------------------------------------------------
    def __getitem__( self, varname ):
        """
        The [] operator returns variables stored in self.quan. The variables are read from the grib file or the
        alternative grib file or are generated using the compute() method.
        """

        verbose = False

        if varname not in self.quan:
            if verbose: print('>>> not loaded: ', varname)
            loaded = False

            if (not self.translation is None) and (varname in self.translation) :
                if verbose: print('>>> trying to load translated variable failed: ', self.translation[varname])
                try :
                    self.__getitem__( self.translation[varname] )
                    varname_ = self.translation[varname]
                    loaded = True
                except ValueError :
                    if verbose: print('>>> loading failed: ', varname)
                
            if not loaded :
                varname_ = varname

                if self.could_be_loadable(varname) :
                    if verbose : print('>>> it could be possible to load: ', varname)
                    try :
                        self.load_variables( [varname] )
                        loaded = True
                        if verbose: print('>>> loaded: ', varname)
                    except ValueError :
                        if verbose: print('>>> loading failed: ', varname)

                if not loaded :
                    if verbose: print('>>> not yet loaded: ', varname)
                    if varname in self.computable_quantities :
                        if verbose: print('>>> will be computed: ', varname)
                        self.compute( varname, store=True )
                    else :
                        raise ValueError("Could neither be loaded not computed: ", varname)
        else :
            varname_ = varname

        return self.quan[varname_]

    #--------------------------------------------------------------------------------
    def could_be_loadable( self, varname ) :
        """
        Check if there is a chance that variable <varname> could be loaded from the grib file
        or the alternative grid file
        """
        if (self.vardir is None) or (self.constfile != '' and self.constfiledir is None) :
            # we have no information on the contents of the grib file (and the alternative grib file)
            # --> the variable may be available...
            retval = True
        else :
            # we have information about the file contents and can check whether the short_name specified
            # in <varname> is available.

            # determine the short_name from the full variable name (which may contain modifiers like ":M")
            re_shortname = re.compile('^([a-zA-Z0-9_\.]+)')
            m = re_shortname.match(varname)
            if m :
                shn = m.group(1)
                retval = False
                if shn in self.vardir :
                    retval = True
                if (not (self.constfiledir is None)) and (shn in self.constfiledir) :
                    retval = True
            else :
                raise ValueError('Cannot parse variable name '+varname)

        return retval

    #--------------------------------------------------------------------------------
    def keys(self) :
        return list(self.quan.keys())

    #--------------------------------------------------------------------------------
    def list_variables( self, verbose=True, very_verbose=False, use_constfile=False, retval=False, narrow=False ) :
        """
        Print out a list of variables contained in the grib file

        :param verbose:        Be more verbose

        :param very_verbose:   Be extremely verbose

        :param use_constfile:  Use list variables in self.constfile, not in self.filename

        :param retval:         If set to True, return a dictionary containing a list of all variables.
        """

        if use_constfile :
            if self.constfiledir is None:
                self.load_variables([],use_constfile=True,verbose=verbose)
            vardir = self.constfiledir
        else :
            if self.vardir is None :
                self.load_variables([],verbose=verbose)
            vardir = self.vardir

        if verbose : #vardir[md['shortName']][md['typeOfLevel']][md['validityTime']]['levels'].append(
            # print list of variables ........................................
            print()
            if narrow :
                print(("%-25s %25s %5s %6s %7s %15s" % \
                    ('short_name','type_of_level','class', '#times','#levels','units')))
            else :
                print(("%-25s %25s %5s %6s %7s %15s  long name" % \
                    ('short_name','type_of_level','class', '#times','#levels','units')))

            for short_name in sorted(vardir.keys()) :
                for type_of_level in list(vardir[short_name].keys()) :

                    gtol = '-'
                    for l in list(self.level_types.keys()) :
                        if type_of_level in self.level_types[l] :
                            gtol = l
                            break

                    times = sorted(vardir[short_name][type_of_level].keys())
                    v = vardir[short_name][type_of_level][times[0]]
                    levels = sorted(v['levels'])
                    print("%-25s %25s  (%1s)  %6d %7d %15s " % (short_name, type_of_level, gtol, len(times), len(levels), v['md']['units'] ), end=' ')
                    if not narrow :
                        print(v['md']['name'], end=' ')
                        if len(levels) == 1 :
                            print(", level=", levels[0], end=' ')
                        if len(times) == 1 :
                            print(", valid at step ", times[0], end=' ')
                        if 'validityTime' in v['md'] :
                            print(' validityTime=',v['md']['validityTime'], end=' ')
                    print()
                    if very_verbose :
                        if len(levels) > 1 : print(('............ levels : ', levels))
                        if len(times)  > 1 : print(('.... valid at steps : ', times))
            print()

        if retval :
            return vardir

    #--------------------------------------------------------------------------------
    def load_variables( self, short_names_, type_of_level=None, step=None, member=None, ignore_missing=False,
                        verbose=False, very_verbose=False, benchmark=False, use_constfile=False ) :
        """
        Loads variables specified in <short_names> from COSMO output file and stores them in self.quan

        :param short_names_:  list of variable short names (grib_api short_name keys)
                              The following modifiers may be appended to each short name to select on of several
                              variables with the same short name or to load only a specific time step or ensemble
                              member:
                              :<type_of_level>  where <type of level> is either grib_api key 'typeOfLevel'
                                                or one of the classes in self.level_types.keys()
                              #<member index>   where <member index>  is the grib_api key 'perturbationNumber'
                              @<time index>     where <time index>    is the grib_api key 'step'

                              Example: 't:M#13@5' will return temperature on model levels for member 13 and step 5

        :param type_of_level: Consider only variables with the specified grib_api key 'typeOfLevel'
                              Can also be set to one of the classes in self.level_types.keys()

        :param step:          Consider only the specified time step

        :param member:        Consider only the specified member

        :param verbose:       Be more verbose

        :param very_verbose:  Be positively verbose

        :param benchmark:     Print out timing information

        :param use_constfile: Load variables from the alternative grib file self.constfile
        """


        if     ((use_constfile == False) and (self.vardir       is None)) \
            or ((use_constfile == True ) and (self.constfiledir is None)) :
            fill_vardir = True
            vardir  = CaseInsensitiveDict()
        else :
            fill_vardir = False

        varvals = CaseInsensitiveDict()
        varmeta = CaseInsensitiveDict()

        if type(short_names_) == list :
            short_names = [v.upper() for v in short_names_]
        else :
            short_names = [short_names_.upper()]

        re3 = re.compile('^([a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)$')
        re2 = re.compile('^([a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)$')
        re1 = re.compile('^([a-zA-Z0-9_\.]+)([:@#][a-zA-Z0-9_\.]+)$')
        re0 = re.compile('^([a-zA-Z0-9_\.]+)$')
              
        if use_constfile :
            fname = self.constfile
        else :
            fname = self.filename

        if verbose and not len(short_names) == 0 : print(('CosmoState: trying to load from %s : ' % fname, short_names))

        if sys.version_info[0] == 2 :
            time_clock = time.clock
        else :
            time_clock = time.process_time

        starttime_read = time_clock()
        bytes_read = 0

        # loop over the records in the grib file
        record_index = 0
        f = open(fname)
        while 1:            
            gid = grib_new_from_file(f)
            if gid is None: break
            record_index += 1

            # get metadata
            md = {}
            for nmsp in ['ls','mars','time', 'parameter', 'geography'] :
                iterid = grib_keys_iterator_new( gid, nmsp )
                while grib_keys_iterator_next(iterid):
                    keyname = grib_keys_iterator_get_name(iterid)

                    # if requested, skip problematic keys
                    if self.singlestep and (keyname in self.step_keys) :
                        continue
                    else :
                        try :
                            md[keyname] = grib_get_string(gid,keyname)

                        except GribInternalError as err:
                            md[keyname] = 0
                            print(('Warning: GribInternalError encountered for key ', keyname))

                grib_keys_iterator_delete(iterid)

            md['origin'] = fname

            #if md['shortName'] == 'q' :
            #    print md['typeOfLevel'], md['level'], md['step']
            #    if md['typeOfLevel'] != 'generalVerticalLayer' :
            #        print md
            #        sys.exit(-1)

            # store all short names as upper case strings
            md['shortName'] = self.translate_grib2user( md['shortName'].upper() )

            # level information is missing in RTTOV images
            if md['shortName'].startswith('SYNMSG'):
                md['typeOfLevel'] = 'TOA'
                md['level'] = 0

            # put unknown level types into the list of level types to avoid errors
            if 'typeOfLevel' in list(md.keys()) and not md['typeOfLevel'] in self.level_type_list:
                self.level_types["U"].append(md['typeOfLevel'])
                self.level_type_list.append(md['typeOfLevel'])

            # time information is missing if singlestep was specified or the keys in step_keys cause
            # gribapi internal errors
            if not ('validityTime' in list(md.keys())) :
                md['validityTime'] = '0000'
            if not ('step' in list(md.keys())) :
                md['step'] = 0
            md['step'] = int(md['step'])

            # determine member index, if available
            # ( see http://www.cosmo-model.org/content/model/releases/histories/int2lm_2.01.htm )
            try :
                md['member_index'] = grib_get(gid,'perturbationNumber')
            except GribInternalError as err:
                try :
                    md['member_index'] = grib_get(gid,'localActualNumberOfEnsembleNumber')
                except GribInternalError as err:
                    md['member_index'] = -1

            if very_verbose :
                print()
                print(('---------- meta information for record #%d : ' % record_index))
                for k in md :
                    print((("%20s : " % k), md[k]))
                print('----------')
                print()

            if fill_vardir : # create entry in variable directory ...............................

                if not md['shortName'] in list(vardir.keys()):
                    vardir[md['shortName']] = {}

                if 'typeOfLevel' in list(md.keys()):
                    if not md['typeOfLevel'] in list(vardir[md['shortName']].keys()):
                        vardir[md['shortName']][md['typeOfLevel']] = {}
                    if not md['step'] in list(vardir[md['shortName']][md['typeOfLevel']].keys()):
                        vardir[md['shortName']][md['typeOfLevel']][md['step']] = {'md': md}

                if 'typeOfLevel' in list(md.keys()):
                    if not 'levels' in list(vardir[md['shortName']][md['typeOfLevel']][md['step']].keys()):
                        vardir[md['shortName']][md['typeOfLevel']][md['step']]['levels'] = []

                    if 'levelist' in list(md.keys()):
                        vardir[md['shortName']][md['typeOfLevel']][md['step']]['levels'].append(
                            int(md['levelist']))
                    elif 'level' in list(md.keys()):
                        vardir[md['shortName']][md['typeOfLevel']][md['step']]['levels'].append(
                            int(md['level']))
                    #if md['shortName'] == 'q' :
                    #    print md['typeOfLevel'], md['validityTime'], md['step'], len(vardir[md['shortName']][md['typeOfLevel']][md['step']]['levels']), vardir[md['shortName']][md['typeOfLevel']][md['step']]['levels']

            # check whether record belongs to one of the requested variables
            hit = False
            for short_name in short_names :

                # parse variable name of the form <name>[:<type of level>][#<member index>][@<step index>]
                m3 = re3.match(short_name)
                if m3 :
                    shn = m3.group(1)
                    tokens = m3.group(2,3,4)
                else :
                    m2 = re2.match(short_name)
                    if m2 :
                        shn = m2.group(1)
                        tokens = m2.group(2,3)
                    else :
                        m1 = re1.match(short_name)
                        if m1 :
                            shn = m1.group(1)
                            tokens = [m1.group(2)]
                        else :
                            m0 = re0.match(short_name)
                            if m0 :
                                shn = m0.group(1)
                                tokens = []
                            else :
                                raise ValueError('Cannot parse variable name '+short_name)
                stp = step
                mem = member
                tol = type_of_level
                for t in tokens :
                    if t[0] == ':' :
                        tol = t[1:]
                    elif t[0] == '#' :
                        mem = int(t[1:])
                    elif t[0] == '@' :
                        stp = int(t[1:])
                    else :
                        raise ValueError('Unknown variable name modifier '+t)

                if tol is None and not (self.default_leveltype is None) :
                    tol = self.default_leveltype

                if very_verbose :
                    print('VARIABLE NAME ', shn, end=' ')
                    if not (stp is None) : print(' stp=', stp, end=' ')
                    if not (mem is None) : print(' mem=', mem, end=' ')
                    if not (tol is None) : print(' tol=', tol, end=' ')
                    print()

                # check if record would in principle match
                potential_match = True
                if shn != md['shortName'] :
                    potential_match = False
                    if very_verbose : print(('name mismatch ', md['shortName']))
                if (not (stp is None)) and (stp != md['step']) :
                    potential_match = False
                    if very_verbose : print(('step mismatch ', md['step']))
                if (not (mem is None)) and (mem != md['member_index']) :
                    potential_match = False
                    if very_verbose : print(('mem mismatch ', md['member_index']))
                if not (tol is None) :
                    if tol == md['typeOfLevel'] :
                        pass
                    elif (tol in list(self.level_types.keys())) and (md['typeOfLevel'] in self.level_types[tol]) :
                        pass
                    else :
                        potential_match = False
                        if very_verbose : print(('tol mismatch ', md['typeOfLevel']))

                if not self.ignore_leveltype is None :
                    if (self.ignore_leveltype in list(self.level_types.keys())) and (md['typeOfLevel'] in self.level_types[self.ignore_leveltype]) :
                        #print( 'ignoring {} with level type {}'.format( md['shortName'], md['typeOfLevel']) )
                        potential_match = False

                if not potential_match :
                    continue

                if very_verbose : print(('  --> could match ', md['typeOfLevel'], md['member_index'], md['step']))

                hit = True
                if short_name in varmeta : # we have records for this variable name before
                    if very_verbose : print('     the variable is already known.')

                    # do step, member and tol agree with the already known variable?
                    if md['typeOfLevel']  != varmeta[short_name]['typeOfLevel']  : hit = False
                    if not (mem is None) :
                        if md['member_index'] != varmeta[short_name]['member_index'] : hit = False
                    if not (step is None) :
                        if md['step']         != varmeta[short_name]['step']         : hit = False
                    
                if not hit :
                    if very_verbose : print('     but is not compatible to previously read records...')
                else :
                    if very_verbose : print('     and is not yet known or compatible to previously read records...')

                if hit :
                    vname = short_name
                    break

            if hit : # the record belongs to a requested variable --> read & store it

                # get horizontal dimensions
                try :
                    nx = grib_get_long(gid,"Ni")
                    ny = grib_get_long(gid,"Nj")
                except :
                    # probably unstructure grid -> nx will be set later
                    nx = ny = 0

                md['nx'], md['ny'] = nx, ny

                # determine level
                if 'levelist' in list(md.keys()) :
                    level = int(md['levelist'])
                elif 'level' in list(md.keys()) :
                    level = int(md['level'])
                else :
                    level = 0

                if not (vname in list(varvals.keys())) :
                    varvals[vname] = {}
                    varmeta[vname] = md

                if not level in list(varvals[vname].keys()) :
                    varvals[vname][level] = {}
                
                if not md['step'] in list(varvals[vname][level].keys()) :
                    varvals[vname][level][md['step']] = {}

                if ny == 0 : # this is probably an unstructured ICON grid...
                    varvals[vname][level][md['step']][md['member_index']] = grib_get_values(gid)[...]
                    nx = varvals[vname][level][md['step']][md['member_index']].size
                    md['nx'] = nx
                else :
                    varvals[vname][level][md['step']][md['member_index']] = grib_get_values(gid).reshape((ny,nx))

                #print '>>>', type(varvals[vname][level][md['step']][md['member_index']][0,0]), sys.getsizeof( varvals[vname][level][md['step']][md['member_index']][0,0] )
                # getsizeof() = 32 for type=float64 ???

                bytes_record = nx * maximum(ny,1) * 8 #* sys.getsizeof( varvals[vname][level][md['step']][md['member_index']][0,0] )
                #bytes_record = sys.getsizeof( varvals[vname][level][md['step']][md['member_index']] )
                bytes_read += bytes_record

                if verbose : print(('found %d byte record for %s ( nx=%d, ny=%d, level=%d, step=%d, tol=%s)' % \
                                    (bytes_record,vname,nx,ny,level,md['step'],md['typeOfLevel'])))

            grib_release(gid)

        endtime_read = time_clock()
        # all records are read .................................................

        # if necessary, save variable directory
        if fill_vardir :
            if use_constfile :
                self.constfiledir = vardir
            else :
                self.vardir = vardir

        # join 2D levels to higher-dimensional variables, if necessary
        starttime_join = time_clock()

        for vname in list(varvals.keys()) :

            nx = varmeta[vname]['nx']
            ny = varmeta[vname]['ny']

            nz = len(varvals[vname])
            z0 = list(varvals[vname].keys())[0]

            nt = len(varvals[vname][z0])
            t0 = list(varvals[vname][z0].keys())[0]

            nm = len(varvals[vname][z0][t0])
            m0 = list(varvals[vname][z0][t0].keys())[0]

            if verbose : print(('>>>', vname, nx, ny, nz, nt, nm))

            if ny > 0 :
                dimensions = [ny, nx]
                dimnames   = ['lat','lon']
            else :
                dimensions = [nx]
                dimnames   = ['cell_index']

            if nz > 1 :
                dimensions.append(nz)
                dimnames.append('level')
                levels = sorted(varvals[vname].keys())
                varmeta[vname]['levels'] = levels
            else :
                levels = [z0]

            if nt > 1 :
                dimensions.append(nt)
                dimnames.append('time')
                steps = sorted(varvals[vname][z0].keys())
                varmeta[vname]['steps'] = steps
            else :
                steps = [t0]

            if nm > 1 :
                dimensions.append(nm)
                dimnames.append('ensemble')
                members = sorted(varvals[vname][z0][t0].keys())
                varmeta[vname]['members'] = members
            else :
                members = [m0]

            if ny > 0 :
                varmeta[vname]['nlon']   = nx
                varmeta[vname]['nlat']   = ny
            else :
                varmeta[vname]['ncells']   = nx

            varmeta[vname]['nlevel'] = nz
            varmeta[vname]['ntime']  = nt
            varmeta[vname]['nens']   = nm
            varmeta[vname]['dimnames'] = dimnames
            varmeta[vname]['ndim'] = len(dimnames)
            self.meta[vname] = varmeta[vname]

            if nz == 1 and nt == 1 and nm == 1 :
                # 2d array : just copy it...
                self.quan[vname] = varvals[vname][z0][t0][m0]
            else :
                # create empty array for variable
                if very_verbose :
                    print('creating empty variable with dimensions ', dimensions)
                self.quan[vname] = zeros(dimensions)

                # fill with data from records
                for k, kk in enumerate(levels) :
                    for l, ll in enumerate(steps) :
                        for m, mm in enumerate(members) :

                            if very_verbose :
                                print('filling part of the variable with something of dimensions ', varvals[vname][kk][ll][mm].shape)


                            if nz > 1 :
                                if nt > 1 :
                                    if nm > 1 :
                                        self.quan[vname][...,k,l,m] = varvals[vname][kk][ll][mm]
                                    else :
                                        self.quan[vname][...,k,l]   = varvals[vname][kk][ll][mm]
                                else :
                                    if nm > 1 :
                                        self.quan[vname][...,k,m]   = varvals[vname][kk][ll][mm]
                                    else :
                                        self.quan[vname][...,k]     = varvals[vname][kk][ll][mm]
                            else :
                                if nt > 1 :
                                    if nm > 1 :
                                        self.quan[vname][...,l,m]   = varvals[vname][kk][ll][mm]
                                    else :
                                        self.quan[vname][...,l]     = varvals[vname][kk][ll][mm]
                                else :
                                    if nm > 1 :
                                        self.quan[vname][...,m]     = varvals[vname][kk][ll][mm]
                                    else :
                                        self.quan[vname][...]       = varvals[vname][kk][ll][mm]
        endtime_join = time_clock()

        if benchmark : #verbose :
                print(('time for reading %f MB : %f sec --> %f MB/sec' % ( bytes_read/1e6, endtime_read-starttime_read,
                                                                        (bytes_read/1e6)/(endtime_read-starttime_read))))
                print(('time for joining records : %f sec' % (endtime_join - starttime_join)))

        # check whether all requested variables were found
        missing = []
        for vname in short_names :
            if vname not in self.quan:
                missing.append(vname)

        if len(missing) > 0 :
            n_computed = 0
            for miss in missing :
                if not miss in self.computable_quantities :
                    print('CosmoState: The following variable could not be found in %s : ' % fname, miss)
                else :
                    if self.constfile == '' :
                        if verbose: print(('>>> will be computed: ', miss))
                        self.compute( miss, store=True )
                        n_computed += 1
                    else :
                        if verbose :
                            print('not computing ', miss, ', because it may be in the constfile...')
            if n_computed < len(missing) :
                if not use_constfile and self.constfile != '' and os.path.exists(self.constfile) :
                    self.load_variables( missing, type_of_level=type_of_level, step=step, verbose=verbose, use_constfile=True )
                else :
                    if not ignore_missing :
                        raise ValueError('CosmoState: Could not find variable '+vname)

    #--------------------------------------------------------------------------------
    def vertical_cut( self, vnames, lat=None, lon=None ) :

        if not (lat is None ) :
            if lon is None : # lat=const cut
                print('implement me')
            else :           # cut along path defined by (lat,lon)
                print('implement me')
        else : # lat is None
            if not (lon is None) : # lon=const cut
                print('implement me')
            else : # lat and lon are None
                raise ValueError('CosmoState.vertical_cut : cut location not specified')

    #--------------------------------------------------------------------------------
    def latlons_of_variable( self, vname, **kw ) :

        if not vname in self.meta :
            self.load_variables( [vname])
        return cosmo_grid( configuration=self.meta[vname], **kw )

    #--------------------------------------------------------------------------------
    def cosmo_indices( self, lat, lon, method='nearest', variable=None, debug=False ) :
        """
        Compute COSMO array indices from given lat and lon arrays.
        lat/lon combinations outside of COSMO domain result in indices -1.

        :param lat:    Latitude array
        :param lon:    Longitude array
        :param method: Interpolation method -- 'nearest' is fast, 'linear' is more accurate
        :param variable: Use lat/lon coordinates of this variable instead of RLAT, RLON
        :returns ilat_itp, ilon_itp : COSMO index arrays
        """

        import scipy.interpolate

        if variable is None :
            rlat, rlon = self['RLAT'], self['RLON']
        else :
            rlat, rlon = self.latlons_of_variable(variable)

        nlat, nlon = rlat.shape
        ilat =            arange(nlat).repeat(nlon).reshape((nlat,nlon))
        ilon = transpose( arange(nlon).repeat(nlat).reshape((nlon,nlat)) )
        
        if debug :

            print(('input lat = ', lat))
            print(('input lon = ', lon))

            import pylab as plt
            plt.figure(1)
            plt.clf()
            plt.imshow( rlat, origin='lower')
            plt.colorbar()
            plt.savefig('rlat.png') # varies with the first (larger, 461) dimension
            plt.clf()
            plt.imshow( rlon, origin='lower')
            plt.colorbar()
            plt.savefig('rlon.png') # varies with the second (smaller, 421) dimension

            plt.clf()
            plt.imshow( ilat, origin='lower')
            plt.colorbar()
            plt.savefig('ilat.png') # varies with the first (larger, 461) dimension
            plt.clf()
            plt.imshow( ilon, origin='lower')
            plt.colorbar()
            plt.savefig('ilon.png') # varies with the second (smaller, 421) dimension

            print()
            print('variation in first dimension')
            print(('rlat[0:3,0]', rlat[0:3,0]))
            print(('ilat[0:3,0]', ilat[0:3,0]))
            print(('rlon[0:3,0]', rlon[0:3,0]))
            print(('ilon[0:3,0]', ilon[0:3,0]))
            print()
            print('variation in second dimension')
            print(('rlat[0,0:3]', rlat[0,0:3]))
            print(('ilat[0,0:3]', ilat[0,0:3]))
            print(('rlon[0,0:3]', rlon[0,0:3]))
            print(('ilon[0,0:3]', ilon[0,0:3]))

        points = transpose(vstack((rlat.ravel(),rlon.ravel())))
        
        if debug :
            print()
            print('shapes:')
            print(('rlat, rlon  : ', rlat.shape, rlon.shape))
            print((' + ravel()  : ', rlat.ravel().shape))
            print((' + vstack() : ', vstack((rlat.ravel(),rlon.ravel())).shape))
            print((' + transp.  : ', transpose(vstack((rlat.ravel(),rlon.ravel()))).shape))

        if method == 'nearest' : # fast, results are integers
            ilat_itp = scipy.interpolate.griddata( points, ilat.ravel(), (lat, lon), method='nearest', fill_value=-1)
            ilon_itp = scipy.interpolate.griddata( points, ilon.ravel(), (lat, lon), method='nearest', fill_value=-1)
        else : # e.g. method = 'linear' : slower but more accurate, results are floats
            ilat_itp = scipy.interpolate.griddata( points, ilat.ravel().astype(float), (lat, lon), method=method, fill_value=-1)
            ilon_itp = scipy.interpolate.griddata( points, ilon.ravel().astype(float), (lat, lon), method=method, fill_value=-1)
        
        if debug :
            print()
            print(('output ilat ', ilat_itp, type(ilat_itp[0])))
            print(('output ilon ', ilon_itp))
            print()
            print(('rlat(ilat,ilon)', rlat[ilat_itp.astype(int),ilon_itp.astype(int)]))        
            print(('rlon(ilat,ilon)', rlon[ilat_itp.astype(int),ilon_itp.astype(int)]))
            print()

        return ilat_itp, ilon_itp

    #--------------------------------------------------------------------------------
    def compute( self, vnames, store=False ) :
        """Compute various derived quantities"""

        results = []

        if type(vnames) == list :
            vnames_ = vnames
        else :
            vnames_ = [vnames]
    
        for vname in vnames_ :
            res = None
            meta = None

            if vname == 'HML' :
                res = self.adjust_levels(self['HHL'], like=self['HHL'][:,:,1:])
                meta = {'name': 'geometric model level height above sea level', 'units': 'm'}

            elif vname == 'PHL' : # pressure in model layers
                z = self['HHL']
                res = self.pref(z) + self.adjust_levels( self['PP'], like=self['HHL'] )
                meta = { 'name':'pressure on half model levels', 'units':'Pa'}

            elif vname == 'P' : # pressure at model levels
                z = self['HML']
                res = self.pref(z) + self['PP']
                meta = {'name': 'pressure on full model levels', 'units':'Pa'}

            elif vname == 'RLAT' :
                try :
                    self.load_variables(['T:M'])
                    gridinf = self.meta['T:M']
                except :
                    print('WARNING: could not load T:M to determine grid definition -> using default grid.')
                    gridinf = None
                res = cosmo_grid(configuration=gridinf)[0]
                meta = {'name':'geographical latitude', 'units':'deg N' }

            elif vname == 'RLON' :
                try :
                    self.load_variables(['T:M'])
                    gridinf = self.meta['T:M']
                except :
                    print('WARNING: could not load T:M to determine grid definition -> using default grid.')
                    gridinf = None
                res = cosmo_grid(configuration=gridinf)[1]
                meta = {'name':'geographical longitude', 'units':'deg E' }

            elif vname == 'DEN' :
                Rd = 287.058 # J/(kg·K)
                Rv = 461.495 # J/(kg·K)
                res = self['PRES'] / ( Rd*self['T']*(1+(Rv/Rd-1)*self['QV']-self['QC']-self['QI']))
                meta = {'name': 'density', 'units':'kg/m3'}

            elif vname == 'dz' :
                res = self['HHL'][:,:,:-1] - self['HHL'][:,:,1:]
                meta = {'name': 'level spacing', 'units':'m'}

            elif vname == 'AREA' : # surface area of cell in m**2
                res = cosmo_grid( area=True )
                meta = {'name': 'cell area', 'units':'m2'}

            elif vname == 'TQV' :
                res = (self['QV'] *self['DEN']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
                meta = {'name': 'Total Column-Integrated Water vapour', 'units':'kg m-2'}

            elif vname == 'TQC' :
                res = (self['QC'] *self['DEN']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
                meta = {'name': 'Total Column-Integrated Cloud Water', 'units':'kg m-2'}

            elif vname == 'TQI' :
                res = (self['QI'] *self['DEN']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
                meta = {'name': 'Total Column-Integrated Cloud Ice', 'units':'kg m-2'}

            elif vname == 'TQC_DIA' :
                res = (self['QC_DIA'] *self['DEN']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
                meta = {'name': 'Total Column-Integrated Cloud Water (incl. subgrid contrib.)', 'units':'kg m-2'}

            elif vname == 'TQI_DIA' :
                res = (self['QI_DIA'] *self['DEN']*self['dz']).sum(axis=2) # kg/kg * kg/m3 * m = kg/m2
                meta = {'name': 'Total Column-Integrated Cloud Ice (incl. subgrid contrib.)', 'units':'kg m-2'}

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
                zpvs = b1*exp( b2w*(self['T']-b3) / (self['T']-b4w) )
                zqvs = rdv*zpvs / (self['PRES'] - o_m_rdv*zpvs)  # was 'P' until 2020/08/09
                # Set minimum value of relhum to 0.01 (was 0 before)
                res = maximum( self['Q']/zqvs * 100, 0.01 )     # was 'QV' until 2020/08/09
                meta = {'name': 'Relative Humidity', 'units':'%'}

            else :
                raise ValueError("CosmoState.compute: I don't know how to compute "+vname)

            results.append(res)
            if store :
                #print 'storing '+vname
                self.quan[vname] = res
                meta['short_name'] = vname
                self.meta[vname] = meta

        if type(vnames) == list :
            return results
        else :
            return results[0]

    def adjust_levels( self, src, like=None, nz=None ) :
        """
        Return a copy of src with the same number of vertical levels as in trg.
        Linear interpolation or extrapolation is used.
        The number of vertical levels in src and trg is not allowed to differ by more than 1.
        """

        nz_src = src.shape[2]
        if nz is None :
            if like is None :
                raise ValueError("adjust_level: specify target nz")
            else :
                nz_trg = like.shape[2]
        else :
            nz_trg = nz
        trg_shape = list(src.shape)
        trg_shape[2] = nz_trg

        if nz_trg == nz_src-1 : # levels to layers
            retval = 0.5*( src[:,:,1:] + src[:,:,:-1])

        elif nz_trg == nz_src+1 : # layers to levels
            retval = zeros(trg_shape)
            retval[:,:,1:-1] = 0.5*( src[:,:,1:] + src[:,:,:-1])
            retval[:, :,  0] = src[:, :,  0] - (retval[:, :,  1] - src[   :, :,  0] )
            retval[:, :, -1] = src[:, :, -1] + (src[   :, :, -1] - retval[:, :, -2] )

        else :
            raise ValueError("adjust_levels: I don't know how to do that for nz_src=%d and nz_trg=%d" % (nz_src,nz_trg))

        return retval

    #--------------------------------------------------------------------------------
    def pref( self, z ) :
        """Compute reference pressure for given height field z"""

        # from data_constants.f90 :
        r_d  = 287.05
        g    = 9.80665
        # from COSMO User's Guide, Section 3.1
        psl  = 100000.0 # Pa
        tsl  = 288.15   # K
        beta = 42       # K
        return psl * exp( -(tsl/beta) * (1 - sqrt( 1 - (2*beta*g*z)/(r_d*(tsl**2)) ) ) )

    # --------------------------------------------------------------------------------
    def distribution( self, zvarname, varnames, binedges=None, nbins=None, stats=None, verbose=False ) :
        """
        Compute distribution of mean values, std. dev., min. and max. values of the
        variables <varnames> as function of the variable <zvarname>
        """

        if type(varnames) == list :
            varnames_ = varnames
        else :
            varnames_ = [varnames]

        if (nbins is None) and (binedges is None) :
            nbins = 20
        if binedges is None :
            binedges = linspace( self[zvarname].min(), self[zvarname].max(), nbins+1 )
        nbins = binedges.size-1

        if stats is None :
            stats = ['mean','std'] #,'min','max']

        profiles = []
        for vname in varnames_ :
            profiles.append({})
            for s in stats :
                profiles[-1][s] =  zeros(nbins)
        npoints = zeros(nbins,dtype=int)

        bincenters = zeros(nbins)
        for i in range(nbins) :
            zmin = binedges[i]
            zmax = binedges[i+1]
            bincenters[i] = 0.5*( zmin + zmax )
            idcs = where( (self[zvarname] >= zmin) & (self[zvarname] < zmax) )
            npoints[i] = len(idcs[0])
            if verbose : print(("distribution: %f < %s < %f : %d points" % (zmin,zvarname,zmax,npoints[i])))

            if npoints[i] == 0 :
                print(("Warning: 0 points in bin ", i))
                for ivn, vn in enumerate(varnames_):
                    for s in stats:
                        profiles[ivn][s][i] = 0
            else :
                for ivn,vn in enumerate(varnames_) :
                    for s in stats :
                        if s == 'mean' :
                            profiles[ivn][s][i] = self[vn][idcs].mean()
                        elif s == 'std' :
                            profiles[ivn][s][i] = self[vn][idcs].std()
                        elif s == 'min':
                            profiles[ivn][s][i] = self[vn][idcs].min()
                        elif s == 'max':
                            profiles[ivn][s][i] = self[vn][idcs].max()
                        else :
                            raise ValueError('distribution: unknown statistics type '+s)

        if type(varnames) == list:
            return profiles, bincenters, npoints
        else :
            return profiles[0], bincenters, npoints


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='List grib file variables')
    parser.add_argument( '-V', '--variables',  dest='variables', help='comma-separated list of variables to be inspected', default='' )
    parser.add_argument( '-v', '--verbose',    dest='verbose',   help='be more verbose', action='store_true' )
    parser.add_argument(       '--grid-test',  dest='grid_test', help='recompute coordinates', action='store_true' )
    parser.add_argument(       '--translation',  dest='translation', help='translate variable names', default=None )
    parser.add_argument( 'gribfiles', metavar='gribfiles', help='grib file names', nargs='*' )
    args = parser.parse_args()

    for fname in args.gribfiles :

        print()
        print(('[ I N S P E C T I N G ] ' + fname + ' -' * 50))
        cs = CosmoState( fname, verbose=args.verbose, translation=args.translation )

        if args.grid_test :

            # if they are not found in the grib file, these variables will be set to COSMO-DE default values
            lat, lon = cs['RLAT'], cs['RLON']

            if args.variables == '' :
                print('no variables specified -- trying to use T...')
                vname = 'T'
            else :
                vname = args.variables.split(',')[0]
                print(('using variable ', vname))

            #vval = cs[vname] # load variable
            lat_v, lon_v = cs.latlons_of_variable(vname)

            print(('differences between default RLAT, RLON and those for variable %s : %f %f' % ( vname, abs(lat-lat_v).max(), abs(lon-lon_v).max())))

            meta = cs.meta[vname]
            pollon =  float(meta['longitudeOfSouthernPoleInDegrees']) - 180
            pollat = -float(meta['latitudeOfSouthernPoleInDegrees'])

            print('non-rotated coordinates:')
            print(('LAT', lat.shape, lat.min(), lat.max()))
            print(('LON', lon.shape, lon.min(), lon.max()))

            lat_r, lon_r = nonrot_to_rot_grid(lat,lon,pollon=pollon,pollat=pollat)

            print('rotated coordinates:')
            print(('LAT', lat_r.shape, lat_r.min(), lat_r.max()))
            print(('LON', lon_r.shape, lon_r.min(), lon_r.max()))

            from matplotlib import pyplot as plt
            plt.figure(1,figsize=(10,10))
            plt.clf()
            plt.scatter( lon[  ::5,::5], lat[  ::5,::5], c='r', s=20, edgecolors='', linewidths=0, label='default RLAT, RLON' )
            plt.scatter( lon_v[::5,::5], lat_v[::5,::5], c='k', s=5,  edgecolors='', linewidths=0, label='LAT, LON for '+vname )
            plt.legend(frameon=False)
            plt.savefig('coords.png')

            plt.clf()
            plt.scatter( lon_r[::5,::5], lat_r[::5,::5], c='r', s=20, edgecolors='', linewidth=0, label='default rotated LAT, LON' )
            plt.savefig('coords_rot.png')

            #for k in cs.meta['RLON'] :
            #    print k, ' = ', cs.meta['RLON'][k]
            continue

        if args.variables == '' :
            print('list of variables :')
            cs.list_variables( verbose=True, very_verbose=args.verbose )

        else :
            varnames = args.variables.split(',')
            print()
            print(('loading ', varnames, '...'))
            cs.load_variables( varnames, verbose=args.verbose )
            print('done.')

            for varname in  varnames :
                print()
                print(('variable %s :' % varname))
                print('meta information :')
                for k in list(cs.meta[varname].keys()) :
                    print(('    %30s : ' % k, cs.meta[varname][k]))
                print(('shape : ', cs[varname].shape))
                print(('min / mean / max values : ', cs[varname].min(), cs[varname].mean(), cs[varname].max()))

        #vn = 'v:M#13'
        #vn = 'CAPE_ML'
        #cs.load_variables(vn)
        #print "DONE ", cs[vn].shape, cs.meta[vn]['typeOfLevel'], cs.meta[vn]['member_index'], cs.meta[vn]['step'], cs.meta[vn]['dimnames']
        #cs.load_variables('HHL:M#13@0',very_verbose=True)

        #print 'u',   cs['u'].shape,   cs.meta['u']['typeOfLevel'],   cs.meta['u']['levels']
        #print 'u:P', cs['u:P'].shape, cs.meta['u:P']['typeOfLevel'], cs.meta['u:P']['levels']
        #print 'u:M', cs['u:M'].shape, cs.meta['u:M']['typeOfLevel'], cs.meta['u:M']['levels']
        #print 'u:Z', cs['u:Z'].shape, cs.meta['u:Z']['typeOfLevel'], cs.meta['u:Z']['levels']
        #print 'SYNMSG_RAD_CS_IR10.8', cs['SYNMSG_RAD_CS_IR10.8'].shape
        #print 'prmsl', cs['prmsl'].shape, cs.meta['prmsl']['typeOfLevel'], cs.meta['prmsl']['levels']       

    if False :
        # debug cosmo_indices
        cs = CosmoState('/home/userdata/leo.scheck/bacy_ahutt/cosmo_letkf/data/1000.05b/20140516060000/laf20140516080000.det',
                        constfile='/home/userdata/leo.scheck/bacy_ahutt/cosmo_letkf/data/1000.05b/20140516090000/lff20140516080000.det',
                        preload=['QC','W','HSURF','T_SO','RLAT'],verbose=True)
        #cs['T']
        print((cs.cosmo_indices( lat=arange(48,52), lon=arange(8,12), debug=True )))
        print((cs.compute('PHL').shape))




