#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ E X P
#
#  2020.6 L.Scheck 

from __future__ import absolute_import, division, print_function
import os, sys, subprocess, tempfile, re, pickle
from time import perf_counter
from datetime import datetime, timedelta
from math import fabs
from kendapy.bacy_utils import t2str, str2t, add_branch, adjust_time_axis

#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="Provide basic information about bacy experiment parameters and files")

    # plot type and parameters
    parser.add_argument( '-f', '--plot-files',      dest='plot_files',      action='store_true',  help='generate plot showing which files are available at which time' )
    parser.add_argument( '-c', '--config',          dest='config',          action='store_true',  help='print out configuration parameters' )
    parser.add_argument(       '--no-cache',        dest='no_cache',        action='store_true',  help='do not read (but write) file tree cache DEFAULT SINCE 2021/05/22!!!' )
    parser.add_argument(       '--cache',           dest='cache',           action='store_true',  help='read file tree cache, if available' )

    parser.add_argument(       '--time-range', dest='time_range',     default=None,              help='time range, e.g. 20190629170000,20190630120000' )

    parser.add_argument( '-o', '--output-path', dest='output_path',        default=None,              help='output path' )
    parser.add_argument( '-i', '--image-type',  dest='image_type',         default='png',             help='[ png | eps | pdf ... ]' )

    parser.add_argument( '-v', '--verbose',        dest='verbose',         action='store_true',  help='be more verbose' )

    parser.add_argument( 'experiments', metavar='<experiment path(s)>', help='path(s) to experiment(s) / files', nargs='+' )

    return parser


#----------------------------------------------------------------------------------------------------------------------
class BacyExp(object) :
    """Class representing a bacy experiment"""

    #--------------------------------------------------------------------------
    def __init__( self, exp_path, verbose=False, model=None, read_cache=False ) :
        """Initialize object using information from the experiment directory"""

        self.exp_path = exp_path # full path

        p =  exp_path[:-1] if exp_path.endswith('/') else exp_path
        self.exp_dir = p.split('/')[-1] # experiment directory name = last element of the path
        
        self.verbose  = verbose

        # guess which model was used
        if model is None :
            self.model = self.guess_model( exp_path )
        else :
            self.model = model
        
        #t0 = clock()

        # read all module config files
        self.config  = self.get_config( self.exp_path, self.model )
        self.n_ens   = self.config['BA_ENSIZE_ASS']

        # grid will be loaded on demand
        self.grid = {}

        #t1 = clock()

        # initialize file tree
        cachedirname = '.bacy_exp_py_cache'
        cachedirpath = os.path.join( os.getenv('HOME'), cachedirname )
        cachefilepath = os.path.join( cachedirpath, self.exp_dir+'__filetree_cache.pickle' )
        compute_ff = True
        save_ff = False
        if os.path.exists( cachedirpath ) :
            if os.path.exists( cachefilepath ) and read_cache :
                print('loading cached file tree from {} ...'.format(cachefilepath))
                now = perf_counter()
                with open( cachefilepath,'rb') as f :
                    ff = pickle.load( f )
                print('loading file tree cache took {}sec...'.format(perf_counter()-now))
                compute_ff = False
            else :
                save_ff = True
        if compute_ff :
            now = perf_counter()
            ff = self.find_files( self.exp_path, config=self.config, verbose=verbose )
            print('finding files took {}sec...'.format(perf_counter()-now))
        if save_ff :
            now = perf_counter()
            with open( cachefilepath,'wb') as f :
                #print('saving file tree to cache file {} ...'.format(cachefilepath))
                pickle.dump( ff, f, pickle.HIGHEST_PROTOCOL )
            #print('saving file tree took {}sec...'.format(perf_counter()-now))

        #t2 = clock()
        #print('CONFIG TIME = ', t1-t0, ', FILE TREE TIME = ', t2-t1 )

        self.file_tree      = ff['filetree']
        self.obs_types      = ff['obstypes']
        self.file_types     = ff['filetypes']
        self.valid_times    = ff['valid_times']
        self.fc_start_times = ff['fc_start_times']
        self.fc_lead_times  = ff['fc_lead_times']
        

    #--------------------------------------------------------------------------
    def get_filename( self, filetype, time_valid=None, time_start=None, lead_time=None,
                      latlon=False, mem=None, obs_type=None, mon_type=None, visop_type=None ) :
        """return file name, e.g.
           - spread of 1h first guess forecast started at 10UTC : get_filename('fg',time_valid='2019062911',mem='spread')
        """

        if not time_valid is None :
            tv  = time_valid if type(time_valid) == datetime else str2t(time_valid)
            tv3 = datetime( tv.year, tv.month, tv.day, (tv.hour//3)*3, 0, 0 )

        if not time_start is None :
            ts  = time_start if type(time_start) == datetime else str2t(time_start)
            ts3 = datetime( ts.year, ts.month, ts.day, (ts.hour//3)*3, 0, 0 )
            dtbc_sec = (ts - ts3).total_seconds()
            dtbc_hour = dtbc_sec // 3600
            dtbc_min  = (dtbc_sec - 3600*dtbc_hour) // 60
            dtbc = '{:02.0f}{:02.0f}'.format(dtbc_hour,dtbc_min)
            tl = '{:d}'.format(int(lead_time.total_seconds()/3600)) if type(lead_time) == timedelta else '{:d}'.format(int(lead_time))
            # FIXME what about e.g. 90min forecasts?

        if mem is None or mem == 'det' or mem == 0 :
            mem_ = ''
            imem = 0
        elif mem in ['mean','spread'] :
            mem_ = mem
            imem = None
        else :
            imem = int(mem)
            mem_ = '{:03d}'.format(int(mem))
        ensmem = '_ens'+mem_ if mem_ != '' else ''
        mem_ = ('.' + mem_)  if mem_ != '' else ''
        ensmem0 = '_ens{:03d}'.format(imem) if not imem is None else None

        grid = self.res_to_grid(self.config['BA_IL_IRES_DET']) if mem_ == '' else self.res_to_grid(self.config['BA_IL_IRES_ENS'])

        if filetype in ['fg','fg_ll','an','an_inc'] :
            ft   = filetype.replace('fg','fc')
            
            ll   = '_ll'  if latlon             else ''
            inc  = '_inc' if '_inc' in filetype else ''
            filename = '{}/data/{}/{}_{}{}.{}{}{}'.format( self.exp_path, t2str(tv3), ft, grid, ll, t2str(tv), inc, mem_ )

        elif filetype == 'fof' :                        
            filename = '{}/data/{}/fof{}_{}{}.nc'.format( self.exp_path, t2str(tv3), obs_type, t2str(tv), ensmem )

        elif filetype.startswith('seviri') :
            if '_obs' in filetype :
                em = ''
            else :
                #em = ensmem0
                em = ensmem
            # e.g.     .../data/20200802120000/seviri_sim_202008021200_ens000.grb
            if '2019_July' in self.exp_path :
                #print('USING 2019_July CONVENTION FOR DATOOL/SEVIRI FILES!')
                filename = '{}/data/{}/{}_{}{}.grb2'.format( self.exp_path, t2str(tv3), filetype, t2str(tv)[:12], em )
            elif '2020_November' in self.exp_path :
                #print('USING 2020_November CONVENTION FOR DATOOL/SEVIRI FILES!')
                filename = '{}/data/{}/{}_{}{}.grb'.format( self.exp_path, t2str(tv3), filetype, t2str(tv)[:12], '' if  '_obs' in filetype else ensmem0 )
            else :
                filename = '{}/data/{}/{}_{}{}.grb'.format( self.exp_path, t2str(tv3), filetype, t2str(tv)[:12], em )

        elif filetype.startswith('visop') :
            if not time_start is None : # main forecast
                # e.g.      <exp_dir>/visop_std/20200802120000/visop.20200802120000_02hr/refl_sat_SEVIRI_VIS006.nc
                filename = '{}/visop{}/{}/visop.{}_{}hr{}/refl_sat_SEVIRI_VIS006.nc'.format( 
                            self.exp_path, '/' + visop_type if not visop_type is None else '',
                            t2str(ts), t2str(ts), tl.zfill(2), mem_ )
            else : # first guess
                # e.g.     <exp_dir>/visop_std/20200802120000/visop.20200802120000.022/refl_sat_SEVIRI_VIS006.nc
                filename = '{}/visop{}/{}/{}.{}{}/refl_sat_SEVIRI_VIS006.nc'.format( self.exp_path,
                            '/' + visop_type if not visop_type is None else '',
                            t2str(tv3), filetype, t2str(tv), mem_ )

        elif filetype == 'ekf' :
            filename = '{}/feedback/{}/ekf{}.nc.{}'.format( self.exp_path, t2str(tv3), obs_type, t2str(tv) )

        elif filetype == 'mon' :
            filename = '{}/feedback/{}/mon{}.nc.{}'.format( self.exp_path, t2str(tv3), mon_type, t2str(tv) )

        elif filetype == 'ver' :
            filename = '{}/veri/{}/ver{}_{}.nc'.format( self.exp_path, t2str(tv3), obs_type, t2str(tv) )

        elif filetype in ['fc','fc_ll'] :
            # e.g. fc_R19B07_ll.20190629180000_0hr
            ft   = filetype[:2]
            ll   = '_ll'  if latlon             else ''
            filename = '{}/data/{}/main{}/{}_{}{}.{}_{}hr{}'.format( self.exp_path, t2str(ts3), dtbc, ft, grid, ll, t2str(ts), tl, mem_ )

        elif filetype == 'grid' :            
            grid_fname = self.res_to_grid( self.config['BA_IL_IRES_DET'] if mem_ == '' else self.config['BA_IL_IRES_ENS'], filename=True )            
            filename = '{}/{}'.format( self.config['BA_ICON_GRIDDIR'], grid_fname )

        else :
            raise ValueError('Unknown file type '+filetype)
        
        # quick & dirty fix that allows for analyzing experiments on rcl that were computed on xce
        if '/e/rhome/routfox' in filename :
            print('WARNING: Replacing /e/rhome/routfox --> /hpc/rhome/routfor')
            filename = filename.replace('/e/rhome/routfox','/hpc/rhome/routfor')

        return filename

    #-------------------------------------------------------------------------------
    @staticmethod
    def guess_model( exp_path ) :
        # FIXME
        return 'ICON-LAM'

    #-------------------------------------------------------------------------------
    @staticmethod
    def get_config( exp_path, model='ICON-LAM' ) :
        """Generate configuration information for bacy experiment using print_config,
        convert to dictionary"""

        if False : # use print_config script located in the experiment dir
            script_name = 'print_config.sh'    
            if not os.path.exists( os.path.join( exp_path, 'modules', script_name ) ) :
                raise ValueError('Could not find '+script_name)

            # call script
            cwd = os.getcwd()
            os.chdir( os.path.join( exp_path, 'modules' ) )
            output = subprocess.check_output([script_name, model])
            os.chdir( cwd )

        else : # use print_config script located in the kendapy dir
            script_name = os.path.join( os.path.dirname(__file__), 'print_config.sh' )
            temp_dir = tempfile.mkdtemp() # print_config requires a temporary dir
            output_ = subprocess.check_output([script_name, exp_path, model, temp_dir])
            if type(output_) == str :
                output = output_
            else :
                output = output_.decode()
            os.rmdir(temp_dir)

        # parse output
        config = { 'model':model, 'path':exp_path }
        mdl = model.replace('-','')
        for i, l in enumerate(output.split("\n")) :
            if l[:2] == '  ' and l[2] != '.' :
                tkns = l[2:].split('=')
                varname, varval = tkns[0].strip(), ''.join(tkns[1:]).strip()

                if '[' in varname : # array element -> put into python dictionary
                    dictname, entryname = varname.split('[')
                    if not dictname in config :
                        config[dictname] = {}
                    config[dictname][entryname.replace(']','').strip()] = varval

                else : # normal variable
                    config[varname] = varval

                    # model-specific variables: add copy without model name
                    if varname.endswith('_'+mdl) :                # e.g. BA_DELT_ASS_ICON
                        varname_gen = varname.replace('_'+mdl,'') # -->  BA_DELT_ASS
                        if not varname_gen in config :
                            config[varname_gen] = varval
                        else :
                            print('WARNING: ', varname_gen, ' exists already...')

        return config


    #-------------------------------------------------------------------------------
    @staticmethod
    def res_to_grid( res, filename=False, radiation=False, nest=False ) :
        
        #res2grid = { 'D2.1':'R19B07' }
        #if res in res2grid :
        #    return res2grid[res]
        #else :
        #    raise ValueError('I do not know which grid is used for resolution '+res)

        # definitions copied from bacy_fcns.sh / function f_bacy_icon_set_grid_spec
        # FIXME: Can we just "source" that function?

        out_array = {}

        if res == 'D2.1' :
            # Resolutions: 2.2 km (atmosphere), 26 km (radiation)
            out_array['ATMGRID']='R19B07'
            out_array['RADGRID']='R19B06'
            out_array['NESTGRID']='R03B08_N02'
            out_array['GRIDDESC']='R19B07'        # Naming convention for grids with nest

            # Select (horizontal) "Number of grid used":
            out_array['GRIDNUM_G']=44         # Dynamics
            out_array['GRIDNUM_R']=43         # Radiation
            out_array['GRIDNUM_N']=27         # Nest

        elif res == 'D2.5' :
            # Resolutions: 2.2 km (atmosphere), 26 km (radiation)
            out_array['ATMGRID']='R02B10'
            out_array['RADGRID']='R02B09'
            out_array['NESTGRID']='R03B08_N02'
            out_array['GRIDDESC']='R02B10'        # Naming convention for grids with nest

            # Select (horizontal) "Number of grid used":
            out_array['GRIDNUM_G']=42         # Dynamics
            out_array['GRIDNUM_R']=41         # Radiation
            out_array['GRIDNUM_N']=27         # Nest

        else :
            raise ValueError('Unknown grid definition')

        if radiation :
            gname = out_array['RADGRID']
            gid   = out_array['GRIDNUM_R']
        elif nest :
            gname = out_array['NESTGRID']
            gid   = out_array['GRIDNUM_N']
        else :
            gname = out_array['ATMGRID']
            gid   = out_array['GRIDNUM_G']

        if filename :
            return 'icon_grid_{:04d}_{}_L.nc'.format(gid,gname)
        else :
            return gname


    #-------------------------------------------------------------------------------
    def get_grid( self, mem='det', verbose=False ) :
        
        fn_grid = self.get_filename( 'grid', mem=mem)

        if not fn_grid in self.grid :

            import netCDF4
            grid = netCDF4.Dataset( fn_grid, 'r')

            if verbose :
                clon = grid.variables['clon'][...]
                clat = grid.variables['clat'][...]
                print( '{:35s} has {:10d} cells, {:10d} vertices. Bounding box: {:7.2f} <= lon [deg] <= {:7.2f}, {:7.2f} <= lat [deg] <= {:7.2f}'.format(
                    fn_grid, grid.variables['clon'].size, grid.variables['vlon'].size, clon.min()*180/np.pi, clon.max()*180/np.pi, clat.min()*180/np.pi, clat.max()*180/np.pi) )

            self.grid[fn_grid] = grid
            
        return self.grid[fn_grid]


    #-------------------------------------------------------------------------------
    @staticmethod
    def suffix_to_int( sfx ) :
        """Convert suffices like member index, 'mean', 'spred' and 'det' to indices"""
        try :
            i = int(sfx)
        except :
            if sfx == 'det' :
                i = 0
            elif sfx == 'mean' :
                i = -1
            elif sfx == 'spread' :
                i = -2
            else :
                i = -999 # unkown
        return i

    #-------------------------------------------------------------------------------
    @staticmethod
    def find_files( exp_path, config=None, verbose=False ) :
        """Find files in data and feedback directories and store their paths in a
        easy to use tree structure"""

        if config is None :
            config = BacyExp.get_config( exp_path )

        datadir     = os.path.join( exp_path, 'data' )
        feedbackdir = os.path.join( exp_path, 'feedback' )

        # EXAMPLES
        # first level:       20190629120000
        # 
        # fg          mem    fc_R19B07.20190629140000.017
        # fg          mean   fc_R19B07.20190629130000.mean
        # fg          spread fc_R19B07.20190629130000.spread
        # fg_latlon   mem    fc_R19B07_ll.20190629140000.017
        # ana         mem    an_R19B07.20190629120000.001
        # ana         mean   an_R19B07.20190629140000.mean
        # ana         spread an_R19B07.20190629140000.spread
        # ana_inc     mem    an_R19B07.20190629120000_inc.001
        # ana_inc     det    an_R19B07.20190629140000_inc
        # fof         det    fofAIREP_20190629130000.nc
        # fof         mem    fofAIREP_20190629130000_ens021.nc
        # satimg      mem    seviri_sim_201907021800_ens023.grb2

        grid_ens = BacyExp.res_to_grid(config['BA_IL_IRES_ENS'])
        grid_det = BacyExp.res_to_grid(config['BA_IL_IRES_DET'])

        filetree = {}
        obstypes = set()
        filetypes = set()    
        fc_start_times = set()
        fc_lead_times = set()
        valid_times = {}

        re_datetime14 = re.compile('^[0-9]{14}$')

        # search for files in data directory ......................................

        re_files = {
            'an'      : re.compile('^an_(?P<grid>R..B..)\.(?P<time>[0-9]+)(\.(?P<suffix>.+))?$'),
            'an_inc'  : re.compile('^an_(?P<grid>R..B..)\.(?P<time>[0-9]+)_inc(\.(?P<suffix>.+))?$'),
            'fg'      : re.compile('^fc_(?P<grid>R..B..)\.(?P<time>[0-9]+)(\.(?P<suffix>.+))?$'),
            'fg_ll'   : re.compile('^fc_(?P<grid>R..B..)_ll\.(?P<time>[0-9]+)(\.(?P<suffix>.+))?$'),
            'fof'     : re.compile('^fof(?P<obstype>[A-Z].+)_(?P<time>[0-9]+)(_ens(?P<suffix>[0-9][0-9][0-9]))?\.nc$'),
            'seviri'  : re.compile('^seviri_(?P<simobs>[a-z]+)_(?P<time>[0-9]+)(_ens(?P<suffix>[0-9][0-9][0-9]))?\.grb2$')
        }
        re_idx = re.compile('^[0-9]+$')

        re_mainfiles = {
            'fc'    : re.compile('^fc_(?P<grid>R..B..)\.(?P<starttime>[0-9]+)_(?P<leadtime>[0-9]+)hr(\.(?P<suffix>.+))?$'),
            'fc_ll' : re.compile('^fc_(?P<grid>R..B..)_ll\.(?P<starttime>[0-9]+)_(?P<leadtime>[0-9]+)hr(\.(?P<suffix>.+))?$')
        }
        # fc_R19B07.20190629180000_0hr
        # fc_R19B07_ll.20190629180000_0hr


        for q in re_files :
            valid_times[q] = set()

        cwd = os.getcwd()
        os.chdir( datadir )    
        for (dirpath, dirnames, filenames) in os.walk( '.', topdown=True, onerror=None, followlinks=False ) :        
            for f_ in filenames :
                f = os.path.join(dirpath,f_)[2:]
                tkns = f.split('/')
                lvl = len(tkns) - 1
                if lvl > 0 : 
                    rootdir = tkns[0]
                                        
                    if lvl == 1 and re_datetime14.match(tkns[0]) : # cycling results inside root level directories like "20190703230000"

                        for q in re_files :
                            m = re_files[q].match(tkns[1])
                            if m :
                                filetypes.add(q)

                                # no member index means deterministic member
                                sfx = 'det' if m.groupdict()['suffix'] is None else m.groupdict()['suffix']

                                # grid name should match info from config
                                if 'grid' in m.groupdict() :
                                    if (sfx == 'det' and m.groupdict()['grid'] != grid_det) or \
                                    (sfx != 'det' and m.groupdict()['grid'] != grid_ens) :
                                        print('WARNING: grid mismatch ', tkns[1], m.groupdict()['grid'], grid_ens, grid_det)

                                tim = m.groupdict()['time']
                                if len(tim) == 12 :
                                    tim += '00'

                                # check if time can be divided by assimilation interval
                                dt0  = str2t(tim) - str2t(tim,hour=0,minute=0,second=0)
                                ncyc = dt0.seconds / float(config['BA_DELT_ASS'])
                                if fabs(ncyc-int(ncyc)) > 1e-3 : # IAU file -- ignore...
                                    continue

                                # remember that we found a file of type q at time tim
                                valid_times[q].add(tim)

                                ft_path = [ 'cycles', tim, q ]

                                if q == 'fof' :
                                    ot = m.groupdict()['obstype'].replace('_vis','RAD')
                                    ft_path.append(ot)
                                    obstypes.add(ot)
                                elif q == 'seviri' :
                                    ft_path.append(m.groupdict()['simobs'])

                                if re_idx.match( sfx ) : # member index
                                    add_branch( filetree, ft_path + [ 'mem', int(sfx) ], f )
                                else :
                                    add_branch( filetree, ft_path + [ sfx ], f )

                                continue

                    elif lvl == 2 and re_datetime14.match(tkns[0]) and tkns[1].startswith('main') : # main forecast results in directories like "20190703230000/main0000"

                        for q in re_mainfiles :
                            m = re_mainfiles[q].match(tkns[2])
                            if m :
                                filetypes.add(q)

                                # no member index means deterministic member
                                sfx = 'det' if m.groupdict()['suffix'] is None else m.groupdict()['suffix']

                                # grid name should match info from config
                                if 'grid' in m.groupdict() :
                                    if (sfx == 'det' and m.groupdict()['grid'] != grid_det) or \
                                    (sfx != 'det' and m.groupdict()['grid'] != grid_ens) :
                                        print('WARNING: grid mismatch ', tkns[2], m.groupdict()['grid'], grid_ens, grid_det)

                                # remember start and lead times
                                starttime = m.groupdict()['starttime']
                                leadtime  = m.groupdict()['leadtime']                                
                                fc_start_times.add(starttime)
                                fc_lead_times.add(leadtime)

                                # add branch to file tree
                                
                                ft_path = [ 'forecast', starttime, leadtime, q ]

                                if re_idx.match( sfx ) : # member index
                                    add_branch( filetree, ft_path + [ 'mem', int(sfx) ], f )
                                else :
                                    add_branch( filetree, ft_path + [ sfx ], f )

                                continue


        # search for files in feedback directory ..................................

        re_files = {
            'ekf'    : re.compile('^ekf(?P<obstype>[A-Z]+)\.nc\.(?P<time>[0-9]+)$'),
            'mon'    : re.compile('^mon(?P<montype>[A-Z]+)\.nc\.(?P<time>[0-9]+)$')
        }
        for q in re_files :
            valid_times[q] = set()
        
        os.chdir( feedbackdir )
        for (dirpath, dirnames, filenames) in os.walk( '.', topdown=True, onerror=None, followlinks=False ) :        
            for f_ in filenames :
                f = os.path.join(dirpath,f_)[2:]

                tkns = f.split('/')
                lvl = len(tkns) - 1
                if lvl > 0 : 
                    rootdir = tkns[0]
                    
                    if lvl == 1 and re_datetime14.match(tkns[0]) : # cycling results inside root level directories like "20190703230000"

                        for q in re_files :
                            m = re_files[q].match(tkns[1])
                            if m :
                                filetypes.add(q)

                                tim = m.groupdict()['time']
                                if len(tim) == 12 :
                                    tim += '00'

                                # check if time can be divided by assimilation interval
                                dt0  = str2t(tim) - str2t(tim,hour=0,minute=0,second=0)
                                ncyc = dt0.seconds / float(config['BA_DELT_ASS'])
                                if fabs(ncyc-int(ncyc)) > 1e-3 : # IAU file -- ignore...
                                    continue

                                # remember that we found a file of type q at time tim
                                valid_times[q].add(tim)

                                ft_path = [ 'cycles', tim, q ]

                                if q == 'ekf' :
                                    ft_path.append(m.groupdict()['obstype'])
                                    obstypes.add(m.groupdict()['obstype'])
                                elif q == 'mon' :
                                    ft_path.append(m.groupdict()['montype'])

                                add_branch( filetree, ft_path, f )
        os.chdir( cwd )

        if verbose :
            for start in sorted(filetree['cycles']) :
                print(start, end= ' : ')
                for q in filetypes :
                    if q in filetree['cycles'][start] :
                        print( q, sorted(filetree['cycles'][start][q].keys()), end='  ' )
                print()

        # convert to sorted lists
        valid_times_ = {}
        for q in valid_times :
            valid_times_[q] = sorted(list(valid_times[q]))

        return { 'filetree':filetree, 'obstypes':obstypes, 'filetypes':filetypes, 'valid_times':valid_times_,
                 'fc_start_times':sorted(list(fc_start_times)), 'fc_lead_times':sorted([int(t) for t in fc_lead_times]) }

    
    #-------------------------------------------------------------------------------
    def info( self ) :

        if len(self.fc_start_times) > 0 :
            if len(self.fc_start_times) > 2 :
                fc_str = 'forecasts started at {}, {}, ... {}'.format( self.fc_start_times[0], self.fc_start_times[1], self.fc_start_times[-1] )
            else :
                fc_str = 'forecasts started at ' + (','.join( self.fc_start_times ))
        else :
            fc_str = 'no forecasts'
        print('directory = {}, experiment id = {}, model = {}, '.format(
            self.exp_dir, self.config['CY_EXPID'], self.config['model']), end='')
        if 'an' in self.valid_times and len(self.valid_times['an']) > 1 :
            print('analyses at {}, {}, ... {}, {}'.format(self.valid_times['an'][0], self.valid_times['an'][1], self.valid_times['an'][-1], fc_str ))
        else :
            print('no analyses, {}'.format(fc_str))
        print('observation types:', ' '.join(self.obs_types))


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    import argparse

    # parse command line arguments
    parser = define_parser()
    args = parser.parse_args()

    # provide basic info
    for xp_path in args.experiments :
        xp = BacyExp( xp_path, read_cache=args.cache, verbose=args.verbose )
        xp.info()

    #print( xp.get_filename('grid') )
    #print('VISOP FILENAME ', xp.get_filename( 'visop', time_start=xp.fc_start_times[-1], lead_time=10 ))

    if False :
        for xp_path in args.experiments :
            xp = BacyExp(xp_path)
            print( xp.get_filename( 'fc', time_start='20190703220000', lead_time=2, latlon=True, mem=2 ) )
            # -> .../data/20190703210000/main0100/fc_R19B07_ll.20190703220000_2hr.002

    if args.config : # dump all settings from modules/*/conf/*
        for xp_path in args.experiments :
            print()
            print('configuration of experiment in ', xp_path, '-'*30)
            config = BacyExp.get_config( xp_path )
            for p in sorted(config) :
                print( p, config[p] )

    if args.plot_files : # generate file overview plot

        from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
        from matplotlib.lines import Line2D

        for xp_path in args.experiments :
            xp = BacyExp(xp_path)
            ft = xp.file_tree
            
            cycle_times = sorted(ft['cycles'].keys())
            cycle_quans = set()
            for tc in cycle_times :
                cycle_quans.update( ft['cycles'][tc].keys() )
            cycle_quans = sorted(list(cycle_quans))

            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            
            fig, ax = plt.subplots(figsize=(20,10))

            cycle_dates = [ str2t(t) for t in cycle_times ]
            for it, t in enumerate(cycle_times) :
                d = cycle_dates[it] # datetime object
                for iq, q in enumerate(cycle_quans) :
                    if q in ft['cycles'][t] :
                        for tp in ft['cycles'][t][q].keys() :
                            if tp == 'det' :
                                ax.plot_date( d, iq, c='r', marker='.' )
                            elif  tp == 'mem' :
                                ax.plot_date( d, iq-0.1, c='b', marker='.' )
                            elif  tp == 'mean' :
                                ax.plot_date( d, iq-0.2, c='g', marker='.' )
                            elif  tp == 'spread' :
                                ax.plot_date( d, iq-0.3, c='#cc00cc', marker='.' )
                            else :
                                ax.plot_date( d, iq, c='#999999', marker='.' )
            
            if 'forecast' in ft :

                fcst_times = sorted(ft['forecast'].keys())
                print('FCST times ', fcst_times)
               
                fcst_dates = [ str2t(t) for t in fcst_times ]
                for it, t in enumerate(fcst_times) :
                    d = fcst_dates[it] # datetime object

                    for ldt in ft['forecast'][t] :
                        for iq, q in enumerate(ft['forecast'][t][ldt]) :
                            if q in ['fc','fc_ll'] :
                                p = -1 if  q == 'fc' else -2
                                if 'mem' in ft['forecast'][t][ldt][q] :
                                    nmem = len(ft['forecast'][t][ldt][q].keys())
                                    if nmem == xp.n_ens :
                                        ax.plot_date( d, p - 0.9*float(ldt)/24, c='b', marker='x' )
                                    else :
                                        ax.plot_date( d, p - 0.9*float(ldt)/24, c='#666666', marker='x' )
                                else :
                                    ax.plot_date( d, p - 0.9*float(ldt)/24, c='r', marker='x' )

            ax.set_xlim( cycle_dates[0], cycle_dates[-1] )
            adjust_time_axis( fig, ax, cycle_dates )

            #ax.set_ylim((-2,len(cycle_quans)))
            ax.set_yticks([-2,-1] + [i for i in range(len(cycle_quans))])
            ax.set_yticklabels(['fc_ll','fc'] + cycle_quans)
           
            # add legend
            legend_elements = [ Line2D( [0], [0], marker='.', linestyle='', color='r', label='det'),
                                Line2D( [0], [0], marker='.', linestyle='', color='b', label='ens'),
                                Line2D( [0], [0], marker='.', linestyle='', color='g', label='mean'),
                                Line2D( [0], [0], marker='.', linestyle='', color='#cc00cc', label='spread') ]
            ax.legend( handles=legend_elements, loc='lower center', ncol=len(legend_elements) )

            ax.grid()
            fig.savefig( xp_path.split('/')[-1]+'_files.pdf', bbox_inches='tight')
