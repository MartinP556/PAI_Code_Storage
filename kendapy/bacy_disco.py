#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ D I S C O
#
#  2021.5 L.Scheck 

import os, sys, subprocess, tempfile, re, pickle, copy, subprocess
from datetime import datetime, timedelta
from math import ceil

from kendapy.bacy_utils import to_datetime, to_timedelta, midnight, t2str, td2str
from kendapy.bacy_exp   import BacyExp

#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="DISC-space Optimized cycling script")
    parser.add_argument( '-s', '--start-time',       dest='start_time', default=None,  help='cycling start time' )
    parser.add_argument( '-e', '--end-time',         dest='end_time'  , default=None,  help='cycling end   time' )
    parser.add_argument( '-V', '--visop-main',       dest='visop_main',      action='store_true',  help='process main forecasts using visop' )
    parser.add_argument(       '--rerun-visop-main', dest='rerun_visop_main', action='store_true',  help='rerun visop for existing main forecasts' )
    parser.add_argument(       '--skip-final-main',  dest='skip_final_main', action='store_true',  help='do not start forecast at end time' )
    parser.add_argument(       '--no-cycling',       dest='no_cycling',      action='store_true',  help='skip cycling, assume cycling has already been done' )
    parser.add_argument(       '--no-veri',          dest='no_veri',         action='store_true',  help='skip veri step' )
    parser.add_argument(       '--add-main',         dest='add_main',        action='store_true',  help='only add missing main runs, do not rerun existing main runs' )
    parser.add_argument( '-D', '--delete',           dest='delete',          default='',           help='comma seperated list of things delete [latlon|ens<HH>]' )
    parser.add_argument( '-v', '--verbose',          dest='verbose',         action='store_true',  help='be more verbose' )
    parser.add_argument( '-d', '--dry',              dest='dry',             action='store_true',  help='dry run -- just show which commands would be executed' )
    parser.add_argument( 'experiment', metavar='<experiment path>', help='path to experiment' )

    return parser

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # parse command line arguments
    import argparse
    parser = define_parser()
    args = parser.parse_args()

    # read experiment configuration
    xp = BacyExp(args.experiment)

    # define times and intervals
    start_time = to_datetime(args.start_time)
    end_time   = to_datetime(args.end_time)
    dt_main    = to_timedelta( xp.config['BA_DELT_MAIN_ICONLAM'], units='s' )
    dt_cycle   = to_timedelta( xp.config['BA_DELT_ASS_ICONLAM'], units='s' )
    dt_mainlen = to_timedelta( xp.config['BA_FCLNG_DET'], units='h' )

    #print(dt_mainlen)
    #sys.exit(0)

    #   M A I N   L O O P  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    os.chdir( args.experiment+'/modules/cycle')
    current_time = copy.deepcopy( start_time )
    while( current_time <= end_time ) :

        # determine next main fcst time
        next_main_time = midnight(current_time) + ceil( (current_time - midnight(current_time)) / dt_main ) * dt_main

        # C Y C L E . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        if current_time < next_main_time :
            if args.no_cycling :
                print('- NOT cycling from {} to {}... '.format( current_time, next_main_time ))
            else :
                print('- cycling from {} to {}... '.format( current_time, next_main_time ))
                cmd = './cycle ICON-LAM ASS {} {} &> disco_cycle_{}_{}.log'.format( t2str(current_time), t2str(next_main_time), t2str(current_time), t2str(next_main_time) )
                print('  --> running ', cmd)
                if not args.dry :
                    if os.system(cmd) != 0 :
                        raise RuntimeError('cycle failed!')
            current_time = next_main_time

        # M A I N . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        if current_time < end_time or not args.skip_final_main :

            run_main = True
            if args.add_main :
                if os.path.exists( args.experiment + '/data/' + t2str(current_time) + '/main0000' ) :
                    print('- main forecast at ', current_time, ' exists already...')
                    run_main = False

            if run_main :
                print('- starting main forecast at ', current_time)
                cmd = './cycle ICON-LAM MAIN {} {} &> disco_main_{}.log'.format( t2str(current_time), t2str(current_time), t2str(current_time) )
                print('  --> running ', cmd)
                if not args.dry :
                    if os.system(cmd) != 0 :
                        raise RuntimeError('main failed!')

            # V I S O P / M A I N . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if args.visop_main and (run_main or args.rerun_visop_main) :
                print('  processing main forecast with visop...')
                cmd = """. /hpc/uwork/lscheck/visop_env/activate -p; cd __EXPERIMENT__/visop; \
                      visop2_multi.py -F __EXPERIMENT__/modules/mec/const/visop2_argfile_main \
                      -s __DATE__ -e __DATE__ -i __DTMAIN__ -a 0:00 -z __DTMAINLEN__ -S __HSTART__:00 -E __HEND__:00 \
                      -O __EXPERIMENT__/visop -I __EXPERIMENT__ -o __EXPERIMENT__/visop/obs \
                      -Q -w -c --nprocs __NPROCS__ &> __EXPERIMENT__/modules/cycle/disco_visop_main___DATE__.log""".replace(
                      '__EXPERIMENT__', args.experiment                 ).replace(
                      '__DATE__',       t2str(current_time)             ).replace(
                      '__DTMAIN__',     td2str(dt_main)                 ).replace(
                      '__DTMAINLEN__',  td2str(dt_mainlen)              ).replace(
                      '__NPROCS__',     td2str(dt_mainlen)[:2]          ).replace(
                      '__HSTART__',     xp.config['BA_SEVIRIVIS_START'] ).replace(
                      '__HEND__',       xp.config['BA_SEVIRIVIS_END']   )
                print('  --> running visop for main forecast started at '+t2str(current_time)+'...')
                print('-'*80)
                print(cmd)
                print('-'*80)
                if not args.dry :
                    if os.system(cmd) != 0 :
                        raise RuntimeError('visop_main failed!')

        # V E R I . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        prev_main_time = current_time - dt_main
        if not args.no_veri :
            print('- starting veri for period {} - {}'.format(prev_main_time, current_time))
            cmd = './cycle ICON-LAM VERI {} {} &> disco_veri_{}_{}.log'.format( t2str(prev_main_time), t2str(current_time), t2str(prev_main_time), t2str(current_time) )
            print('  --> running ', cmd)
            if not args.dry :
                if os.system(cmd) != 0 :
                    raise RuntimeError('veri failed!')

        # D E L E T E   U N W A N T E D   F I L E S  . . . . . . . . . . . . . . . . . . . . . . .
        if args.delete != '' :
            #sp = subprocess.Popen('du -hs '+args.experiment, shell=True, stdout=subprocess.PIPE)
            #print('=== experiment size before deleting files: ', sp.stdout.read() )            

            if 'latlon' in args.delete :
                prev_time = current_time - dt_cycle
                while prev_time >= prev_main_time :
                    del_path = args.experiment+'/data/'+t2str(prev_time)
                    if os.path.exists(del_path) :
                        print('- deleting latlon files in {} ...'.format(del_path))
                        if args.dry :
                            cmd = 'find PATH -name "fc*ll*" -exec echo deleting {} \;'.replace('PATH',del_path)
                        else :
                            cmd = 'find PATH -name "fc*ll*" -exec rm {} \;'.replace('PATH',del_path)
                        if os.system(cmd) != 0 :
                            raise RuntimeError('deleting latlon files failed!')
                    prev_time -= dt_cycle

            if 'ensemble' in args.delete :                
                # remove first guess and analysis ensmbles, i.e. files like those:
                # 519M an_R19B07.20190717180000.013
                # 432M an_R19B07.20190717180000_inc.013
                # 1.1G fc_R19B07.20190717175500.013
                # 222M fc_R19B07_tiles.20190717175500.013
                # /hpc/uhome/lscheck/work/exp_lm/ILAM_ONLINE_SEVIRI/2019_July/ICOND2_SEVIRI_1203/data/20190717180000/[af][nc]_R*.20*.???
                prev_time = prev_main_time
                prev_main_time2 = prev_main_time - dt_main
                print('- checking for ensemble files to be deleted at times between ', prev_time, ' and ', prev_main_time2)
                while prev_time >= prev_main_time2 :                    
                    if prev_time > start_time :
                        del_path = args.experiment+'/data/'+t2str(prev_time)
                        if os.path.exists(del_path) :
                            print('- deleting ensemble files in {} ...'.format(del_path))
                            cmd = 'rm {}/[af][nc]_R*.20*.???'.format(del_path)
                            print('  --> running ', cmd)
                            if not args.dry :
                                if os.system(cmd) != 0:
                                    print('      something went wrong -- continuing anyway...')
                        else :
                            print('- would delete ensemble files in {}, but there are none...'.format(del_path))
                    else :
                        print('- not deleting files for ', prev_time, ' because that is <= the start time.')
                    prev_time -= dt_cycle

            #sp = subprocess.Popen('du -hs '+args.experiment, shell=True, stdout=subprocess.PIPE)
            #print('=== experiment size after deleting files: ',  sp.stdout.read() )

        current_time = next_main_time + dt_cycle

