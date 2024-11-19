#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ O B S
#
#  2020.6 L.Scheck 

from __future__ import absolute_import, division, print_function

import os, sys, subprocess, glob, tempfile, re, argparse
from datetime import datetime, timedelta
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from kendapy.ekf        import Ekf
from kendapy.bacy_utils import t2str, str2t, add_branch, common_subset, default_color_sequence, expand_color_name, adjust_time_axis
from kendapy.bacy_exp   import BacyExp


#----------------------------------------------------------------------------------------------------------------------
def define_parser() :

    parser = argparse.ArgumentParser(description="Generate basic observation space plots for bacy experiments")

    # plot type and parameters
    parser.add_argument( '-E', '--evo',        dest='evo',            action='store_true',       help='plot error evolution in observation space' )
    parser.add_argument(       '--sawtooth',   dest='sawtooth',       action='store_true',       help='use sawtooth style for error evolution plots' )
    parser.add_argument(       '--fg-only',    dest='fg_only',        action='store_true',       help='plot only first guess errors' )
    parser.add_argument(       '--an-only',    dest='an_only',        action='store_true',       help='plot only analysis    errors' )

    parser.add_argument( '-F', '--filter',     dest='filter',         default='state=active',    help='filter string (default: "state=active")' )
    parser.add_argument( '-O', '--obstypes',   dest='obstypes',       default=None,              help='comma-separated list of observation types, e.g. SYNOP,TEMP,RAD' )
    parser.add_argument( '-V', '--varnames',   dest='varnames',       default=None,              help='comma-separated list of variable names, e.g. U,V,T' )
    parser.add_argument( '-m', '--metrics',    dest='metrics',        default='rmse,bias,spread,nobs', help='comma-separated list of metrics to be plotted [rmse,bias,spread,nobs] (default:all)' )

    parser.add_argument( '-C', '--compare',    dest='compare',        action='store_true',       help='generate comparison plot instead of individual plots for all experiments' )
    parser.add_argument(       '--separate-sets', dest='separate_sets', action='store_true',     help='do not use a common subset of observations, but separate sets' )

    parser.add_argument(       '--time-range', dest='time_range',     default=None,              help='time range, e.g. 20190629170000,20190630120000' )
    parser.add_argument(       '--yrange',     dest='yrange',         default=None,              help='y-axis range <y-min>,<y-max>' )

    # output options
    parser.add_argument( '-o', '--output-path', dest='output_path',        default=None,              help='output path' )
    parser.add_argument( '-i', '--image-type',  dest='image_type',         default='png',             help='[ png | eps | pdf ... ]' )
    parser.add_argument(       '--dpi',         dest='dpi',                default=100, type=int,     help='dots per inch for pixel graphics (default: 100)' )
    parser.add_argument(       '--figsize',     dest='figsize',            default='5,4',             help='<figure width>,<figure height> [inch]' )
    parser.add_argument(       '--colors',      dest='colors',             default=None,              help='comma-separated list of colors (e.g. "r,#ff0000,pink")' )
    parser.add_argument(       '--legloc',      dest='legloc',             default='best',            help='location of legend (default="best", see matplotlib documentation, "outside"=to the right of plot window)' )

    parser.add_argument( '-v', '--verbose',     dest='verbose', action='store_true',  help='be extremely verbose' )

    parser.add_argument( 'experiments', metavar='<experiment path(s)>', help='path(s) to experiment(s)', nargs='*' )

    return parser


#----------------------------------------------------------------------------------------------------------------------
def plot_obs_evo( experiments, vargs, verbose=True ) :

    # convert input to list of BacyExp structures, if necessary
    if type(experiments) == list :
        xps = []
        for xp_ in experiments :
            if type(xp_) == BacyExp :
                xps.append(xp_)
            else :
                xps.append( BacyExp(xp_) )# created from path
        n_exps = len(xps)
    else :
        if type(experiments) == BacyExp :
            xp = experiments
        else :
            xp = BacyExp(experiments) # created from path
        xps = [xp]
        n_exps = 1

    # determine time range
    valid_times = common_subset([ xp.valid_times['ekf'] for xp in xps ])
    if not vargs['time_range'] is None :
        t0, t1 = vargs['time_range'].split(',')
        valid_times = [ t for t in valid_times if ((str2t(t) >= str2t(t0)) and (str2t(t) <= str2t(t1))) ]

    obs_types   = common_subset([ xp.obs_types         for xp in xps ])

    if verbose :
        print('  [plot_obs_evo] valid times           = ', valid_times )
        print('                 available obs. types  = ', obs_types )

    metrics = vargs['metrics'].split(',')
    if verbose :
        print('                 metrics               = ', ' '.join(metrics) )


    # plotting style

    mark_problems = True # if n_exps == 1 else False
    alpha_connector = 0.7
    colseq = default_color_sequence() if vargs['colors'] is None else [ expand_color_name(s) for s in vargs['colors'].split(',') ]
    col_met = { 'rmse':colseq[0], 'bias':colseq[1], 'spread':colseq[2], 'eo':colseq[3], 'mean':colseq[4], 'obsmean':colseq[5], 'spread_plus_eo':colseq[6], 'nobs':'#999999' }
    styleseq = ['-','--',':','_.']
    widthseq = [ 1, 2, 0.5, 3, 0.25 ]

    # decide how color [c], linestyle [s] and linewidth [w]
    # should vary with metric [m], experiment [e] and fg/an [f]
    # ([x] = no variation)
    if n_exps == 1 :
        if len(metrics) == 1 :
            vary = { 's':'m', 'w':'e', 'c':'f' }
        else :
            vary = { 'c':'m', 'w':'e', 's':'f' }
    else :
        if len(metrics) == 1 :
            vary = { 'w':'m', 'c':'e', 's':'f' }
        else :
            if vargs['sawtooth'] :
                vary = { 'c':'m', 's':'e', 'w':'f' }
            else :
                vary = { 'c':'m', 's':'e', 'w':'f' }

    stl_c, stl_s, stl_w = {}, {}, {}
    for ixp in range(n_exps) :
        stl_c[ixp], stl_s[ixp], stl_w[ixp] = {}, {}, {}
        for imet, met in enumerate(metrics) :
            stl_c[ixp][met], stl_s[ixp][met], stl_w[ixp][met] = {}, {}, {}
            for ifgan in [0] if vargs['sawtooth'] else [0,1] :
                # color
                if vary['c'] == 'e' :
                    stl_c[ixp][met][ifgan] = colseq[ixp]
                elif vary['c'] == 'm' :
                    stl_c[ixp][met][ifgan] = col_met[met]
                elif vary['c'] == 'f' :
                    stl_c[ixp][met][ifgan] = colseq[ifgan]
                else :
                    stl_c[ixp][met][ifgan] = 'k'
                # line style
                if vary['s'] == 'e' :
                    stl_s[ixp][met][ifgan] = styleseq[ixp]
                elif vary['s'] == 'm' :
                    stl_s[ixp][met][ifgan] = styleseq[imet]
                elif vary['s'] == 'f' :
                    stl_s[ixp][met][ifgan] = styleseq[ifgan]
                else :
                    stl_s[ixp][met][ifgan] = '-'
                # line width
                if vary['w'] == 'e' :
                    stl_w[ixp][met][ifgan] = widthseq[ixp]
                elif vary['w'] == 'm' :
                    stl_w[ixp][met][ifgan] = widthseq[imet]
                elif vary['w'] == 'f' :
                    stl_w[ixp][met][ifgan] = widthseq[ifgan]
                else :
                    stl_w[ixp][met][ifgan] = 1


    # loop over observation types
    for obs_type in obs_types if vargs['obstypes'] is None else vargs['obstypes'].split(',') :
        print('processing observation type {}...'.format(obs_type))
        varnames = []
        nobs = {}
        fa = {}
        it_ekf = -100
        ekfs = None

        # loop over verification times
        pdt = None
        for it, t in enumerate(valid_times) :
            dt = str2t(t) # datetime object
            if verbose :
                print('  - ', t, dt)

            # open ekf files
            prev_ekfs = ekfs
            prev_it_ekf = it_ekf
            ekfs = []
            ekf_filter = []
            complete = True
            for ixp, xp in enumerate(xps) :

                ekf_filename = xp.get_filename('ekf', time_valid=t, obs_type=obs_type)
                if os.path.exists(ekf_filename) :

                    ef = vargs['filter']
                    if 'active_first' in ef :
                        if ixp == 0 :
                            print('>>> USING ACTIVE OBS FROM FIRST EXPERIMENT <<<')
                            ef = ef.replace('active_first','active')
                        else :
                            print('>>> USING ALL OBS FROM EXPERIMENT #', ixp+1, '<<<')
                            ef = ef.replace('active_first','all')

                    ekf_filter.append( ef )
                    ekfs.append( Ekf(ekf_filename,filter=ef) )
                else :
                    complete = False
                    break

            if complete : # ekf file is present for every experiment
                it_ekf = it

                # find new common variables (may not be present for all times)
                for v in common_subset([ ekfs[ixp].all_varnames for ixp in range(n_exps) ]) :
                    if vargs['varnames'] is None or v in vargs['varnames'].split(',') :
                        if not v in varnames :
                            varnames.append(v)

                # loop over common variables
                for iv, v in enumerate(varnames) :

                    print('      {:10s} : '.format(v), end='')

                    # select variable
                    for iekf, ekf in enumerate(ekfs) :
                        #ekf.add_filter(filter='varname='+v)
                        ekf.replace_filter(filter=ekf_filter[iekf]+' varname='+v)

                    # select common subset of observations in each ekf object
                    if (not vargs['separate_sets']) and (len(ekfs) > 1) :
                        Ekf.filter_common( ekfs, filtername='common_'+v )
                    else :
                        # rename current filter
                        ekfs[0].filtered_idcs['common_'+v] = ekfs[0].filtered_idcs[ekfs[0].filtername]

                    if not v in fa :
                        if 'nobs' in metrics :
                            fig, ax = plt.subplots( 2, gridspec_kw={'height_ratios':[3,1],'hspace':0.03}, sharex=True, figsize=(15,4.5))
                        else :
                            fig, ax_ = plt.subplots(figsize=(15,3))
                            ax = [ax_]
                        fa[v] = {'fig':fig,'ax':ax}                                
                    else : 
                        fig, ax = fa[v]['fig'], fa[v]['ax']

                    n = ekfs[0].n_obs()
                    if n > 0 :
                        if v in nobs :
                            nobs[v] += n
                        else :
                            nobs[v] = n

                        for iekf, ekf in enumerate(ekfs) :
                            
                            # plot vertical segments (fg and ana valid at same time)

                            for met in metrics :
                                stl = { 'marker':'',
                                        'linestyle':stl_s[iekf][met][0],
                                        'color':    stl_c[iekf][met][0],
                                        'linewidth':stl_w[iekf][met][0] }

                                if met == 'rmse' :
                                    fge = np.sqrt( ((ekf.fgmean() - ekf.obs())**2).mean() )
                                    ane = np.sqrt( ((ekf.anamean() - ekf.obs())**2).mean() )
                                    if vargs['sawtooth'] :
                                        if mark_problems and fge < ane :
                                            stl['color'] = 'r'
                                        ax[0].plot_date( (dt,dt), (fge,ane), **stl )

                                elif met == 'bias' :
                                    fgb = ( ekf.fgmean()  - ekf.obs() ).mean() 
                                    anb = ( ekf.anamean() - ekf.obs() ).mean()
                                    if vargs['sawtooth'] :
                                        if mark_problems and np.abs(fgb) < np.abs(anb) :
                                            stl['color'] = 'r'
                                        ax[0].plot_date( (dt,dt), (fgb,anb), **stl )
        
                                elif met == 'spread' :
                                    fgs = ekf.fgspread().mean()
                                    ans = ekf.anaspread().mean()
                                    if vargs['sawtooth'] :
                                        if mark_problems and np.abs(fgs) < np.abs(ans) :
                                            stl['color'] = 'r'
                                        ax[0].plot_date( (dt,dt), (fgs,ans), **stl )

                                elif met == 'spread_plus_eo' :
                                    fgse = np.sqrt( ( ekf.fgspread()**2  + ekf.obs(param='e_o')**2 ).mean() )
                                    anse = np.sqrt( ( ekf.anaspread()**2 + ekf.obs(param='e_o')**2 ).mean() )
                                    if vargs['sawtooth'] :
                                        ax[0].plot_date( (dt,dt), (fgse,anse), **stl )

                                elif met == 'mean' :
                                    fgm = ekf.fgmean().mean()
                                    anm = ekf.anamean().mean()
                                    if vargs['sawtooth'] :
                                        ax[0].plot_date( (dt,dt), (fgm,anm), **stl )

                            # if we have data for previous verification time, add slant segments (ana from previous, ana from current time)
                            if not (prev_ekfs is None) and (prev_it_ekf == it - 1) :
                                prev_ekf = prev_ekfs[iekf]
                                #print('EXISTING FILTERS: ', iekf, prev_ekf.filtered_idcs.keys())
                                if 'common_'+v in prev_ekf.filtered_idcs :
                                    prev_ekf.activate_exsisting_filter( 'common_'+v )

                                    if 'nobs' in metrics :
                                        ax[1].plot_date( (pdt,dt), (prev_ekf.n_obs(),n), marker='', linestyle='-', color = col_met['nobs'] )

                                    if prev_ekf.n_obs() > 0 :
                                        for met in metrics :

                                            stl = { 'marker':'', 'linestyle':stl_s[iekf][met][0], 'color':stl_c[iekf][met][0], 'linewidth':stl_w[iekf][met][0] }
                                            if not vargs['sawtooth'] :
                                                if vargs['fg_only'] or vargs['an_only'] :
                                                    stl1 = stl
                                                else :
                                                    stl1 = { 'marker':'', 'linestyle':stl_s[iekf][met][1], 'color':stl_c[iekf][met][1], 'linewidth':stl_w[iekf][met][1] }

                                            if met == 'rmse'  : # connect previous an with current fg
                                                prev_ane = np.sqrt( ((prev_ekf.anamean() - prev_ekf.obs())**2).mean() )
                                                if vargs['sawtooth'] :
                                                    ax[0].plot_date( (pdt,dt), (prev_ane,fge), alpha=alpha_connector, **stl )
                                                else :             # connect previous an, fg with current an, fg
                                                    if not vargs['fg_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_ane,ane), **stl )
                                                    prev_fge = np.sqrt( ((prev_ekf.fgmean() - prev_ekf.obs())**2).mean() )
                                                    if not vargs['an_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_fge,fge), **stl1 )

                                            if met == 'bias'  :
                                                prev_anb = ( prev_ekf.anamean() - prev_ekf.obs() ).mean()
                                                if vargs['sawtooth'] :
                                                    ax[0].plot_date( (pdt,dt), (prev_anb,fgb), alpha=alpha_connector, **stl )
                                                else :
                                                    if not vargs['fg_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_anb,anb), alpha=alpha_connector, **stl )
                                                    prev_fgb = ( prev_ekf.fgmean() - prev_ekf.obs() ).mean()
                                                    if not vargs['an_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_fgb,fgb), alpha=alpha_connector, **stl1 )

                                            if met == 'spread'  :
                                                prev_ans = prev_ekf.anaspread().mean()
                                                if vargs['sawtooth'] :
                                                    ax[0].plot_date( (pdt,dt), (prev_ans,fgs), alpha=alpha_connector, **stl )
                                                else :
                                                    if not vargs['fg_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_ans,ans), alpha=alpha_connector, **stl )
                                                    prev_fgs = prev_ekf.fgspread().mean()
                                                    if not vargs['an_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_fgs,fgs), alpha=alpha_connector, **stl1 )

                                            if met == 'mean'  :
                                                prev_anm = prev_ekf.anamean().mean()
                                                if vargs['sawtooth'] :
                                                    ax[0].plot_date( (pdt,dt), (prev_anm,fgm), alpha=alpha_connector, **stl )
                                                else :
                                                    if not vargs['fg_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_anm,anm), alpha=alpha_connector, **stl )
                                                    prev_fgm = prev_ekf.fgmean().mean()
                                                    if not vargs['an_only'] :
                                                        ax[0].plot_date( (pdt,dt), (prev_fgm,fgm), alpha=alpha_connector, **stl1 )

                                            # if 'spread_plus_eo' in metrics :
                                            #     if vargs['sawtooth'] :
                                            #         prev_anse = np.sqrt( ( prev_ekf.anaspread()**2 + prev_ekf.obs(param='e_o')**2 ).mean() )
                                            #         ax[0].plot_date( (pdt,dt), (prev_anse,fgse), marker='', linestyle='-', color=col['spread_plus_eo'], alpha=alpha_connector )

                                            # if 'eo' in metrics :
                                            #     if vargs['sawtooth'] :
                                            #         eo      = ekf.obs(     param='e_o').mean()
                                            #         prev_eo = prev_ekf.obs(param='e_o').mean()
                                            #         ax[0].plot_date( (pdt,dt), (prev_eo,eo), marker='', linestyle='-', color=col['eo'], alpha=alpha_connector, label=lbl )

                                            # if 'mean' in metrics :
                                            #     if vargs['sawtooth'] :
                                            #         prev_anm = prev_ekf.anamean().mean()
                                            #         ax[0].plot_date( (pdt,dt), (prev_anm,fgm), marker='', linestyle='-', color=col['mean'], alpha=alpha_connector )
                                            #         obm = ekf.obs(filter='varname='+v).mean() 
                                            #         prev_obm = prev_ekf.obs().mean()
                                            #         ax[0].plot_date( (pdt,dt), (prev_obm,obm), marker='', linestyle='-', color=col['obsmean'], alpha=alpha_connector, label=lbl )
                                        
            else :
                if verbose :
                    print('    incomplete set of ekfs -- there are only ', len(ekfs))
                prev_ekf = None
            pdt = dt

        # finish and save plots
        print('finishing and saving plots for observation type {}...'.format(obs_type))
        for v in varnames :
            if v in nobs :
                print( '  -- ', obs_type, v, nobs[v] )

                # plot zero line
                dts = [ str2t(t) for t in valid_times ]
                fa[v]['ax'][0].plot_date( (dts[0],dts[-1]), (0,0), 'k', alpha=0.25, linewidth=0.5 )

                # adjust axes
                fa[v]['ax'][0].set_xlim((dts[0],dts[-1]))

                if not vargs['yrange'] is None :
                    vmin, vmax = [ float(s) for s in vargs['yrange'].split(',') ]
                    fa[v]['ax'][0].set_ylim((vmin,vmax))
                fa[v]['ax'][0].set_ylabel( '{} :: {} [{}]'.format(obs_type,v,nobs[v]) )

                adjust_time_axis( fa[v]['fig'], fa[v]['ax'][0], dts, density=1 )

                fa[v]['ax'][0].grid(alpha=0.2)

                if 'nobs' in metrics :
                    fa[v]['ax'][1].set_ylabel('#obs')
                    fa[v]['ax'][1].set_ylim(bottom=0)

                # construct and plot legend
                leglines = []
                legdesc  = []
                legtitle = ''

                if n_exps > 1 :
                    for ixp in range(n_exps) :
                        leglines.append( Line2D( [0], [0],
                                                 color = stl_c[ixp][metrics[0]][0] if vary['c'] == 'e' else 'k',
                                                 ls    = stl_s[ixp][metrics[0]][0] if vary['s'] == 'e' else '-',
                                                 lw    = stl_w[ixp][metrics[0]][0] if vary['w'] == 'e' else 1  ) )
                        legdesc.append(xps[ixp].exp_dir)
                else :
                    legtitle += xps[0].exp_dir + ' '

                if len(metrics) > 1 :
                    for met in metrics :
                        if met != 'nobs' : 
                            leglines.append( Line2D( [0], [0],
                                                    color = stl_c[0][met][0] if vary['c'] == 'm' else 'k',
                                                    ls    = stl_s[0][met][0] if vary['s'] == 'm' else '-',
                                                    lw    = stl_w[0][met][0] if vary['w'] == 'm' else 1  ) )
                            legdesc.append(met)
                else :
                    legtitle += metrics[0] + ' '

                if (not vargs['sawtooth']) and not (vargs['fg_only'] or vargs['an_only']) :
                    for ianfg in [0,1] :
                        leglines.append( Line2D( [0], [0],
                                                 color = stl_c[0][metrics[0]][ianfg] if vary['c'] == 'f' else 'k',
                                                 ls    = stl_s[0][metrics[0]][ianfg] if vary['s'] == 'f' else '-',
                                                 lw    = stl_w[0][metrics[0]][ianfg] if vary['w'] == 'f' else 1  ) )
                        legdesc.append( ['AN','FG'][ianfg] )
                else :
                    if not vargs['sawtooth'] :
                        if vargs['fg_only'] :
                            legtitle += 'FG'
                        if vargs['an_only'] :
                            legtitle += 'AN'

                if vargs['legloc'] == 'outside' : # to the right of the plot axes
                    fa[v]['ax'][0].legend( leglines, legdesc, title=legtitle, loc='upper left', bbox_to_anchor=(1., 0., 0.3, 1.0), frameon=False )
                else :
                    fa[v]['ax'][0].legend( leglines, legdesc, title=legtitle, loc=vargs['legloc'] )
                    
                # save figure
                fa[v]['fig'].savefig(obs_type+'_'+v+'_'+('-'.join(metrics))+'.'+vargs['image_type'], bbox_inches='tight')
                plt.close(fa[v]['fig'])


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # parse command line arguments
    parser = define_parser()
    args = parser.parse_args()
    
    # convert argparse object to dictionary
    vargs = vars(args)

    # remember current working directory
    cwd = os.getcwd()

    # create output directory, if required
    if not args.output_path is None :
        if not os.path.exists( args.output_path ) :
            os.makedirs( args.output_path )

    if args.evo :

        if args.compare : # comparison plots (all experiments in the same plot)

            if not args.output_path is None :
                os.chdir( args.output_path )

            plot_obs_evo( args.experiments, vargs )

        else : # individual plots for all experiments

            for xp_path in args.experiments :
                print()
                print('processing {}...'.format(xp_path))

                xp = BacyExp( xp_path )

                # by default, put plots in directory named like the experiment directory
                if args.output_path is None :
                    if not os.path.exists(xp.exp_dir) :
                        os.mkdir(xp.exp_dir)
                    os.chdir( xp.exp_dir)
                else :
                    os.chdir( args.output_path )

                plot_obs_evo( xp, vargs )

                os.chdir(cwd)
