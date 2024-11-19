#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . P L O T _ E K F
#  plot contents of EKF file
#
#  2016.10 L.Scheck 

from __future__ import absolute_import, division, print_function
# import matplotlib/pylab such that it does not require a X11 connection
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from numpy import *
import sys, argparse
from kendapy.ekf import Ekf, tables
from kendapy.binplot import binplot

def mean_distance( o, b, p, pmin=200 ) :
    #print '=== ', p.size, p[0], p[-1]
    k = 1
    e = 0
    while k < p.size and p[k] >= pmin :

        if (o[k]-b[k])*(o[k-1]-b[k-1]) > 0 : # same o-b sign at k and k-1
            e += 0.5*(p[k-1]-p[k])*( abs(o[k]-b[k]) + abs(o[k-1]-b[k-1]) )
            #print '=1= ', k, p[k], p[k-1], o[k], b[k], o[k-1], b[k-1], e
        else : # different signs -> lines cross
            alpha = -( o[k-1] - b[k-1] ) / ( (o[k]-o[k-1]) - (b[k]-b[k-1]) )
            e += 0.5*(p[k-1]-p[k])*abs( o[k-1] - b[k-1] )*alpha
            e += 0.5*(p[k-1]-p[k])*abs( o[k]   - b[k]   )*(1.0-alpha)
            #print '=2= ', k, p[k], p[k-1], o[k], b[k], o[k-1], b[k-1], e, alpha
        k += 1

    #print '=p= ', p[0] - p[k-1]
    return e / ( p[0] - p[k-1] )

#-------------------------------------------------------------------------------
def plot_radiosonde( ekf, irep, path='', vars=None, pmin=200, return_errors=False, nsmooth=1, file_format='png' ) :
    """Plot radio sonde profiles"""
    if vars is None :
        vars = ['RH', 'T', 'U', 'V']

    ret = {}

    fig, ax = plt.subplots( 2, len(vars), figsize=(5*len(vars),9) )

    filter = ekf.get_filter()
    ttl = None
    for i, vname in enumerate(vars) :
        ekf.add_filter(filter='varname=%s report=%d'%(vname,irep))
        if ttl is None :
            ttl = 'radio sonde started at %sUTC%dmin, latitude %.2f, longitude %.2f' % \
                 ( ekf.attr['verification_ref_time'], ekf.obs(param='time')[0],
                   ekf.obs(param='lat')[0], ekf.obs(param='lon')[0] )
        p = ekf.obs(param='plevel')/100

        obs     = ekf.obs()
        fgmean  = ekf.fgmean()
        anamean = ekf.anamean()

        if nsmooth == 2 :
            obs[1:-1]     = 0.25*obs[:-2] + 0.5*obs[1:-1] + 0.25*obs[2:]
            fgmean[1:-1]  = 0.25*fgmean[:-2] + 0.5*fgmean[1:-1] + 0.25*fgmean[2:]
            anamean[1:-1] = 0.25*anamean[:-2] + 0.5*anamean[1:-1] + 0.25*anamean[2:]
        elif nsmooth == 3 :
            obs[1:-1]     = 0.33*obs[:-2] + 0.34*obs[1:-1] + 0.33*obs[2:]
            fgmean[1:-1]  = 0.33*fgmean[:-2] + 0.34*fgmean[1:-1] + 0.33*fgmean[2:]
            anamean[1:-1] = 0.33*anamean[:-2] + 0.34*anamean[1:-1] + 0.33*anamean[2:]


        if False :
            idcs = where( p > pmin )
            efg = sqrt(((fgmean  - obs)**2)[idcs].mean())
            ean = sqrt(((anamean - obs)**2)[idcs].mean())
        else :
            efg = mean_distance( obs, fgmean,  p, pmin=pmin )
            ean = mean_distance( obs, anamean, p, pmin=pmin )
            print('>>>>>>>>>>>>> MEAN DISTANCES : ', efg, ean)

        ret[vname+'_rmse_fg']  = efg
        ret[vname+'_rmse_ana'] = ean
        ret[vname+'_delta_rmse'] = ean - efg

        ax[0,i].plot( obs,     p, 'k', label='%s %f'%(vname,efg-ean))
        ax[0,i].plot( fgmean,  p, 'b', label='FG %f'%efg)
        ax[0,i].plot( anamean, p, 'r', label='AN %f'%ean)
        ax[0,i].set_ylim((1000,pmin))
        if vname == 'RH' :
            ax[0,i].set_xlim((0,1))
        elif vname == 'T' :
            ax[0,i].set_xlim((200,300))
        ax[0,i].legend(frameon=False,fontsize=10)

        ax[1,i].plot( fgmean  - obs, p, 'b', label='FG %f'%efg )
        ax[1,i].plot( anamean - obs, p, 'r', label='AN %f'%ean )
        ax[1,i].plot( (0,0), (1000,pmin), 'k--', linewidth=0.4, alpha=0.5 )
        ax[1,i].set_ylim((1000,pmin))
        if vname == 'RH' :
            ax[1,i].set_xlim((-0.5,0.5))
        elif vname == 'T' :
            ax[1,i].set_xlim((-3,3))
        else :
            ax[1,i].set_xlim((-5,5))

        ekf.replace_filter(filter=filter)

    fig.suptitle(ttl)
    fig.savefig( path + ('/' if path != '' else '') + 'radiosonde'+('_smooth%d'%nsmooth if nsmooth > 1 else '')+'.'+file_format )
    plt.close(fig)

    if return_errors :
        return ret

#-------------------------------------------------------------------------------
def plot_departures( ekf, vars=[], vmin=-5, vmax=5, nbins=101, output_path='', file_type='png' ) :
    """Plot first guess and analysis ensemble departures from observations"""

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    plt.figure(55,figsize=(10,10))
    col = {'anaens':'r','fgens':'b'}

    for vname in vars :
        print(('plotting departures for variable ', vname, '...'))

        plt.clf()
        for veri in ['anaens','fgens'] :
            dep = ekf.departures( veri, normalize=True, varname_filter=vname ).ravel()
            bins = linspace(vmin,vmax,nbins)
            dep_hist, dep_hist_edges = histogram( dep, bins=bins )
            binplot( dep_hist_edges, dep_hist, color=col[veri], semilogy=True, label=veri )

            if vname == 'REFL' and veri == 'fgens' :
                fgmean = ekf.fgmean(varname_filter=vname)
                obs   = ekf.obs(varname_filter=vname)

                #idcs = where

                #print 'SHSSHSHSAPES ', fgmean.shape, obs.shape, dep.shape
                e_o = (1 + 5*(abs(fgmean-obs)-0.15))*ekf.obs(param='e_o',varname_filter=vname)
                dep = ekf.departures( veri, normalize=True, e_o=e_o, varname_filter=vname ).ravel()
                dep_hist, dep_hist_edges = histogram( dep, bins=bins )
                binplot( dep_hist_edges, dep_hist, color='g', semilogy=True, label=veri+' e.m.' )

        plt.grid()
        plt.legend( frameon=False, loc='upper left' )
        plt.title( 'normalized departures for %s (%d observations)' % (vname,ekf.obs(varname_filter=vname).size) )
        figfname = path+fname+'_normdep_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )

#-------------------------------------------------------------------------------
def plot_rankhist( ekf, vars=[], output_path='', file_type='png' ) :

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    plt.figure(55,figsize=(10,10))
    col = {'anaens':'r','fgens':'b'}

    for vname in vars :
        print(('plotting rank histogram for variable ', vname, '...'))

        stat = ekf.statistics(rankhist=True,varname_filter=vname)
        print(("STATISTICS AVAILABLE ", list(stat.keys())))

        plt.clf()
        for veri in ['anaens','fgens'] :
            binplot( arange(ekf.nens+1)+1, stat[veri]['rankhist'], color=col[veri], label=veri )
        plt.grid()
        plt.legend( frameon=False, loc='upper left' )
        plt.title( 'rank histogram %s (%d observations)' % (vname,ekf.obs(varname_filter=vname).size) )
        figfname = path+fname+'_rankhist_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )

#-------------------------------------------------------------------------------
def plot_distribution( ekf, vars=[], vmin=None, vmax=None, nbins=31, output_path='', file_type='png' ) :
    """Plot first guess and analysis ensemble and observation value distribution"""

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    plt.figure(55,figsize=(10,10))
    col = {'anaens':'r','fgens':'b'}

    for vname in vars :
        print(('plotting value distribution for variable ', vname, '...'))

        obs    = ekf.obs(    varname_filter=vname )
        anaens = ekf.anaens( varname_filter=vname )
        fgens  = ekf.fgens(  varname_filter=vname )

        # set bins
        if vmin is None :
            vmin_ = r_[ obs.min(),anaens.min(),fgens.min() ].min()
        else :
            vmin_ = vmin
        if vmax is None :
            vmax_ = r_[ obs.max(),anaens.max(),fgens.max() ].max()
        else :
            vmax_ = vmax
        bins = linspace(vmin_,vmax_,nbins)

        # compute histograms
        obs_hist, hist_edges = histogram( obs,    bins=bins )
        ana_hist, hist_edges = histogram( anaens, bins=bins )
        fg_hist,  hist_edges = histogram( fgens,  bins=bins )

        # normalize -> nens members can be compared to one observation
        ana_hist = ana_hist/float(ekf.nens)
        fg_hist  = fg_hist /float(ekf.nens)

        plt.clf()
        binplot( hist_edges, fg_hist,  color='b', label='fgens' )
        binplot( hist_edges, ana_hist, color='r', label='anaens' )
        binplot( hist_edges, obs_hist, color='k', label='obs (%d)'%obs.size )
        plt.grid()
        plt.legend( frameon=False, loc='upper left' )
        plt.title( 'value distribution for '+vname )
        figfname = path+fname+'_valdist_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )

#-------------------------------------------------------------------------------
def plot_model_vs_obs( ekf, vars=[], output_path='', file_type='png' ) :

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    plt.figure(55,figsize=(10,10))
    
    #var = ekf['var']
    #if len(vars) == 0 : vars = var.keys()

    #stat = kendapy.ekf.compute_statistics(ekf)

    for vname in vars :

        obs      = ekf.obs(              varname_filter=vname )
        e_o      = ekf.obs( param='e_o', varname_filter=vname )
        fgmean   = ekf.fgmean(   varname_filter=vname )
        anamean  = ekf.anamean(  varname_filter=vname )
        fgspread = ekf.fgspread( varname_filter=vname )

        #anaens = ekf.anaens( varname=vname )
        #fgens  = ekf.fgens(  varname=vname )

        stat   = ekf.statistics( varname_filter=vname )

        print(('plotting variable ', vname, '...'))

        dalpha=0.2

        plt.clf()
        if obs.size < 1000 :
            for i in range(obs.size) :
                plt.plot( [obs[i],obs[i]],
                          [fgmean[i]-0.5*fgspread[i],fgmean[i]+0.5*fgspread[i]],
                          color='b', linewidth=0.5 )
                plt.plot( [obs[i]-0.5*e_o[i],obs[i]+0.5*e_o[i]],
                          [fgmean[i],fgmean[i]], color='b', linewidth=0.5 )
            for i in range(obs.size) :
                plt.plot( [obs[i]-0.5*e_o[i],obs[i]+0.5*e_o[i]],
                          [anamean[i],anamean[i]], color='r', linewidth=0.5 )
        plt.scatter( obs, fgmean, color='b', s=5, alpha=dalpha, edgecolor='none' )
        plt.scatter( obs, anamean, color='r', s=5, alpha=dalpha, edgecolor='none' )

        plt.xlabel('observation')
        plt.ylabel('mean model equivalent')
        plt.xlim((obs.min(),obs.max() ))
        plt.ylim((obs.min(),obs.max() ))
        plt.plot((obs.min(),obs.max() ), (obs.min(),obs.max() ), '--k', linewidth=0.5 )
        plt.figtext( 0.15, 0.86, 'n = %d' % obs.size )
        plt.figtext( 0.15, 0.83, 'RMSE_fg  = %g' % stat['fgmean']['rmse'],  color='b' )
        plt.figtext( 0.15, 0.80, 'RMSE_ana = %g' % stat['anamean']['rmse'], color='r' )
        plt.figtext( 0.15, 0.77, 'BIAS_fg  = %g' % stat['fgmean']['bias'],  color='b' )
        plt.figtext( 0.15, 0.74, 'BIAS_ana = %g' % stat['anamean']['bias'], color='r' )
        figfname = path+fname+'_model_vs_obs_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )

        fgdep = fgmean-obs
        anadep = anamean-obs
        plt.clf()
        plt.scatter( obs, fgdep, color='b', s=5, alpha=dalpha, edgecolor='none' )
        plt.scatter( obs, anadep, color='r', s=5, alpha=dalpha, edgecolor='none' )
        plt.xlabel('observation')
        plt.ylabel('ensemble mean - observation')
        plt.xlim((obs.min(),obs.max() ))
        plt.plot((obs.min(),obs.max() ), (0,0), '--k', linewidth=0.5 )
        figfname = path+fname+'_departure_vs_obs_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )


#-------------------------------------------------------------------------------
def plot_lochist( ekf, vars=[], nbins=21, output_path='', file_type='png' ) :

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    # plot localization histograms
    plt.figure(55,figsize=(10,10))

    for vname in vars :
        print(('plotting horizontal localization histogram for variable ', vname, '...'))

        h_loc = ekf.obs(param='h_loc',varname_filter=vname)
        if isinstance(h_loc,ma.MaskedArray) :
            print(('WARNING: Only %d of %d h_loc entries are valid!' % (h_loc.count(),h_loc.size)))

        vmin  = h_loc.min()
        vmean = h_loc.mean()
        vmax  = h_loc.max()
        if vmax-vmin < vmean*1e-2 :
            vmax = vmean*1.1
            vmin = vmean*0.9
        #bins = linspace(vmin,vmax,nbins)
        print(('>>> vmin,vmean,vmax = ', vmin, vmean, vmax))
        bins = linspace( float(vmin), float(vmax), int(nbins) )
        h_loc_hist, edges = histogram( h_loc, bins=bins )

        plt.clf()
        binplot( edges, h_loc_hist )
        plt.grid()
        plt.title( 'horizontal localization radii for %s (%d observations)' % (vname,ekf.obs(varname_filter=vname).size) )
        figfname = path+fname+'_h_loc_hist_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )

        v_loc = ekf.obs(param='v_loc',varname_filter=vname)
        print(('v_loc type', type(v_loc)))
        if v_loc is None :
            print('could not find v_loc')
            v_loc = h_loc*0.0
        if isinstance(v_loc,ma.MaskedArray) :
            print(('WARNING: Only %d of %d v_loc entries are valid!' % (v_loc.count(),v_loc.size)))
            
        vmin  = v_loc.min()
        vmean = v_loc.mean()
        vmax  = v_loc.max()
        if vmax-vmin < vmean*1e-2 :
            vmax = vmean*1.1
            vmin = vmean*0.9
        print(('>>> vmin,vmean,vmax = ', vmin, vmean, vmax))
        bins = linspace( float(vmin), float(vmax), int(nbins) )
        v_loc_hist, edges = histogram( v_loc, bins=bins )

        plt.clf()
        binplot( edges, v_loc_hist )
        plt.grid()
        plt.title( 'vertical localization radii for %s (%d observations)' % (vname,ekf.obs(varname_filter=vname).size) )
        figfname = path+fname+'_v_loc_hist_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )


        # observation error histogram
        e_o = ekf.obs(param='e_o',varname_filter=vname)
        print(('e_o type', type(e_o)))
        if isinstance(e_o,ma.MaskedArray) :
            print(('WARNING: Only %d of %d e_o entries are valid!' % (e_o.count(),e_o.size)))
            
        vmin  = e_o.min()
        vmean = e_o.mean()
        vmax  = e_o.max()
        if vmax-vmin < vmean*1e-2 :
            vmax = vmean*1.1
            vmin = vmean*0.9
        print(('>>> vmin,vmean,vmax = ', vmin, vmean, vmax))
        bins = linspace( float(vmin), float(vmax), int(nbins) )
        e_o_hist, edges = histogram( e_o, bins=bins )

        plt.clf()
        binplot( edges, e_o_hist )
        plt.grid()
        plt.title( 'observation errors for %s (%d observations)' % (vname,ekf.obs(varname_filter=vname).size) )
        figfname = path+fname+'_e_o_hist_'+vname+'.'+file_type
        print(('saving %s ...' % figfname)) 
        plt.savefig( figfname )


#-------------------------------------------------------------------------------
def plot_obsloc( ekf, vars=[], proj='geos', region='D2', output_path='', file_name=None, file_type='png',
                 statnames=False, plot_hloc=False, markersize=1 ) :

    from matplotlib.patches import Ellipse

    if len(vars) == 0 : vars = ekf.varnames
    
    if file_name is None :
        fname = ekf.fname.replace('.nc','').split('/')[-1]
    else :
        fname = file_name
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    # plot observation locations for all variables

    try :
        from mpl_toolkits.basemap import Basemap, cm
        use_bm = True
    except :
        print('WARNING: Basemap not available')
        use_bm = False
    if proj is None or proj == '' or proj.lower() == 'none' :
        use_bm = False

    if use_bm :
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])

        if region == 'DE' :
            lat_min = 45.0
            lat_max = 56.5
            lon_min = 3.0
            lon_max = 20.0
        elif region == 'D2' :
            lat_min = 43.0
            lat_max = 58.0
            lon_min = -3.0
            lon_max = 20.0
        else :
            lat_min, lon_min, lat_max, lon_max = [ float(f) for f in region.split(',') ]

        if proj == 'stere' : # stereographic
            m = Basemap(projection='stere', lon_0=10, lat_0=90., #lat_ts=lat_0,
                        llcrnrlat=lat_min,urcrnrlat=lat_max,
                        llcrnrlon=lon_min,urcrnrlon=lon_max,
                        rsphere=6371200.,resolution='l',area_thresh=1000)
        else : # geostationary
            lon_0=11.
            m = Basemap(projection='geos',lon_0=lon_0,resolution='l',\
                        llcrnrlat=lat_min-0.1,urcrnrlat=lat_max-0.2,
                        llcrnrlon=lon_min,urcrnrlon=lon_max)
        
            # if proj == 'stere' : # stereographic
            #     m = Basemap(projection='stere', lon_0=10, lat_0=90., #lat_ts=lat_0,
            #                 llcrnrlat=45.0,urcrnrlat=56.5,
            #                 llcrnrlon=3.0,urcrnrlon=20.0,
            #                 rsphere=6371200.,resolution='l',area_thresh=1000)
            # else : # geostationary
            #     lon_0=11.
            #     m = Basemap(projection='geos',lon_0=lon_0,resolution='l',\
            #                 llcrnrlat=44.9,urcrnrlat=56.3,
            #                 llcrnrlon=3.0,urcrnrlon=20.0)

        #m = Basemap(projection='stere', lon_0=10, lat_0=90., #lat_ts=lat_0,
        #             llcrnrlat=45.0,urcrnrlat=56.0,
        #             llcrnrlon=3.0,urcrnrlon=20.0,
        #             rsphere=6371200.,resolution='h',area_thresh=1000)

        m.drawcoastlines(linewidth=0.25)
        m.drawcountries(linewidth=0.25)
        #map.fillcontinents(color='coral',lake_color='aqua')
        # draw the edge of the map projection region (the projection limb)
        #map.drawmapboundary(fill_color='aqua')
        # draw lat/lon grid lines every 30 degrees.

        #m.drawcoastlines()
        #m.drawstates()
        #m.drawcountries()

        parallels = arange(0.,90,5.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        meridians = arange(0.,180.,5.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    else :
        lat = ekf.obs(param='lat',varname_filter=vars[0])
        lon = ekf.obs(param='lon',varname_filter=vars[0])
        
        fig = plt.figure(figsize=( 8, 8*((lat.max()-lat.min())/(lon.max()-lon.min()))/cos(lat.mean()*pi/180.0) ) )
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        m = ax

    cols = ['k', 'r','g','b','#ff00ff','#00ffff','#ffff00','#ff0099','#ff9900','#9900ff','#0099ff']

    for i, v in enumerate(vars) :
        lat = ekf.obs(param='lat',varname_filter=v)
        lon = ekf.obs(param='lon',varname_filter=v)
        if 'h_loc' in ekf.data.keys() :
            h_loc = ekf.obs(param='h_loc',varname_filter=v)
        else :
            h_loc = None

        dlat = 0.0
        dlon = 0.0
        if i == 1 : dlat = 0.05
        if i == 2 : dlon = 0.05
        if use_bm :
            x, y = m( lon+dlon, lat+dlat )

            if statnames : # add station name
                statids = ekf.statids(varname_filter=v)
                for k in range(lat.size) :
                    if (lat[k] >= lat_min) & (lat[k] <= lat_max) & (lon[k] >= lon_min) & (lon[k] <= lon_max) :
                        ax.text( x[k], y[k], ' '+statids[k], fontsize=5 )

            if plot_hloc and not h_loc is None :
                for k in range(lat.size) :
                    #           km         degree / km
                    dlon_circ = h_loc[k] * 180.0 / ( cos(lat[k]*pi/180.0) * pi*6371.0)
                    dlat_circ = h_loc[k] * 180.0 / (                        pi*6371.0)

                    n_circ = 20
                    zeta = linspace(0,2*pi,n_circ)
                    x_circ, y_circ = m( lon[k] + dlon + dlon_circ * cos(zeta), lat[k] + dlat + dlat_circ * sin(zeta) )
                    
                    m.plot( x_circ, y_circ, color=cols[i%len(cols)], alpha=0.1 )

                #for k in range(lat.size) :
                #    e = Ellipse( xy=(x[k],y[k]), width=h_loc[k]*180.0/(cos(lat[k]*pi/180.0)*pi*6371.0), height=h_loc[k]*180.0/(pi*6371.0) )
                #    m.add_artist(e)
                #    e.set_clip_box(m.bbox)
                #    e.set_alpha(0.025)
                #    e.set_facecolor(cols[i%len(cols)])
        else :
            x, y = lon+dlon, lat+dlat

            if plot_hloc and not h_loc is None :
                if isinstance(h_loc,ma.MaskedArray) :
                    print(('WARNING: Only %d of %d h_loc entries are valid!' % (h_loc.count(),h_loc.size)))

                #else :
                    for k in range(lat.size) :
                        e = Ellipse( xy=(x[k],y[k]), width=h_loc[k]*180.0/(cos(lat[k]*pi/180.0)*pi*6371.0), height=h_loc[k]*180.0/(pi*6371.0) )
                        m.add_artist(e)
                        e.set_clip_box(m.bbox)
                        e.set_alpha(0.025)
                        e.set_facecolor(cols[i%len(cols)])           

        m.scatter( x, y, marker='o',color=cols[i%len(cols)], label=v, s=markersize )
    plt.legend(frameon=False,loc='upper left')
    plt.title(ekf.fname.split('/')[-1]+' : observation locations'+(' + h_loc' if plot_hloc else ''))
    figfname = path+fname+'_lat_lon_map.'+file_type
    print('plot_obsloc: saving {} ...'.format(figfname))
    plt.savefig( figfname, bbox_inches='tight' )


#-------------------------------------------------------------------------------
def plot_map( ekf, vars=[], proj='geos', output_path='', file_type='png' ) :

    from matplotlib.patches import Ellipse

    if len(vars) == 0 : vars = ekf.varnames
    
    fname = ekf.fname.replace('.nc','').split('/')[-1]
    if output_path == '' :
        path = output_path
    else :
        path = output_path + '/'

    try :
        from mpl_toolkits.basemap import Basemap, cm
        use_bm = True
    except :
        print('WARNING: Basemap not available')
        use_bm = False
    if proj is None or proj == '' or proj.lower() == 'none' :
        use_bm = False

    for vname in vars :

        for veri in ['fgmean','anamean','obs','fgspread'] :

            # plot observations on map
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_axes([0.1,0.1,0.8,0.8])


            if use_bm :
                if proj == 'stere' : # stereographic
                    m = Basemap(projection='stere', lon_0=10, lat_0=90., #lat_ts=lat_0,
                                llcrnrlat=45.0,urcrnrlat=56.5,
                                llcrnrlon=3.0,urcrnrlon=20.0,
                                rsphere=6371200.,resolution='l',area_thresh=1000)
                else : # geostationary
                    lon_0=11.
                    m = Basemap(projection='geos',lon_0=lon_0,resolution='l',\
                                llcrnrlat=44.9,urcrnrlat=56.3,
                                llcrnrlon=3.0,urcrnrlon=20.0)
                #m = Basemap(projection='stere', lon_0=10, lat_0=90., #lat_ts=lat_0,
                #             llcrnrlat=45.0,urcrnrlat=56.0,
                #             llcrnrlon=3.0,urcrnrlon=20.0,
                #             rsphere=6371200.,resolution='h',area_thresh=1000)
                m.drawcoastlines()
                m.drawstates()
                m.drawcountries()
                parallels = arange(0.,90,5.)
                m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
                meridians = arange(0.,360.,5.)
                m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
            else :
                m = ax

            lat = ekf.obs(param='lat',varname_filter=vname)
            lon = ekf.obs(param='lon',varname_filter=vname)
            if veri == 'obs' :
                val = ekf.obs(varname_filter=vname)
            else :
                val = ekf.veri(veri,varname_filter=vname)

            if use_bm :
                x, y = m( lon, lat )
            else :
                x, y = lon, lat

            mpb = m.scatter( x, y, c=val, marker='o', edgecolor='',s=30 )

            plt.title(ekf.fname.split('/')[-1]+' : '+vname)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(mpb, cax=cax)

            figfname = path+fname+'_'+veri+'_map_'+vname+'.'+file_type
            print(('SAVING ', figfname))
            plt.savefig( figfname, bbox_inches='tight' )




#-------------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------------
#-------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Plot contents of ekf files')
    # plot types
    parser.add_argument( '-d', '--departures',    dest='plot_departures',    help='plot departures', action='store_true' )
    parser.add_argument( '-D', '--distributions', dest='plot_distribution',  help='plot value distributions', action='store_true' )
    parser.add_argument( '-r', '--rankhist',      dest='plot_rankhist',      help='plot rank histogram', action='store_true' )
    parser.add_argument( '-M', '--model-vs-obs',  dest='plot_model_vs_obs',  help='plot model vs. obs values', action='store_true' )
    parser.add_argument( '-l', '--locations',     dest='plot_obsloc',        help='plot observation locations', action='store_true' )
    parser.add_argument(       '--statnames',     dest='statnames',          help='add station names to observation location plot', action='store_true' )
    parser.add_argument(       '--h-loc',         dest='plot_hloc',          help='add localization radii to observation location plot', action='store_true' )
    parser.add_argument(       '--markersize',    dest='markersize',         help='marker size for observations [default 3]', type=int, default=3 )
    parser.add_argument( '-R', '--region',        dest='region',             help='plotting region [ D2 | DE | lat_min,lon_min,lat_max,lon_max ]', default='D2' )
    parser.add_argument( '-L', '--localization',  dest='plot_lochist',       help='plot localization radius histogram', action='store_true' )
    parser.add_argument( '-m', '--maps',          dest='plot_map',           help='plot map', action='store_true' )
    # settings
    parser.add_argument( '-V', '--variables',     dest='vars',               help='comma-separated list of variables to be plotted (default: all)', default='' )
    parser.add_argument( '-f', '--filter',        dest='filter',             help='filter string (see ekf.py)', default='state=active' )
    parser.add_argument( '-p', '--path',          dest='output_path',        help='path to the directory in which the plots will be saved', default='' )
    parser.add_argument(       '--filetype',      dest='file_type',          help='file type [ png (default) | pdf | eps | svg ...]', default='png' )
    parser.add_argument( 'ekffiles', metavar='ekffiles', help='ekf file names', nargs='*' )
    args = parser.parse_args()

    plot_all = False
    if args.plot_departures | args.plot_distribution |  args.plot_rankhist | args.plot_model_vs_obs | args.plot_obsloc | args.plot_lochist | args.plot_map == False :
        plot_all = True

    if args.vars == '' :
        vars = []
    else :
        vars = args.vars.split(',')

    for f in args.ekffiles :
        print(("========================= %s ======================" % f))
        #ekf = kendapy.ekf.read_ekf( f, verbose=True, statistics=True, filter='valid' )
        ekf = Ekf(f, filter=args.filter)
        if args.plot_departures   or plot_all : plot_departures(   ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 
        if args.plot_distribution or plot_all : plot_distribution( ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 
        if args.plot_rankhist     or plot_all : plot_rankhist(     ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 
        if args.plot_model_vs_obs or plot_all : plot_model_vs_obs( ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 
        if args.plot_obsloc       or plot_all : plot_obsloc(       ekf, vars=vars, output_path=args.output_path, file_type=args.file_type,
                                                                   statnames=args.statnames, region=args.region, plot_hloc=args.plot_hloc,
                                                                   markersize=args.markersize ) 
        if args.plot_lochist      or plot_all : plot_lochist(      ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 
        if args.plot_map          or plot_all : plot_map(          ekf, vars=vars, output_path=args.output_path, file_type=args.file_type ) 

