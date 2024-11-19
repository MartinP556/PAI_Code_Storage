#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#
#   P L O T M A P
#   generate map plots for unstructured grid data

import os, sys, time, pickle
from matplotlib import pyplot as plt
import netCDF4
import numpy as np
import xarray as xr
from numba import jit, float64, float32, int32

from kendapy.cosmo_grid import cosmo_grid, nonrot_to_rot_grid, rot_to_nonrot_grid

#-----------------------------------------------------------------------------------------------------------------------
def define_parser() :
    parser = argparse.ArgumentParser(description='Generate map plots for unstructured grid data')

    parser.add_argument( '-g', '--gridfile',        dest='gridfile',      default='',             help='file containing model grid' )
    parser.add_argument( '-s', '--statefile',       dest='modelvarsfile', default='',             help='file containing model state variables' )
    parser.add_argument( '-V', '--variables',       dest='variables',     default='T,P.PRES,HHL,DEN,QV_DIA,QC_DIA,QI_DIA,CLC', help='names (and internal names) of variables to be read from model state file' )
    parser.add_argument( '-t', '--time-index',      dest='time_index',    type=int, default=0,    help='time index' )
    parser.add_argument(       '--grid-resolution', dest='gridres', type=float, default=150.0,    help='nominal grid resolution [m]' )
    parser.add_argument(       '--skip-layers-top', dest='skip_layers_top', default=0, type=int,  help='number of layers to ignore at the model top')
    
    parser.add_argument( '-l', '--level',           dest='level', default=-1, type=int,           help='model level index (0=top, -1=bottom)' )
    parser.add_argument(       '--hslice',          dest='hslice', default=None,                  help='definition of horizontal slice, e.g. P=50000, on which the variables will be plotted' )

    parser.add_argument(       '--vmin',            dest='vmin', type=float, default=None,        help='value correspnding to lower end of color bar' )
    parser.add_argument(       '--vmax',            dest='vmax', type=float, default=None,        help='value correspnding to upper end of color bar' )
    parser.add_argument(       '--cmap',            dest='cmap', default='magma',                 help='value correspnding to upper end of color bar' )

    parser.add_argument( '-r', '--rotated',         dest='rotated',    action='store_true',       help='plot on rotated lat-lon grid' )
    parser.add_argument(       '--oversample',      dest='oversample', default=1,   type=int,     help='oversampling factor for rotated grid')
    parser.add_argument(       '--pixperdeg',       dest='pixperdeg',  default=100, type=int,     help='lat-lon grid resolution in pixels per degree')
    
    parser.add_argument( '-i', '--image-type',      dest='image_type', default='png',             help='[ png | eps | pdf ... ]' )

    parser.add_argument( '-v', '--verbose', dest='verbose', action='store_true', help='be more verbose' )

    return parser

#-----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, nogil=True)
def det2d( ux, uy, vx, vy ) :
    return ux*vy - uy*vx

#-----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, nogil=True)
def point_in_triangle( lat, lon, vlat, vlon ) :

    # see http://mathworld.wolfram.com/TriangleInterior.html
    d12 = det2d( vlon[1]-vlon[0], vlat[1]-vlat[0], vlon[2]-vlon[0], vlat[2]-vlat[0] )
    if d12 != 0 :
        a   =   ( det2d( lon, lat, vlon[2]-vlon[0], vlat[2]-vlat[0] ) \
                - det2d( vlon[0], vlat[0], vlon[2]-vlon[0], vlat[2]-vlat[0] ) ) / d12
        b   = - ( det2d( lon, lat, vlon[1]-vlon[0], vlat[1]-vlat[0] ) \
                - det2d( vlon[0], vlat[0], vlon[1]-vlon[0], vlat[1]-vlat[0] ) ) / d12
    else :
        a = 0
        b = 0

    inside = False
    if (a>0) and (b>0) and (a+b<1) :
        inside = True
    return inside

#-----------------------------------------------------------------------------------------------------------------------
def tri2latlon( ll_grid, tri_grid, tri_var, method='coarse', nsearch=3 ) :
    """Map variable defined on triangular grid onto lat-lon grid"""

    nlat, nlon = int(round(ll_grid['nlat'])), int(round(ll_grid['nlon']))
    latlon_var = np.zeros((nlat, nlon),dtype=np.float64)

    if method == 'coarse' :
        latlon_hits =  np.zeros((nlat, nlon),dtype=np.int32)
        hits_min = tri2latlon_coarse( latlon_var, latlon_hits,
                                      ll_grid['lat_min'], ll_grid['lon_min'], ll_grid['dlat'], ll_grid['dlon'],
                                      tri_var.astype(np.float64), tri_grid['clat'].astype(np.float64), tri_grid['clon'].astype(np.float64) )
        print('Triangles/quad > ', hits_min)
    elif method == 'fine' :
        latlon_hits =  np.zeros((nlat, nlon),dtype=np.int32)
        misses = tri2latlon_fine( latlon_var, latlon_hits,
                                  ll_grid['lat_min'], ll_grid['lon_min'], ll_grid['dlat'], ll_grid['dlon'],
                                  tri_var.astype(np.float64), tri_grid['clat'].astype(np.float64), tri_grid['clon'].astype(np.float64),
                                  tri_grid['vlat'].astype(np.float64), tri_grid['vlon'].astype(np.float64), tri_grid['vertex_of_cell'].astype(np.int32), nsearch )
        print('misses in tri2latlon_fine (triangles -> quads) : ', misses)
    else :
        raise ValueError("ERROR: Method {} not implemented".format(method) )
        
    return latlon_var

#-----------------------------------------------------------------------------------------------------------------------
@jit('int32(           float64[:,:], int32[:,:],  float64, float64, float64, float64, float64[:], float64[:], float64[:])', nopython=True, nogil=True)
def tri2latlon_coarse( latlon_var,   latlon_hits, lat_min, lon_min, dlat,    dlon,    tri_var,    clat,       clon      ) :
    """Assume lat-lon grid is coarser than unstructured grid so that each lat-lon cell contains at least one triangle"""

    nlat, nlon = latlon_var.shape
    for i in range(clon.size) :
        ilat = int( (clat[i]-lat_min)/dlat )
        ilon = int( (clon[i]-lon_min)/dlon )
        latlon_hits[ ilat, ilon ] += 1
        latlon_var[  ilat, ilon ] += tri_var[i]

    for ilat in range(nlat) :
        for ilon in range(nlon) :
            if latlon_hits[ ilat, ilon ] > 1 :
                latlon_var[  ilat, ilon ] /= latlon_hits[ ilat, ilon ]

    return latlon_hits.min()

#-----------------------------------------------------------------------------------------------------------------------
@jit('int32(         float64[:,:], int32[:,:],  float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:],     int32   )', nopython=True, nogil=True)
def tri2latlon_fine( latlon_var,   latlon_hits, lat_min, lon_min, dlat,    dlon,    tri_var,    clat,       clon,       vlat,       vlon,       vertex_of_cell, nsearch ) :
    """Assume lat-lon grid is finer than unstructured grid so that each triangle contains at least one lat-lon quad"""

    #figure(44,figsize=(8,8))
    #clf()

    nlat, nlon = latlon_var.shape
    trilat = np.zeros(3)
    trilon = np.zeros(3)

    # determine dimensions of first triangle
    cidx=0
    trilat[0] = vlat[vertex_of_cell[0,cidx]]
    trilat[1] = vlat[vertex_of_cell[1,cidx]]
    trilat[2] = vlat[vertex_of_cell[2,cidx]]
    trilon[0] = vlon[vertex_of_cell[0,cidx]]
    trilon[1] = vlon[vertex_of_cell[1,cidx]]
    trilon[2] = vlon[vertex_of_cell[2,cidx]]
    dlat_tri = trilat.max() - trilat.min()
    dlon_tri = trilon.max() - trilon.min()

    # search area
    nsearch = int(np.maximum( dlat_tri/dlat, dlon_tri/dlon ))+1

    for cidx in range(clon.size) :

        # find cell containing triangle center
        ilat = int( round( (clat[cidx]-lat_min)/dlat ) )
        ilon = int( round( (clon[cidx]-lon_min)/dlon ) )

        if ilat < 0 or ilat >= nlat or ilon < 0 or ilon >= nlon :
            continue

        # get vertices of triangle
        trilat[0] = vlat[vertex_of_cell[0,cidx]]
        trilat[1] = vlat[vertex_of_cell[1,cidx]]
        trilat[2] = vlat[vertex_of_cell[2,cidx]]
        trilon[0] = vlon[vertex_of_cell[0,cidx]]
        trilon[1] = vlon[vertex_of_cell[1,cidx]]
        trilon[2] = vlon[vertex_of_cell[2,cidx]]

        #plot( [ trilon[i] for i in (0,1,2,0)], [ trilat[i] for i in (0,1,2,0)], 'k' )

        # identify neighbor cells with centers in the same triangle
        for i in range( np.maximum(ilat-nsearch,0), np.minimum(ilat+1+nsearch,nlat) ) :
            for j in range( np.maximum(ilon-nsearch,0), np.minimum(ilon+1+nsearch,nlon) ) :
                if point_in_triangle( i*dlat+lat_min, j*dlon+lon_min, trilat, trilon ) :
                    #scatter( j*dlon+lon_min, i*dlat+lat_min, s=20, facecolor='r', edgecolor=None )
                    latlon_hits[ i, j ] += 1
                    latlon_var[  i, j ] += tri_var[cidx]
                #else :
                #    scatter( j*dlon+lon_min, i*dlat+lat_min, s=20, facecolor='g', edgecolor=None )

        #scatter( ilon*dlon+lon_min, ilat*dlat+lat_min, s=20, facecolor='b', edgecolor=None )
        #scatter( clon[cidx], clat[cidx], s=5, facecolor='k', edgecolor=None )

    misses = 0
    for ilat in range(nlat) :
        for ilon in range(nlon) :
            if latlon_hits[ ilat, ilon ] == 0 :  # this should not be necessary, but it is...
                if (ilat > 0) and (ilat < nlat-1) and (ilon > 0) and (ilon < nlon-1) :
                    s = latlon_hits[ ilat-1:ilat+2, ilon-1:ilon+2 ].sum()
                    if s > 0 :
                        latlon_var[  ilat, ilon ] = (  latlon_hits[ ilat-1:ilat+2, ilon-1:ilon+2 ] \
                                                     * latlon_var[ ilat-1:ilat+2, ilon-1:ilon+2 ]).sum() / s
                        misses += 1

            elif latlon_hits[ ilat, ilon ] > 1 : # this also should not be necessary, but it is...
                latlon_var[  ilat, ilon ] = latlon_hits[ ilat, ilon ] / latlon_hits[ ilat, ilon ]
                latlon_hits[ ilat, ilon ] = 1

    return misses

#-------------------------------------------------------------------------------------------------
def generate_latlon_grid( lat_min, lon_min, lat_max, lon_max, nlat, nlon, dim=1, first_dim='lon' ) :
    """Generate regular latitude-longitude grid with the specified limits and resolution"""

    r =    { 'lat_min':lat_min, 'lon_min':lon_min, 'lat_max':lat_max, 'lon_max':lon_max, 'nlat':nlat, 'nlon':nlon,
             'dlat':(lat_max-lat_min)/nlat, 'dlon':(lon_max-lon_min)/nlon,
             'lat':lat_min + (lat_max-lat_min)*np.arange(nlat+1)/float(nlat),
             'lon':lon_min + (lon_max-lon_min)*np.arange(nlon+1)/float(nlon) }

    if dim == 2 : # create also coordinate 2d-fields
        if first_dim == 'lon' :
            lon2d, lat2d = np.meshgrid( r['lon'], r['lat'], sparse=False, indexing='ij' )
            # lon changes with first index, lat with second
        elif first_dim == 'lat' :
            lat2d, lon2d = np.meshgrid( r['lat'], r['lon'], sparse=False, indexing='ij' )
            # lat changes with first index, lon with second
        else :
            raise ValueError('generate_latlon_grid: I do not understand first_dim='+first_dim)

        r.update({ 'lon2d':lon2d, 'lat2d':lat2d })

    return r

#-----------------------------------------------------------------------------------------------------------------------
def get_subdomain( modelstatefile, gridfile, variables,
                   lat_min = 43.03 * np.pi/180, lat_max = 58.16 * np.pi/180,
                   lon_min = -4.15 * np.pi/180, lon_max = 20.50 * np.pi/180,
                   time_index=0, skip_layers_top=0, verbose=True ) :
    """Read specified variables from specified part of model state and grid"""

    # full ICON-D2 model grid: -4.139700 < lon <20.491785, 43.035024 < lat < 58.151829

    # determine subdomain grid .........................................................................................

    if verbose : print('[get_subdomain] opening horizontal grid file {}...'.format(gridfile))

    grid_full = netCDF4.Dataset( args.gridfile,'r')
    if verbose :
        print('    full model grid : %f < lon <%f, %f < lat < %f' % (
            np.array(grid_full.variables['vlon']).min()*180/np.pi, np.array(grid_full.variables['vlon']).max()*180/np.pi,
            np.array(grid_full.variables['vlat']).min()*180/np.pi, np.array(grid_full.variables['vlat']).max()*180/np.pi ))

    print('*** constructing subdomain grid'); starttime = time.perf_counter()

    nvertices_full = len(grid_full.dimensions['vertex'])
    ncells_full    = len(grid_full.dimensions['cell'])

    cell_indices             = np.zeros( ncells_full,    dtype=np.int32 ) - 1
    vertex_indices           = np.zeros( nvertices_full, dtype=np.int32 ) - 1
    translate_cell_indices   = np.zeros( ncells_full,    dtype=np.int32 ) - 1
    translate_vertex_indices = np.zeros( nvertices_full, dtype=np.int32 ) - 1

    print('    determining subdomain indices for %f <= lat < %f, %f <= lon < %f' % (lat_min*180/np.pi, lat_max*180/np.pi, lon_min*180/np.pi, lon_max*180/np.pi))

    clon                = np.array(grid_full.variables['clon'])
    clat                = np.array(grid_full.variables['clat'])
    vlon                = np.array(grid_full.variables['vlon'])
    vlat                = np.array(grid_full.variables['vlat'])
    cell_area           = np.array(grid_full.variables['cell_area'])
    vertex_of_cell      = np.array(grid_full.variables['vertex_of_cell'])      - 1
    neighbor_cell_index = np.array(grid_full.variables['neighbor_cell_index']) - 1
    cells_of_vertex     = np.array(grid_full.variables['cells_of_vertex'])      - 1
    ncells, nvertices = subdomain_indices( clon, clat, vlon, vlat, vertex_of_cell,
                                           lat_min, lat_max, lon_min, lon_max,
                                           cell_indices, vertex_indices, translate_cell_indices, translate_vertex_indices )
    print('    subdomain contains %d of %d cells and %d of %d vertices' % ( ncells, ncells_full, nvertices, nvertices_full ))
    print('    sqrt(area) of first cell : {:.0f}m'.format(np.sqrt(cell_area[0])))

    print('    converting horizontal grid...')
    grid = dict()
    grid['ncells']    = ncells
    grid['nvertices'] = nvertices
    grid['clat'] = clat[cell_indices[:ncells]]
    grid['clon'] = clon[cell_indices[:ncells]]
    grid['cell_area'] = cell_area[cell_indices[:ncells]]
    grid['vlat'] = vlat[vertex_indices[:nvertices]]
    grid['vlon'] = vlon[vertex_indices[:nvertices]]
    grid['neighbor_cell_index'] = translate_cell_indices[ neighbor_cell_index[:,cell_indices[:ncells]] ]
    grid['vertex_of_cell']      = translate_vertex_indices[ vertex_of_cell[:,cell_indices[:ncells]] ]
    grid['cells_of_vertex']      = translate_cell_indices[ cells_of_vertex[:,vertex_indices[:nvertices]] ]
    grid['full_grid_cell_indices'] = cell_indices[:ncells]

    print('*** subdomain grid construction took %f seconds' % (time.perf_counter() - starttime))

    # open model state file ...........................................................................................

    if verbose : print('[get_subdomain] opening model state file {}...'.format(modelstatefile))

    try :
        modelstate = netCDF4.Dataset( modelstatefile, 'r')
        print('    ...which is a NetCDF file...')
        modelvars_available = list(modelstate.variables.keys())
        nz_full = len(modelstate.dimensions['height'])
        ftype='netcdf'

    except :
        if verbose : print('    ...which seems to be a GRIB file...')
        from kendapy.cosmo_state import CosmoState
        modelstate = CosmoState( modelstatefile )
        varlist = modelstate.list_variables(retval=True)
        modelvars_available = list(varlist.keys())

        # determine number of layers
        nz_full = -1
        for v in modelvars_available :
            levtypes =  list(varlist[v].keys())
            for lt in levtypes :
                if lt.endswith('Layer') :
                    steps = list(varlist[v][lt].keys())
                    nz_full = len( varlist[v][lt][steps[0]]['levels'] )
                    break
            if nz_full > 0 : break
        ftype='grib'

    if verbose :
        print('    available variables : ', modelvars_available)
        print('    number of layers    : ', nz_full)


    # list available output times ......................................................................................

    if ftype == 'netcdf' :
        outputtimes = modelstate.variables['time'][:]
        print('    output times available :')
        for ti, ot in enumerate(outputtimes) :
            ot_date = int(ot)
            ot_time = ot-ot_date
            ot_hour = int(ot_time*24.0)
            ot_min  = int((ot_time - ot_hour/24.0)*60.0)
            print('      (', ti, ') --- ',  ot_date, ot_hour, ot_min)
        print('    selected time index : ', time_index)
    else :
        if time_index > 0 :
            raise ValueError( 'time index > 0 probably not yet supported for grib files...' )


    # read subdomain model variables ...................................................................................

    if grid['ncells'] <= 0 :

        print('    subdomain does not contain any model grid cells... ', end=' ')
        modelvars = {}

    else :

        print('    extracting model data... ', end=' ')

        # dimension check ..............................................................................................

        if ftype == 'netcdf' :
            ncells_full_model    = len(modelstate.dimensions['ncells'])
            if ncells_full_model != ncells_full :
                print('ERROR: ncells mismatch between grid and model data', ncells_full_model, ncells_full)
        else :
            # we cannot check this easily for grib files -> assume everything is ok...
            ncells_full_model = ncells_full

        # determine vertical part to be read ...........................................................................

        nz = nz_full - skip_layers_top
        print('    using %d of %d layers... ' % ( nz, nz_full ))


        # read variables ...............................................................................................
        
        print('    available variables : ', modelvars_available)
        print('    variables to be read: ', ', '.join(variables))

        if type(variables) != list :
            variables = [variables]

        read_in_chunks = True # is faster...

        modelvars = dict()
        for v in variables :
            if '.' in v :
                vfile, vname = v.split('.')
                print('        - reading [{}->{}]'.format(vfile,vname), end=' ')
            else :
                vname = v
                vfile = v
                print('        - reading [{}]'.format(vname), end=' ')

            kl = skip_layers_top
            if vname != 'HHL' :
                kh = kl + nz
            else :
                kh = kl + nz + 1

            if vfile in modelvars_available :
                if ftype == 'netcdf' :
                    if read_in_chunks :
                        modelvars[vname] = read_part_of_variable( modelstate, vfile, cell_indices[:ncells], time_index=args.time_index )[kl:kh,...]
                    else :
                        modelvars[vname] = modelstate.variables[vfile][args.time_index,:,cell_indices[:ncells]][kl:kh,...]
                else :
                    modelvars[vname] = np.transpose( modelstate[vfile][cell_indices[:ncells],:][...,kl:kh] )

            print(' with shape ', modelvars[vname].shape)

    print('[get_subdomain] done.')

    return grid, modelvars

#-----------------------------------------------------------------------------------------------------------------------
def read_part_of_variable( modelvars_full, vname, ci, check=False, time_index=None, verbose=False ) :

    if len(modelvars_full.variables[vname].shape) == 3 :
        has_timedim = True
    else :
        has_timedim = False

    nlevels = modelvars_full.variables[vname].shape[-2]
    ncells = ci.size
    ncells_full = modelvars_full.variables[vname].shape[-1]
    modelvar_part = np.zeros( (nlevels,ncells) ) # omit leading time dimension
    #ci = cell_indices[:ncells]

    starttime = time.perf_counter()

    nchunks = 50
    ncells_chunk = ncells_full/nchunks
    for ic in range(nchunks) :
        index_min = ic*ncells_chunk
        index_max = (ic+1)*ncells_chunk
        if ic == nchunks-1 : index_max = ncells_full

        idcs = np.where( (ci >= index_min) & (ci < index_max) )
        nrelevant = len(idcs[0])
        if verbose :
            print('   --- chunk %d [ %d <= index < %d ] : %d relevant' % (ic, index_min, index_max, nrelevant))
        if nrelevant > 0 :
            if has_timedim :
                chunk = np.array(modelvars_full.variables[vname][time_index,:,index_min:index_max])
            else :
                chunk = np.array(modelvars_full.variables[vname][:,index_min:index_max])
            chunk_idcs = np.array(ci[idcs],dtype=int)-index_min
            #print 'chunk shapes ', chunk.shape, chunk_idcs.shape, chunk_idcs.min(), chunk_idcs.max(), modelvar_part.shape
            #for k in range(nlevels) :
            #    modelvar_part[k+zeros(nrelevant,dtype=int),idcs] = chunk[k+zeros(nrelevant,dtype=int),chunk_idcs]
            #print 'L', modelvar_part[:,array(idcs,dtype=int)].shape
            #print 'R', chunk[:,chunk_idcs].shape
            #print 'I', asarray(idcs,dtype=int)[0,:].shape
            modelvar_part[:,np.asarray(idcs,dtype=int)[0,:]] = chunk[:,chunk_idcs]

    print('   --- reading and distributing chunks took %f seconds...' % (time.perf_counter() - starttime))

    if check :
        print('ok, checking...')
        if has_timedim :
            fullvar = np.array(modelvars_full.variables[vname][time_index,:,:])
        else :
            fullvar = np.array(modelvars_full.variables[vname][:,:])
        redvar = fullvar[:,ci]
        print('DEVIATION ', (modelvar_part-redvar).min(), (modelvar_part-redvar).max())
        fullvar = ''
        redvar = ''

    return modelvar_part

#-----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True,nogil=True)
def subdomain_indices( clon, clat, vlon, vlat, vertex_of_cell, lat_min, lat_max, lon_min, lon_max,
                       cell_indices, vertex_indices, translate_cell_indices, translate_vertex_indices ) :
    """Save the indices of the cells whose center lies within the given region
       and the indices of vertices forming these cells in cell_indices and vertex_indices.
       Returns number of cells and number of vertices."""

    icell = 0
    for i in range(clon.size) :
        if (clon[i] >= lon_min) and (clon[i] < lon_max) and (clat[i] >= lat_min) and (clat[i] < lat_max) :
          cell_indices[icell] = i
          translate_cell_indices[i] = icell
          icell += 1
          for ii in range(3) :
              if vertex_of_cell[ii,i] > -1 :
                  vertex_indices[ vertex_of_cell[ii,i] ] = 1 # mark as required

    ivertex = 0
    for i in range(vlon.size) :
        if vertex_indices[i] > 0 :
            vertex_indices[ivertex] = i
            translate_vertex_indices[i] = ivertex
            ivertex += 1

    return icell, ivertex

#-----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, nogil=True)
def horizontal_slice( quan, zcoord, zvalue, missing=0.0 ) :
    """
    Extract variable quan at surface defined by zcoord == zvalue,
    where quan and zcoord are numpy arrays with dimensions (nz,ncells)
    """

    nz, ncells = quan.shape
    qslice = np.zeros((ncells),dtype=quan.dtype)   

    if zcoord[1,0] > zcoord[0,0] :
        zvalue_ = zvalue
        zcoord_ = zcoord
    else :
        zvalue_ = -zvalue
        zcoord_ = -zcoord
    # -> values in zcoord_ are increasing with first index

    k = 0
    for i in range(ncells) :
        # find k with zcoord_[k,i] < zvalue_ < zcoord_[k+1,i]
        if zcoord_[k,i] < zvalue_ :
            if zvalue_ <= zcoord_[k+1,i] :
                qslice[i] = quan[k,i] + (quan[k+1,i]-quan[k,i]) * (zvalue_-zcoord_[k,i])/(zcoord_[k+1,i]-zcoord_[k,i])
            else :
                # k is too low
                while zcoord_[k+1,i] < zvalue_ and k < nz - 2 :
                    k += 1
                if zcoord_[k+1,i] < zvalue_ :
                    qslice[i] = missing
                else :
                    qslice[i] = quan[k,i] + (quan[k+1,i]-quan[k,i]) * (zvalue_-zcoord_[k,i])/(zcoord_[k+1,i]-zcoord_[k,i])
        else :
            # k is too high
            while zcoord_[k,i] >= zvalue_ and k > 0 :
                k -= 1
            if zcoord_[k,i] > zvalue_ :
                qslice[i] = missing
            else :
                qslice[i] = quan[k,i] + (quan[k+1,i]-quan[k,i]) * (zvalue_-zcoord_[k,i])/(zcoord_[k+1,i]-zcoord_[k,i])       

    return qslice

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
#-----------------------------------------------------------------------------------------------------------------------

    import argparse
    parser = define_parser()
    args = parser.parse_args()

    # full ICON-D2 model grid -4.139700 < lon <20.491785, 43.035024 < lat < 58.151829
    lat_min = 43.03
    lat_max = 58.16
    lon_min = -4.15
    lon_max = 20.50

    variables_to_read = args.variables.split(',')
    if not args.hslice is None :
        variables_to_read += [ args.hslice.split('=')[0] ]

    print('*** reading model state'); starttime = time.perf_counter()
    grid, modelvars = get_subdomain( args.modelvarsfile, args.gridfile, variables_to_read, verbose=args.verbose )
    print('*** reading model state took %f seconds' % (time.perf_counter() - starttime))

    if args.verbose :
            print('full model grid : %f < lon <%f, %f < lat < %f' % (
                np.array(grid['vlon']).min()*180/np.pi, np.array(grid['vlon']).max()*180/np.pi,
                np.array(grid['vlat']).min()*180/np.pi, np.array(grid['vlat']).max()*180/np.pi ))

            print('-'*20, 'grid', '-'*20)
            for k in grid :
                print( k, grid[k].shape if type(grid[k]) == np.ndarray else grid[k] )
            
            print('-'*20, 'model variables', '-'*20)
            for k in modelvars :
                print( k, modelvars[k].shape if type(modelvars[k]) == np.ndarray else modelvars[k] )
            print()

    if args.rotated :
        # Use rotated COSMO-D2 grid (optionally oversampled) as target
        rgriddef = cosmo_grid( configuration='COSMO-D2', definition=True )

        # remove boundary pixels
        nbnd = 2
        rgriddef['latitudeOfFirstGridPointInDegrees']   += nbnd * rgriddef['iDirectionIncrementInDegrees']
        rgriddef['longitudeOfFirstGridPointInDegrees']  += nbnd * rgriddef['jDirectionIncrementInDegrees']
        rgriddef['nlat']                                -= 2 * nbnd
        rgriddef['nlon']                                -= 2 * nbnd

        # oversample
        rgriddef['iDirectionIncrementInDegrees'] /= args.oversample
        rgriddef['jDirectionIncrementInDegrees'] /= args.oversample
        rgriddef['nlat']                         *= args.oversample
        rgriddef['nlon']                         *= args.oversample

        if args.verbose :
            print('target grid definition: ', rgriddef)

        if False : # compare unstructured grid with COSMO-D2 grid
            lat2d, lon2d = cosmo_grid( configuration='COSMO-D2' )
            fig, ax = plt.subplots(figsize=(10,10))            
            ax.scatter( grid['vlon']*180/np.pi, grid['vlat']*180/np.pi, marker='.', color='r', label='UNSTRUCTERED')
            ax.scatter( lon2d,lat2d, s=1, marker='x', color='k', label='ROTLATLON')
            ax.legend()
            fig.savefig('gridpoints.png')

        # convert grid definition to format compatible with tri2latlon
        latlon_grid = { 'lat_min':rgriddef['latitudeOfFirstGridPointInDegrees'],
                        'lon_min':(rgriddef['longitudeOfFirstGridPointInDegrees'] - 360.0),
                        'dlat':rgriddef['iDirectionIncrementInDegrees'],
                        'dlon':rgriddef['jDirectionIncrementInDegrees'],
                        'nlat':rgriddef['nlat'],
                        'nlon':rgriddef['nlon'] }

        # compute rotated lat-lons for triangular grid
        tri_grid = {}
        tri_grid['clat'], tri_grid['clon'] = nonrot_to_rot_grid( grid['clat']*180/np.pi, grid['clon']*180/np.pi, configuration=rgriddef )
        tri_grid['vlat'], tri_grid['vlon'] = nonrot_to_rot_grid( grid['vlat']*180/np.pi, grid['vlon']*180/np.pi, configuration=rgriddef )
        tri_grid['vertex_of_cell'] = grid['vertex_of_cell']
        extent=None

        # map non-rotated coordinates on rotated grid (for coordinate grid in plots)
        clat = tri2latlon( latlon_grid, tri_grid, grid['clat'], method='fine' )
        clon = tri2latlon( latlon_grid, tri_grid, grid['clon'], method='fine' )

        if args.verbose :
            print('nonrotated  triangle grid: {} < lat < {}, {} < lon < {}'.format( grid['vlat'].min()*180/np.pi, grid['vlat'].max()*180/np.pi,
                                                                                   grid['vlon'].min()*180/np.pi, grid['vlon'].max()*180/np.pi ))
            print('rotated     triangle grid: {} < lat < {}, {} < lon < {}'.format( tri_grid['vlat'].min(), tri_grid['vlat'].max(),
                                                                                   tri_grid['vlon'].min(), tri_grid['vlon'].max() ))

            if False : # check backtransformation
                bck_grid = {}
                bck_grid['clat'], bck_grid['clon'] = rot_to_nonrot_grid( tri_grid['clat'], tri_grid['clon'], configuration=rgriddef )
                bck_grid['vlat'], bck_grid['vlon'] = rot_to_nonrot_grid( tri_grid['vlat'], tri_grid['vlon'], configuration=rgriddef )
                print('backrotated triangle grid: {} < lat < {}, {} < lon < {}'.format( bck_grid['vlat'].min(), bck_grid['vlat'].max(),
                                                                                        bck_grid['vlon'].min(), bck_grid['vlon'].max() ))

            print('quad                 grid: {} < lat < {}, {} < lon < {}'.format( latlon_grid['lat_min'], latlon_grid['lat_min']+latlon_grid['nlat']*latlon_grid['dlat'],
                                                                                   latlon_grid['lon_min'], latlon_grid['lon_min']+latlon_grid['nlon']*latlon_grid['dlon'] ))
    else :
        # Use non-rotated lat-lon grid
        nlat = int( (lat_max-lat_min) * args.pixperdeg )
        nlon = int( (lon_max-lon_min) * args.pixperdeg * 0.5*( np.cos(lat_max*np.pi/180) + np.cos(lat_min*np.pi/180) ) )
        print('    lat-lon grid pixel size : nlon={} x nlat={}'.format(nlon,nlat) )
        print('*** calling generate_lat_lon_grid'); starttime = time.perf_counter()
        latlon_grid  = generate_latlon_grid( lat_min * np.pi/180, lon_min * np.pi/180, lat_max * np.pi/180, lon_max * np.pi/180, nlat, nlon )
        print('*** generate_lat_lon_grid took %f seconds' % (time.perf_counter() - starttime))
        tri_grid = grid
        extent=( lon_min, lon_max, lat_min, lat_max)

        if args.verbose :
            print('triangle grid: {} < lat < {}, {} < lon < {}'.format( tri_grid['vlat'].min(), tri_grid['vlat'].max(),
                                                                        tri_grid['vlon'].min(), tri_grid['vlon'].max() ))
            print('quad     grid: {} < lat < {}, {} < lon < {}'.format( latlon_grid['lat_min'], latlon_grid['lat_min']+latlon_grid['nlat']*latlon_grid['dlat'],
                                                                        latlon_grid['lon_min'], latlon_grid['lon_min']+latlon_grid['nlon']*latlon_grid['dlon'] ))

    for v in args.variables.split(',') :
        print('>>> plotting {}...'.format(v))

        if args.hslice is None :
            var2d = modelvars[v][args.level,:]
        else :
            zcoord_name, zvalue = args.hslice.split('=')
            zvalue = float(zvalue)
            # interpolate variable v on zcoord==zvalue
            print('*** calling horizontal_slice'); starttime = time.perf_counter()
            var2d  = horizontal_slice( modelvars[v], modelvars[zcoord_name], zvalue )
            print('*** horizontal_slice took %f seconds' % (time.perf_counter() - starttime))

        # map variable from triangular to (rotated or non-rotated) latlon grid
        print('*** calling tri2latlon'); starttime = time.perf_counter()
        var_latlon = tri2latlon( latlon_grid, tri_grid, var2d, method='fine' )
        print('*** tri2latlon took %f seconds' % (time.perf_counter() - starttime))


        # plot mapped variable
        fig, ax = plt.subplots(figsize=(10,10))
        mpb = ax.imshow( var_latlon[::-1,:], vmin=args.vmin, vmax=args.vmax, extent=extent, cmap=args.cmap ) #(left, right, bottom, top)
        plt.colorbar( mpb, shrink=0.5)

        # add lat-lon grid        
        if args.rotated :
            # FIXME does not look nice, artifacts near boundaries
            ax.contour( clat[::-1,:]*180/np.pi, np.arange(40,60,5), colors='w', alpha=0.25, linewidths=0.5 )
            ax.contour( clon[::-1,:]*180/np.pi, np.arange(0,25,5),  colors='w', alpha=0.25, linewidths=0.5 )                
        else :
            ax.grid(alpha=0.25, color='w')
        
        fig.savefig( '{}_{}.{}'.format( v, 'rotlatlon' if args.rotated else 'latlon', args.image_type ), bbox_inches='tight')

