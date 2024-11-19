#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . G R I D _ T O O L S
#
#  2019.3 L.Scheck

from numba import jit, float64, float32, int32
import numpy as np

@jit(nopython=True, nogil=True)
def map_to_z_bins(qmodel, zmodel, zbins) :
    """
    Map 3d model variable qmodel with associated interface heights zmodel onto a grid with the
    same horizontal structure but a fixed vertical grid defined by the interface heights in zbins.

    heights in zmodel and zbins should decrease with the index (COSMO: zmodel=HHL)
    vertical dimension should be the last dimension in qmodel
    """

    nx, ny, nz = qmodel.shape
    nb         = zbins.size-1
    qzbins     = np.zeros((nx,ny,nb))

    for i in range(nx):
        for j in range(ny):

            klo = 0
            for m in range(nb) : # loop over z bins

                # initialize with NaN -> we can later identify z-grid cells that had no overlap with model grid cells
                qzbins[i,j,m] = np.nan

                if zbins[m+1] < zmodel[i,j,0] and zbins[m] > zmodel[i,j,nz] : # is bin m overlapping any model cells?

                    # find first (geometrically highest -> lowest index "klo") model cell overlapping bin m
                    while zmodel[i,j,klo+1] > zbins[m] and klo < nz:
                        klo += 1

                    # find last (geometrically lowest -> highest index "khi") model cell overlapping bin m
                    khi = klo
                    while zmodel[i,j,khi+1] > zbins[m+1] and khi < nz-1 :
                        khi +=1

                    if klo == khi : # bin m is contained in model cell klo=khi

                        zblo = zbins[m]
                        if zmodel[i,j,klo] < zbins[m] : # upper end of bin m is higher than the highest model grid cell
                            zblo = zmodel[i,j,klo]

                        zbhi = zbins[m+1]
                        if zmodel[i,j,klo+1] > zbins[m+1] : # lower end of bin m is lower than the lowest model grid cell
                            zbhi = zmodel[i,j,klo+1]

                        qzbins[i,j,m] = (zblo-zbhi) * qmodel[i,j,klo]

                    else :
                        dz_lo = zbins[m] - zmodel[i,j,klo+1]
                        if zmodel[i,j,klo] < zbins[m] : # upper end of bin m is higher than the highest model grid cell
                            dz_lo = zmodel[i,j,klo] - zmodel[i,j,klo+1]

                        dz_hi = zmodel[i,j,khi] - zbins[m+1]
                        if zmodel[i,j,khi+1] > zbins[m+1] : # lower end of bin m is lower than the lowest model grid cell
                            dz_hi = zmodel[i,j,khi] - zmodel[i,j,khi+1]

                        qzbins[i,j,m] = dz_lo * qmodel[i,j,klo] + dz_hi * qmodel[i,j,khi]

                        # add full cells  between klo and khi
                        if khi > klo+1 :
                            qzbins[i,j,m] += ((zmodel[i,j,klo+1:khi]-zmodel[i,j,klo+2:khi+1]) * qmodel[i,j,klo+1:khi]).sum()
    return qzbins


#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # tests

    from kendapy.cosmo_state import CosmoState
    import matplotlib.pyplot as plt
    import sys

    # argument should be path to a grib file containing HHL and QV, e.g.
    # /project/meteo/work/Leonhard.Scheck/userdata/grib_examples/grib1_kenda/lff20140516080000.det

    cs = CosmoState( sys.argv[1] )

    if True : # compute LWC(z), recompute TCWV etc.

        dz = 250 # m
        zbins = np.arange(0,12001,dz)[::-1]

        # append dimension to 2d area field -> can be used to multiply with 3d variables
        area3d = cs['AREA'].reshape(list(cs['AREA'].shape)+[1])

        # compute total volume as a function of height (varies only because there is orography)
        vol_z     = np.nansum(np.nansum(map_to_z_bins(np.ones(cs['QV'].shape) * area3d, cs['HHL'], zbins), axis=0), axis=0)

        # compute mean densities for water vapor, cloud water and cloud ice
        wv_mass_z = np.nansum(np.nansum(map_to_z_bins(cs['QV'] * cs['RHO'] * area3d, cs['HHL'], zbins), axis=0), axis=0) / vol_z
        cw_mass_z = np.nansum(np.nansum(map_to_z_bins(cs['QC'] * cs['RHO'] * area3d, cs['HHL'], zbins), axis=0), axis=0) / vol_z
        ci_mass_z = np.nansum(np.nansum(map_to_z_bins(cs['QI'] * cs['RHO'] * area3d, cs['HHL'], zbins), axis=0), axis=0) / vol_z

        fig, ax = plt.subplots(figsize=(10,10))
        ax.semilogx( wv_mass_z, 0.5*(zbins[1:]+zbins[:-1]), 'k', label='water vapor' )
        ax.semilogx( cw_mass_z, 0.5*(zbins[1:]+zbins[:-1]), 'r', label='cloud water' )
        ax.semilogx( ci_mass_z, 0.5*(zbins[1:]+zbins[:-1]), 'b', label='cloud ice' )
        ax.semilogx( 1e-2*vol_z/vol_z.max(), 0.5*(zbins[1:]+zbins[:-1]), '--g' )
        ax.set_xlim((1e-7,1e-1))
        ax.set_xlabel('mean water content [kg/m3]')
        ax.set_ylabel('height [m]')
        ax.legend(frameon=False)
        fig.savefig('WVC_LWC_IWC_vs_z.png')
        plt.close(fig)

        # recompute TQC
        tqc = np.nansum(map_to_z_bins(cs['QC'] * cs['RHO'], cs['HHL'], zbins), axis=2)
        tqc2 = (cs['QC']*(cs['HHL'][:,:,:-1]-cs['HHL'][:,:,1:])*cs['RHO']).sum(axis=2)
        print( 'mean TQC : model output {} kg/m2, recomputed on z-grid {} kg/m2, recomputed on model grid {} kg/m2'.format( cs['TQC'].mean(), tqc.mean(), tqc2.mean() ) )

        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow( np.log10(cs['TQC']+1e-8),vmin=-4,vmax=1,cmap='jet')
        ax[1].imshow( np.log10(tqc+1e-8),vmin=-4,vmax=1,cmap='jet')
        mpb=ax[2].imshow(cs['TQC']-tqc, vmin=-0.03,vmax=0.03,cmap='RdBu')
        plt.colorbar(mpb,shrink=0.5)
        fig.savefig('TQC.png')


    if True : # basic functionality test

        for dz in [100,5000] :

            # define z grid which fully contains the model grid
            zbins = np.arange(-5000,25001,dz)[::-1]

            print('Z GRID : dz = {}, {} bins'.format( dz, zbins.size-1 ) )

            # test 1 : compare vertical integrals \int Q*dz for the original grid and the new z grid
            qv_z = np.nansum(map_to_z_bins(cs['QV'], cs['HHL'], zbins), axis=2)
            qv_m = (cs['QV']*(cs['HHL'][:,:,:-1]-cs['HHL'][:,:,1:])).sum(axis=2)
            dqv = qv_z - qv_m
            print( 'Vertical WV integral : mean value for z grid = {}, mean value for model grid = {}, maximum error = {}'.format( qv_z.mean(), qv_m.mean(), abs(dqv).max() ))

            # test 2 : integrate over constant value 1 -> should result in vertical extent of model grid
            v_m = np.ones( cs['QV'].shape )
            v_z = map_to_z_bins(v_m, cs['HHL'], zbins)
            deltaz1 = np.nansum( v_z, axis=2)
            deltaz2 = cs['HHL'][:,:,0]-cs['HHL'][:,:,-1]
            print('Vertical extent: mean value for z grid = {}, mean value for model grid = {}, max. error = {}m'.format( deltaz1.mean(), deltaz2.mean(), abs(deltaz1-deltaz2).max() ))
            print('')

        for dz in [50,500] :

            # define z grid covering 500m - 5km
            zbins = np.arange(500,5001,dz)[::-1]

            qv_z = np.nansum(map_to_z_bins(cs['QV'], cs['HHL'], zbins))
            print( 'Full Q*dz integral over layer with dz={} : {}'.format(dz,qv_z) )


