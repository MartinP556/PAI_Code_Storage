from kendapy.experiment import *
from pylab import *
xp = Experiment('/home/userdata/leo.scheck/bacy_ahutt/cosmo_letkf/logfiles/run_cycle_1000.05b.log',dwd2lmu=True)

# get cosmo state of the first guess and analysis ensemble mean in the last step of the experiment
am = xp.get_cosmo( xp.veri_times[-1], prefix='laf', suffix='mean' )
fm = xp.get_cosmo( xp.fcst_start_times[-1], prefix='lff', suffix='mean' )

# list available variables
fm.list_variables()

binedges = arange(100,1001,50)*100 # [Pa]
print('pressure bin edges : ', binedges)
tprof_am, bincenters, npoints = am.distribution( 'PML', 't', binedges=binedges )
tprof_fm, bincenters, npoints = fm.distribution( 'PML', 't', binedges=binedges )

figure(10)
clf()
plot( tprof_fm['mean'],                   bincenters/100, 'b' )
plot( tprof_fm['mean']+tprof_fm['std']/2, bincenters/100, 'b', linewidth=0.5 )
plot( tprof_fm['mean']-tprof_fm['std']/2, bincenters/100, 'b', linewidth=0.5 )
plot( tprof_am['mean'],                   bincenters/100, '--r' )
ylim((1000,100))
ylabel('pressure [hPA]')
xlabel('temperature [K]')


# plot orography + lat/lon grid
figure(1)
clf()
contourf( am['HSURF'], arange(-250,4500,250), cmap='terrain' ) # color maps: http://matplotlib.org/examples/color/colormaps_reference.html
colorbar()
contour( am['RLAT'], arange(0,90,1), colors='w', linestyles=':' )
contour( am['RLON'], arange(0,30,1), colors='w', linestyles=':' )

# overplot SYNOP pressure observation locations
ekf = xp.get_ekf( xp.veri_times[-1], 'SYNOP' )
ilat, ilon = am.cosmo_indices( ekf['var']['PS']['lat'], ekf['var']['PS']['lon'] )
scatter( ilon, ilat, marker='x', color='k', edgecolor='' )

# restore correct plot limits
nlat, nlon = am['HSURF'].shape
xlim((0,nlon-1))
ylim((0,nlat-1))



# lowest level moisture plots for first guess and analysis

figure(2)
clf()
imshow( fm['q'][:,:,-1], origin='lower', cmap='YlGnBu' )
colorbar()
contour( fm['RLAT'], arange(0,90,5), colors='w', linestyles='--' )
contour( fm['RLON'], arange(0,30,5), colors='w', linestyles='--' )

figure(3)
clf()
imshow( am['q'][:,:,-1], origin='lower', cmap='YlGnBu' )
colorbar()
contour( am['RLAT'], arange(0,90,5), colors='w', linestyles='--' )
contour( am['RLON'], arange(0,30,5), colors='w', linestyles='--' )


# compute analysis moisture increments

dq = fm['q'] - am['q']
amp = abs(dq).max()/3

ilat = 100

figure(4)
clf()
imshow( dq[:,:,-1], vmin=-amp, vmax=amp, origin='lower', cmap='RdBu_r' )
colorbar()
contour( am['RLAT'], arange(0,90,5), colors='#999999', linestyles='--' )
contour( am['RLON'], arange(0,30,5), colors='#999999', linestyles='--' )
# add lowest level moisture contours
contour( am['q'][:,:,-1], levels=linspace(am['q'].min(),am['q'].max(),10), colors='k', linewidths=0.5 )
# mark location of cut in next plot
plot((0,nlon-1),(ilat,ilat),'--r')

# vertical cut at ilat = 100
figure(5)
clf()
imshow(  transpose(dq[ilat,:,:]), aspect=3, vmin=-amp, vmax=amp, origin='lower', cmap='RdBu_r' )


# pressure level overlay
phl = fm.compute('PHL')
# HML is not available, only HHL -> use approximation:
pml = 0.5*( phl[...,:-1] + phl[...,1:] )
contour( transpose(pml[ilat,:,:])/1e2, levels=[100,200,500,750,1000], colors='#999999', linewidths=0.5 )

# moisture overlay
contour( transpose(am['q'][ilat,:,:]), levels=linspace(am['q'].min(),am['q'].max(),10), colors='k', linewidths=0.5 )
ylim((dq.shape[2]-1,0))
