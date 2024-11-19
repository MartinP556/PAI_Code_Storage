#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . M A T C H _ O B S E R V A T I O N S
#  For every observation in a EKF file, try to find the corresponding observation in a FOF file
#
#  2018.2 L.Scheck

from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import netCDF4
import argparse

parser = argparse.ArgumentParser(description='Match observations in FOF and EKF files (input and output of the same analysis)')
parser.add_argument( '-e', '--ekf', dest='ekf_fname', help='EKF file name' )
parser.add_argument( '-f', '--fof', dest='fof_fname', help='FOF file name' )
parser.add_argument( '-V', '--variable', dest='variable', help='variable' )
parser.add_argument( '-T', '--assint', dest='assint', default=60, help='assimilation interval [min]', type=int )
parser.add_argument( '-O', '--obstype', dest='obstype', default='', help='observation type [default: detect from file name]' )
parser.add_argument( '-p', '--plot', dest='plot', help='plot results', action='store_true' )
#    parser.add_argument( 'ekffile', metavar='ekffile', help='ekf file name[s]', nargs='*' )
args = parser.parse_args()

if args.obstype == '' :
    obstype = args.ekf_fname.split('/')[-1].split('ekf')[-1].split('_')[0]
else :
    obstype = args.obstype
varnames = { 0:'NUM', 3:'U', 4:'V', 8:'W', 1:'Z', 57:'DZ', 9:'PWC', 28:'TRH', 29:'RH', 58:'RH2M', 2:'T', 59:'TD',
             39:'T2M', 40:'TD2M', 11:'TS', 30:'PTEND', 60:'W1', 61:'WW', 62:'VV', 63:'CH', 64:'CM', 65:'CL', 66:'NHcbh',
             67:'NL', 93:'NM',94:'NH', 69:'C', 70:'NS', 71:'SDEPTH', 72:'E', 79:'TRTR', 80:'RR', 81:'JJ', 87:'GCLG',
             91:'N', 92:'SFALL', 110:'PS', 111:'DD', 112:'FF', 118:'REFL', 119:'RAWBT', 120:'RADIANCE', 41:'U10M',
             42:'V10M', 7:'Q', 56:'VT', 155:'VN', 156:'HEIGHT', 157:'FLEV', 192:'RREFL', 193:'RADVEL', 128:'PDELAY',
             162:'BENDANG', 252:'IMPPAR', 248:'REFR', 245:'ZPD', 246:'ZWD', 247:'SPD', 242:'GUST', 251:'P', 243:'TMIN' }
varno = -1
for v in varnames :
    if varnames[v] == args.variable :
        varno = v
        break
if varno == -1 :
    raise ValueError('Unknown variable')
print(('match_observations: comparing %s/%s(%d) in %s with %s...' % (obstype, args.variable, varno, args.fof_fname, args.ekf_fname)))
print()

# open NetCDF files ....................................................................................................
ekf = netCDF4.Dataset( args.ekf_fname, 'r')
fof = netCDF4.Dataset( args.fof_fname, 'r')
print('EKF HEADER ____________________________________________________________________')
for k in ekf.ncattrs() :
    print(("--   %25s | %s" % (k, getattr(ekf, k))))
print('FOF HEADER____________________________________________________________________')
for k in fof.ncattrs() :
    print(("--   %25s | %s" % (k, getattr(fof, k))))
print()

# identify EKF observations ............................................................................................

print('match_observations: identifying observations in EKF file...')
time_shift = args.assint # compensate for different reference times
ekf_nobs = {}
ekf_first = {}
f = ekf
n_active = 0
to_be_removed=[]
for j in range(ekf.n_hdr) :
    if f.variables['r_state'][j] == 1 : # ACTIVE
        i_body = f.variables['i_body'][j] - 1 # <-- FORTRAN INDEX CONVENTION!
        l_body = f.variables['l_body'][j]
        for i in range(i_body,i_body+l_body) :
            if f.variables['state'][i] == 1 and f.variables['varno'][i] == varno : # ACTIVE
                oid = "%d,%d,%d,%d,%09.5f,%09.5f,%d,%f" % ( f.variables['obstype'][j], f.variables['codetype'][j],
                                                            f.variables['instype'][j], f.variables['time'][j]+time_shift,
                                                            f.variables['lat'][j], f.variables['lon'][j],
                                                            f.variables['varno'][i], f.variables['level'][i] )
                # (varno & level are body arrays -> index i, not j)

                n_active += 1
                if oid in ekf_nobs :
                    ekf_nobs[oid] += 1
                    print(('non-unique : ', i,              oid, f.variables['obs'][i]))
                    print(('             ', ekf_first[oid], oid, f.variables['obs'][ekf_first[oid]]))
                    to_be_removed.append(oid)
                else :
                    ekf_nobs[oid] = 1
                    ekf_first[oid] = i

idcs_nu = np.where( np.array(list(ekf_nobs.values())) > 1 )
n_nu_ekf = np.array(list(ekf_nobs.values()))[idcs_nu].sum()
print(('observations with non-unique identifiers in EKF file : %d of %d' % ( n_nu_ekf, n_active )))

# remove non-unique identifiers
for oid in to_be_removed :
    ekf_nobs.pop(oid)

print(('observations left after removing non-unique : ', len(ekf_nobs)))

# match FOF observations ...............................................................................................

time_shift = 0
fof_nobs = {}
fof_first = {}
f = fof
n_tested = 0
n_unknown = 0
matches = {}
for j in range(f.n_hdr) :
    if f.variables['r_state'][j] == 1 : # ACTIVE
        i_body = f.variables['i_body'][j] - 0 # <-- C INDEX CONVENTION !
        l_body = f.variables['l_body'][j]
        for i in range(i_body,i_body+l_body) :
            if f.variables['state'][i] == 1 and f.variables['varno'][i] == varno : # ACTIVE
                oid = "%d,%d,%d,%d,%09.5f,%09.5f,%d,%f" % ( f.variables['obstype'][j], f.variables['codetype'][j], f.variables['instype'][j],
                                                            f.variables['time'][j]+time_shift, f.variables['lat'][j], f.variables['lon'][j],
                                                            f.variables['varno'][i], f.variables['level'][i] )
                if oid in fof_nobs :
                    fof_nobs[oid] += 1
                else :
                    fof_nobs[oid] = 1
                    fof_first[oid] = i

                n_tested += 1
                if oid in ekf_nobs :
                    if oid in matches :
                        matches[oid].append(i)
                    else :
                        matches[oid] = [i]
                else :
                    n_unknown += 1

idcs_nu = np.where( np.array(list(fof_nobs.values())) > 1 )
print(('observations with non-unique identifiers in FOF file : %d of %d (%d not in EKF)' % ( np.array(list(fof_nobs.values()))[idcs_nu].sum(), n_tested, n_unknown )))

n_multi_matches = 0
for m in matches :
    if len(matches[m]) > 1 :
        n_multi_matches += 1
print(('matched EKF observations : %d out of %d (%d multi matches)' % ( len(matches), len(ekf_nobs), n_multi_matches )))

# compare observation values ...........................................................................................

obs_ekf = np.zeros(len(matches))
obs_fof = np.zeros(len(matches))
bcor_ekf = np.zeros(len(matches))
bcor_fof = np.zeros(len(matches))
i=0
for oid in matches :
    i_fof = matches[oid][0]
    i_ekf = ekf_first[oid]
    obs_ekf[i] = ekf.variables['obs'][i_ekf]
    obs_fof[i] = fof.variables['obs'][i_fof]
    bcor_ekf[i] = ekf.variables['bcor'][i_ekf]
    bcor_fof[i] = fof.variables['bcor'][i_fof]
    i+=1
print(('EKF obs     min/mean/max : ', obs_ekf.min(), obs_ekf.mean(), obs_ekf.max(), ' mean abs : ', abs(obs_ekf).mean(), ' max. |bcor| : ', abs(bcor_ekf).max()))
print(('FOF obs     min/mean/max : ', obs_fof.min(), obs_fof.mean(), obs_fof.max(), ' mean abs : ', abs(obs_fof).mean(), ' max. |bcor| : ', abs(bcor_fof).max()))
dobs = abs(obs_ekf - obs_fof)
print(('|delta obs| min/mean/max : ', dobs.min(), dobs.mean(), dobs.max()))

# plot results .........................................................................................................

if args.plot :
    fig, ax = plt.subplots( 1, 2, figsize=(20,10))

    # EKF vs FOF observation value scatter plot
    omin = np.minimum( obs_ekf.min(), obs_fof.min() )
    omax = np.maximum( obs_ekf.max(), obs_fof.max() )
    ax[0].scatter( obs_ekf, obs_fof, s=10, alpha=0.25, color='k' )
    ax[0].plot( (omin,omax), (omin,omax), 'r', linewidth=0.3 )
    ax[0].set_xlabel('ekf %s/%s' % (obstype,args.variable))
    ax[0].set_ylabel('fof %s/%s' % (obstype,args.variable))
    ax[0].grid()
    xlm = ax[0].get_xlim()
    ylm = ax[0].get_ylim()
    font = FontProperties()
    boldfont = font.copy()
    boldfont.set_weight('bold')
    ax[0].text( xlm[0]+0.05*(xlm[1]-xlm[0]), ylm[0]+0.95*(ylm[1]-ylm[0]), '%s / %s' % (obstype,args.variable), fontsize=24,
                color='#999999', va='top', fontproperties=boldfont )
    ax[0].text( xlm[0]+0.05*(xlm[1]-xlm[0]), ylm[0]+0.90*(ylm[1]-ylm[0]), 'EKF : ' + args.ekf_fname.split('/')[-1], va='top' )
    ax[0].text( xlm[0]+0.05*(xlm[1]-xlm[0]), ylm[0]+0.87*(ylm[1]-ylm[0]), 'FOF : ' + args.fof_fname.split('/')[-1], va='top' )
    ax[0].text( xlm[0]+0.05*(xlm[1]-xlm[0]), ylm[0]+0.84*(ylm[1]-ylm[0]),
                'matched %d of %d obs. (ignored %d non-unique)' % ( len(matches), len(ekf_nobs), n_nu_ekf), va='top' )
    ax[0].text( xlm[0]+0.05*(xlm[1]-xlm[0]), ylm[0]+0.81*(ylm[1]-ylm[0]), 'mean deviation : %f' % dobs.mean(), va='top' )

    # spatial distribution plot
    ax[1].scatter( fof.variables['lon'][:fof.n_hdr], fof.variables['lat'][:fof.n_hdr], s=30, alpha=0.2, color='b', label='FOF' )
    ax[1].scatter( ekf.variables['lon'], ekf.variables['lat'], s=15, alpha=0.2, color='r', label='EKF '+obstype )
    ax[1].legend( frameon=False)
    ax[1].set_xlabel('lon')
    ax[1].set_ylabel('lat')

    fig.savefig( 'match_observations_%s_%s.png' % (obstype,args.variable), bbox_inches='tight' )

