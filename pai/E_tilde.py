import os
#os.system(f'source env/bin/activate')
#from numba import njit
#os.system('deactivate')
os.chdir('/jetfs/home/a12233665/pai-munich-vienna/')
#Remember to load anaconda, enstools and maybe upgrade tbb or  NUMBA_THREADING_LAYER='omp' python PAI_Optimize_GC.py ???
import sys
sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/')
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime
from joblib import Parallel, delayed
from numba import njit
import xarray as xr
from pathlib import Path

from xgrads import open_CtlDataset, CtlDescriptor

from kendapy.ekf import Ekf
from enstools.io import read, write

#import pai.localization as loc
from pai.localization import get_dist_from_obs, gaspari_cohn, vertical_dist, haversine_distance
import pai.localization as loc
import pai.pai_utils as paiut
import pai.observation as oi
import pai.PAI_Optimize_GC as paiogc
import pai.partial_analysis_increment_np as PAI

sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/')
import Plotting.Plot_Functions as pf

import cProfile, pstats, io
from pstats import SortKey
import time
import copy



from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

def resilience(exp_path, dir_time_sonde, sonde_time, dir_time_ana, ana_time, rep_index, save_name, influencing_obs = 'RADAR', influencing_var = 'RREFL', BT_choice_influencer = 0, sat_var = 'REFL', BT_choice_sat = 0, RTPP = True, Title = 'Explanatory_Plot', surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file = True, read_diag_file = True, diag_set = []):
    obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, dir_time_ana, ana_time)
    obs_path_TEMP = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time_ana, ana_time)
    obs_path_TEMP1 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time_sonde, sonde_time)
    obs_path_TEMP2 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, ana_time, ana_time)
    obs_path_INFLUENCER = '{}/feedback/{}/ekf{}.nc.{}'.format(exp_path, dir_time_ana, influencing_obs, ana_time)
    ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, dir_time_ana, ana_time)
    hloc_path = '{}/feedback/{}/monDIAG.nc.{}.ctl'.format(exp_path, dir_time_ana, ana_time)
    print(hloc_path)

    if os.path.isfile(obs_path_TEMP2) == True:
        obs_sonde_T2 = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
    obs_sonde_T1 = paiut.get_ekf(obs_path_TEMP1, "T", active = False)
    if rep_index < len(obs_sonde_T1.reports()):
        rep = obs_sonde_T1.reports()[rep_index]
        obs_sonde_T = paiut.get_ekf(obs_path_TEMP1, "T", active = False)
    else:
        rep = obs_sonde_T2.reports()[rep_index - len(obs_sonde_T1.reports())]
        obs_sonde_T = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
    print(rep, rep_index)
    obs_sonde_T.add_filter(filter=f"report={rep}")
    obslon = obs_sonde_T.obs(param="lon")[0]
    obslat = obs_sonde_T.obs(param="lat")[0]
    rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

    mean = paiut.read_grib(ana_path, vars).isel(time=0) 
    buffer = 0.04
    mean2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, mean.clon, mean.clat, mean)
    mean2 = mean2.thin({'generalVerticalLayer': 2})
    if read_stored_file == False:
        ens = paiut.read_grib_mf(ana_path, ana_time, vars)
        ens2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, ens.clon, ens.clat, ens)
        ensperturb = ens2 - mean2
        ensperturb.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(rep_index))
        del ensperturb
    
    ensperturb = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(rep_index))
    ensperturb = ensperturb.thin({'generalVerticalLayer': 2})

    ## Get horizontal localisation factor
    # load the data into xarray.Dataset
    if read_diag_file == True:
        ctl = open_CtlDataset(hloc_path).isel(time=0)
        LAT, LON = np.meshgrid(ctl.lat.values, ctl.lon.values)
        CLON, CLAT = oi.rotated_pole_to_location(LON.flatten(), LAT.flatten())
        CLON = CLON.reshape(LON.shape)
        CLAT = CLAT.reshape(LAT.shape)
        ctl = ctl.assign_coords(clat=(("lon", "lat"), CLAT))
        ctl = ctl.assign_coords(clon=(("lon", "lat"), CLON))
    else:
        ctl = diag_set
        CLAT = ctl.clat.values
        CLON = ctl.clon.values
    h_loc = 15
    dist = get_dist_from_obs(
        ctl.clat.values, ctl.clon.values, obslat, obslon, h_loc #np.array(ana_mean.clat), np.array(ana_mean.clon), float(obs.data.lat), float(obs.data.lon), h_loc
    )
    closest = np.argmin(dist)
    print(closest)
    h_loc = ctl.where((ctl.clat==CLAT.flatten()[closest])*(ctl.clon==CLON.flatten()[closest]), drop = True).isel(lat =0, lon=0).rlh_1

    mean_pres = mean2.pres.mean(dim='cell').values/100
    closest_coarse_level = [np.argmin(np.abs(mean_pres[i] - h_loc.lev.values)) for i in range(len(mean_pres))]
    mean2 = mean2.assign(hloc=h_loc.values[closest_coarse_level])
    
    infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=25.0)

    infl_reg_latlon_RAD = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]

    print("get obs batch")

    obs_batch_RAD = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon_RAD, obsvar=sat_var, new_loc = False, RAWBT_choice = BT_choice_sat)#, RTPP_correction=RTPP)
    ekf_RAD = paiut.get_ekf(obs_path_RAD, sat_var, *infl_reg_latlon_RAD, whole_domain = False, active = True)
    Y_a = ekf_RAD.anaens()
    Y_a = Y_a - Y_a.mean(axis = 0, keepdims = True)
    rho_h = PAI.compute_hloc(obslat, obslon, 25.0, ekf_RAD.obs(param='lat'), ekf_RAD.obs(param='lon'))
    rinv = 1/ekf_RAD.obs(param='e_o')**2
    E = -(1/39)*np.matmul(Y_a, np.matmul(np.diag(rinv*rho_h), Y_a.T))
    print(E.shape)
    E_norm = estimate_spectral_norm(E, its = 40)
    print('E norm ', E_norm)
    inv_norm = estimate_spectral_norm(np.linalg.inv(np.eye(40) + E), its = 40)
    print('inv norm ' , inv_norm)
    if len(obs_batch_RAD) != 0:
        print('SAT mult: ', np.max(np.array([obs['err'] for obs in obs_batch_RAD])))
    
    # If looking at obs with adaptive horizontal loc, need potentially more observations
    if influencing_var == 'RADVEL':
        infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=16)
        infl_reg_latlon = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
        print(influencing_obs)
    elif influencing_var == 'RREFL':
        infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=16)
        infl_reg_latlon = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
        print(influencing_obs)
    elif influencing_obs != 'RAD':
        infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=np.max(h_loc.values))
        infl_reg_latlon = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
        print('Treated as conventional')
    else:
        infl_reg_latlon = infl_reg_latlon_RAD
    obs_batch_INFLUENCER = paiut.get_obs_batch(obs_path_INFLUENCER, *infl_reg_latlon, obsvar=influencing_var, new_loc = False, RAWBT_choice = BT_choice_influencer)#, RTPP_correction=RTPP)
    if len(obs_batch_INFLUENCER) != 0:
        print('inf mult ', np.max(np.array([1/obs['err'] for obs in obs_batch_INFLUENCER])))
    #print(obs_batch_INFLUENCER)
    if influencing_obs != 'RAD':
        if influencing_obs != 'RADAR':
            for obs in obs_batch_INFLUENCER:
                obs['vloc'] = 0.3

    print(len(obs_batch_INFLUENCER), ' obs that could influence')
    ekf_RAD = paiut.get_ekf(obs_path_RAD, sat_var, *infl_reg_latlon_RAD, whole_domain = False, active = True)
    Y_a_X_form = [np.moveaxis(np.array([ekf_RAD.anaens()]), [1,2], [0,1])]
    Y_a_X_form = [x - x.mean(axis = 0, keepdims = True) for x in Y_a_X_form]

    vars = ['t']
    
    lats = ensperturb.clat.values
    lons = ensperturb.clon.values
    error_list = []
    vertical_list = []
    obs_batch_RAD_change = copy.deepcopy(obs_batch_RAD)
    print(Y_a_X_form[0].shape)
    pressure_levels = mean2.pres.values
    perturbvars = [ensperturb[var].values for var in vars]
    incr = paiogc.pai_loop_over_obs_slow(
            obs_batch_RAD,
            ['t'],
            lons,
            lats,
            perturbvars,
            pressure_levels,
            40
        )
    for vert_index in range(len(mean_pres)):
        print(vert_index)
        for obs in obs_batch_INFLUENCER:
            if influencing_obs == 'RAD':
                obs['hloc'] = 25
            elif influencing_obs == 'RADVEL':
                obs['hloc'] = 16
            elif influencing_obs == 'RREFL':
                obs['hloc'] = 16
            else:
                obs['hloc'] = mean2.hloc.values[vert_index]
        #print(mean_pres[vert_index]*100)
        #obs = obs_batch_INFLUENCER[0]
        #print(obs["obspres"])
        #print(obs["vloc"])
        #print(PAI.compute_vloc(obs["obspres"] / 100.0, obs["vloc"], np.array([mean_pres[vert_index]*100]*np.ones((1, 1))) / 100.0))
        Y_a_X_form_with_E = [np.moveaxis(np.array([np.matmul(np.linalg.inv(np.eye(40) + E), ekf_RAD.anaens() - ekf_RAD.anaens().mean(axis = 0, keepdims = True))]), [1,2], [0,1])] #np.matmul(Y_a_X_form[0], np.linalg.inv(np.eye(40) + E))
        #print('with E: ', Y_a_X_form_with_E[0])
        #print('without E: ', Y_a_X_form[0])
        E_times_departure = paiogc.pai_loop_over_obs_slow(
            obs_batch_INFLUENCER,
            ['t'],
            np.array([obs_sonde_T.obs(param='lon')[0]]),
            np.array([obs_sonde_T.obs(param='lat')[0]]),
            Y_a_X_form_with_E,
            np.array([mean_pres[vert_index]*100]*np.ones((Y_a_X_form[0].shape[1], 1))),
            40
        )
        #print(mean_pres[vert_index]*100)
        #print(E_times_departure)
        print(len(E_times_departure[0][1]))
        print(len(obs_batch_RAD_change))
        perturbvars = [ensperturb[var].values[:, [vert_index], :] for var in vars]
        pressure_levels = np.array(mean2.pres.values)[[vert_index], :]
        for obs_index, obs in enumerate(obs_batch_RAD_change):
            obs['fgmeandep'] = E_times_departure[0][1][obs_index]
        error_term = paiogc.pai_loop_over_obs_slow(
            obs_batch_RAD_change,
            ['t'],
            lons,
            lats,
            perturbvars,
            pressure_levels,
            40,
        )
        print(error_term)
        error_list.append(error_term[0][1])
        vertical_list.append(mean_pres[vert_index])
    return error_list, vertical_list, incr


if __name__ == '__main__':
    exp_path = '/jetfs/shared-data/ICON-LAM-DWD/exp_2'
    dir_time_ana = '20230608120000'
    ana_time = '20230608120000'
    hloc_path = '{}/feedback/{}/monDIAG.nc.{}.ctl'.format(exp_path, dir_time_ana, ana_time)
    ctl = open_CtlDataset(hloc_path).isel(time=0)
    LAT, LON = np.meshgrid(ctl.lat.values, ctl.lon.values)
    CLON, CLAT = oi.rotated_pole_to_location(LON.flatten(), LAT.flatten())
    CLON = CLON.reshape(LON.shape)
    CLAT = CLAT.reshape(LAT.shape)
    ctl = ctl.assign_coords(clat=(("lon", "lat"), CLAT))
    ctl = ctl.assign_coords(clon=(("lon", "lat"), CLON))

    exp_path ='/jetfs/shared-data/ICON-LAM-DWD/exp_2'
    ana_time = '20230608120000'  # analysis time of first guess forecasts
    dir_time_sonde = '20230608090000'
    sonde_time = '20230608110000'
    dir_time_ana = '20230608120000'
    rep_index = 8
    RTPP = False
    error_list_RAWBT0, vert_list_RAWBT0, incr_list_RAWBT0 = resilience(exp_path, dir_time_sonde, sonde_time,
                    dir_time_ana, ana_time,
                    rep_index,
                    '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                    influencing_obs = 'RAD', influencing_var = 'RAWBT', BT_choice_influencer = 0, 
                    sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                    surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                    read_diag_file = False, diag_set = ctl
                    )
    
    error_list_RAWBT1, vert_list_RAWBT1, incr_list_RAWBT1 = resilience(exp_path, dir_time_sonde, sonde_time,
                 dir_time_ana, ana_time,
                 rep_index,
                 '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                 influencing_obs = 'RAD', influencing_var = 'RAWBT', BT_choice_influencer = 1, 
                 sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                 surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                 read_diag_file = False, diag_set = ctl
                 )
    
    error_list_RREFL, vert_list_RREFL, incr_RREFL = resilience(exp_path, dir_time_sonde, sonde_time,
                 dir_time_ana, ana_time,
                 rep_index,
                 '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                 influencing_obs = 'RADAR', influencing_var = 'RREFL', BT_choice_influencer = 0, 
                 sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                 surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                 read_diag_file = False, diag_set = ctl
                 )
    
    error_list_RADVEL, vert_list_RADVEL, incr_list_RADVEL = resilience(exp_path, dir_time_sonde, sonde_time,
                 dir_time_ana, ana_time,
                 rep_index,
                 '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                 influencing_obs = 'RADAR', influencing_var = 'RADVEL', BT_choice_influencer = 0, 
                 sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                 surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                 read_diag_file = False, diag_set = ctl
                 )
    
    PILOT_errors = []
    PILOT_incrs = []
    for P_obs in ['W', 'T', 'U', 'V']:
        print('PILOT')
        print(P_obs)
        error_list_PILOT, vert_list_PILOT, incr_list_PILOT = resilience(exp_path, dir_time_sonde, sonde_time,
                        dir_time_ana, ana_time,
                        rep_index,
                        '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                        influencing_obs = 'PILOT', influencing_var = P_obs, BT_choice_influencer = 0, 
                        sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                        surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                        read_diag_file = False, diag_set = ctl
                        )
        PILOT_errors.append(error_list_PILOT)
        PILOT_incrs.append(incr_list_PILOT)
    
    AIREP_errors = []
    AIREP_incrs = []
    for A_obs in ['T', 'U', 'V', 'RH']:
        print('AIREP')
        print(A_obs)
        error_list_AIREP, vert_list_AIREP, incr_list_AIREP = resilience(exp_path, dir_time_sonde, sonde_time,
                        dir_time_ana, ana_time,
                        rep_index,
                        '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                        influencing_obs = 'AIREP', influencing_var = A_obs, BT_choice_influencer = 0, 
                        sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                        surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                        read_diag_file = False, diag_set = ctl
                        )
        AIREP_errors.append(error_list_AIREP)
        AIREP_incrs.append(incr_list_AIREP)

    SYNOP_errors = []
    SYNOP_incrs = []
    for S_obs in ['PTEND', 'T2M', 'TD2M', 'U10M', 'V10M', 'RH2M']:
        print('SYNOP')
        print(S_obs)
        error_list_SYNOP, vert_list_SYNOP, incr_list_SYNOP = resilience(exp_path, dir_time_sonde, sonde_time,
                        dir_time_ana, ana_time,
                        rep_index,
                        '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/Compare_inf0.png',
                        influencing_obs = 'SYNOP', influencing_var = S_obs, BT_choice_influencer = 0, 
                        sat_var = 'REFL', RTPP = RTPP, Title = 'Compare_Inf',
                        surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T', save = True, read_stored_file=True,
                        read_diag_file = False, diag_set = ctl
                        )
        SYNOP_errors.append(error_list_SYNOP)
        SYNOP_incrs.append(incr_list_SYNOP)
    
    error_as_array = np.array([x[0] for x in error_list_RAWBT0])
    for count, error_list in enumerate([error_list_RAWBT0, error_list_RAWBT1, error_list_RREFL, error_list_RADVEL,
                                        PILOT_errors[0], PILOT_errors[1], PILOT_errors[2], PILOT_errors[3],
                                        AIREP_errors[0], AIREP_errors[1], AIREP_errors[2], AIREP_errors[3],
                                        SYNOP_errors[0], SYNOP_errors[1], SYNOP_errors[2], SYNOP_errors[3], SYNOP_errors[4], SYNOP_errors[5]]):
        error_as_array = np.array([[x[0] for x in error_list]])
        influencer = ['IR6.2', 'IR 7.3', 'Radar REFL', 'Radar RADVEL', 
                                'PILOT W', 'PILOT T', 'PILOT U', 'PILOT V',
                                'AIREP T', 'AIREP U', 'AIREP V', 'AIREP RH',
                                'SYNOP PTEND', 'SYNOP T2M', 'SYNOP TD2M', 'SYNOP U10M', 
                                'SYNOP V10M', 'SYNOP RH2M'][count]
        if count == 0:
            full_dataset = xr.Dataset(
                    data_vars=dict(PAI_error=(["Influencer", "Vert_level", "cell"], error_as_array),
                                   PAI_incr=(["Influencer", "Vert_level", "cell"], np.array([incr_list_RAWBT0[0][1]])),
                    ),
                    coords=dict(
                        influencing_obs=(["Influencer"], np.array([influencer])),
                        plevel=(["Vert_level"], np.array(vert_list_RAWBT0)),
                        clat=(["cell"], np.array([1 for count1 in range(error_list_RAWBT0[0].shape[1])])),
                        clon=(["cell"], np.array([1 for count1 in range(error_list_RAWBT0[0].shape[1])])),
                    ),
                    attrs=dict(description="Weather related data."),
                )
        else:
            errors_dataset = xr.Dataset(
                    data_vars=dict(PAI_error=(["Influencer", "Vert_level", "cell"], error_as_array),
                                   PAI_incr=(["Influencer", "Vert_level", "cell"], np.array([incr_list_RAWBT0[0][1]])),
                    ),
                    coords=dict(
                        influencing_obs=(["Influencer"], np.array([influencer])),
                        plevel=(["Vert_level"], np.array(vert_list_RAWBT0)),
                        clat=(["cell"], np.array([1 for count1 in range(error_list_RAWBT0[0].shape[1])])),
                        clon=(["cell"], np.array([1 for count1 in range(error_list_RAWBT0[0].shape[1])])),
                    ),
                    attrs=dict(description="Weather related data."),
                )
            full_dataset = xr.concat([full_dataset, errors_dataset], dim = 'Influencer')
    full_dataset.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Results_Final/New_Compare_influencers_big_' + str(rep_index) + '.nc')
        