import sys
sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/')
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')
from numba import njit
import xarray as xr
from pathlib import Path
import gc
import os

from kendapy.ekf import Ekf
from enstools.io import read, write

import pai.observation as oi
from pai.localization import get_dist_from_obs, find_analysis_in_area
import pai.pai_utils as paiut
import pai.partial_analysis_increment_np as PAI
import pai.plot_oi_utils as utils

import cartopy.crs as ccrs

import cProfile, pstats, io
from pstats import SortKey
import time

from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units



def rotated_pole_to_location(rlons, rlats):
    #
    rotated_pole = ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    rotated_coords = [
        ccrs.PlateCarree().transform_point(rlon, rlat, rotated_pole)
        for (rlon, rlat) in zip(rlons, rlats)
    ]

    lons, lats = zip(*rotated_coords)

    return np.array(lons), np.array(lats)


def pai_loop_over_obs_slow(obs_list, vars, enslon, enslat, ep_values, mean_pres, nens):
    incr = [[vars[i], np.zeros((ep_values[i].shape[1], ep_values[i].shape[2]))] for i in range(len(vars))]
    #ep_values = [ens_perturb[var].values for var in vars]
    for i, obs in enumerate(obs_list):
        hgc = PAI.compute_hloc(obs["obslat"], obs["obslon"], obs["hloc"], enslat, enslon)

        if obs["vloc"] < 10:
            vgc = PAI.compute_vloc(obs["obspres"] / 100.0, obs["vloc"], mean_pres / 100.0)
        else:
            vgc = 1
        kcol = [PAI.compute_kalman_gain(
            obs["anaensT"],
            obs["err"],
            ep_values[k],
            nens,
            hgc=hgc,
            vgc=vgc
        ) for k in range(len(vars))]
        for i in range(len(vars)):
            incr[i][1] += PAI.compute_increment(kcol[i], obs["fgmeandep"])
    return incr


def pai_loop_over_obs_slow_surface_peak(obs_list, vars, enslon, enslat, ep_values, mean_pres, nens, surface_vloc=1):
    incr = [[vars[i], np.zeros((ep_values[i].shape[1], ep_values[i].shape[2]))] for i in range(len(vars))]
    #ep_values = [ens_perturb[var].values for var in vars]
    surface_vgc = PAI.compute_vloc(1000, surface_vloc, mean_pres / 100.0)
    for i, obs in enumerate(obs_list):
        hgc = PAI.compute_hloc(obs["obslat"], obs["obslon"], obs["hloc"], enslat, enslon)

        if obs["vloc"] < 10:
            surface_influence = PAI.compute_vloc(1000, surface_vloc, np.array([obs["obspres"]]) / 100.0)
            vgc = (PAI.compute_vloc(obs["obspres"] / 100.0, obs["vloc"], mean_pres / 100.0) + surface_vgc)/(1+surface_influence)
        else:
            vgc = 1
        kcol = [PAI.compute_kalman_gain(
            obs["anaensT"],
            obs["err"],
            ep_values[k],
            nens,
            hgc=hgc,
            vgc=vgc
        ) for k in range(len(vars))]
        for i in range(len(vars)):
            incr[i][1] += PAI.compute_increment(kcol[i], obs["fgmeandep"])
    return incr



VIS_coeffs_04VIScloud_raw_full = [-2.68301229, 1.59745498, 0.83275683, 1.45290819, 0.7796353, 0.68094375,  0.43331787, 1.32657363, 1.46727558, 0.04846825, 1.08187253] 
VIS_coeffs_04VIScloud_bounds_full = [0,  1, 0.83275683, 1, 0.7796353, 0.68094375, 0.43331787, 1, 1, 0.04846825, 1]
VIS_coeffs_04VIScloud_raw_above = [-4.80383929, 1.26198514, 0.78204852, 1.19909524, 0.6450851, -0.54036816, -0.04373399, 1.67581813, 2.35614056, -0.1254766, 1.50428088] 
VIS_coeffs_04VIScloud_bounds_above = [0, 1, 0.78204852, 1, 0.6450851,  0, 0, 1, 1, 0, 1]
VIS_coeffs_04VIScloud_raw_below = [0.65895695, 1.90488364, 0.87536263, 1.67338999, 0.8781595, 1.473791, 0.65058917, 1.21743423, 0.3653289,  0.30794885, 0.5730768] 
VIS_coeffs_04VIScloud_bounds_below = [0.65895695, 1, 0.87536263, 1, 0.8781595, 1, 0.65058917, 1, 0.3653289, 0.30794885, 0.5730768 ]

IR0_coeffs_233cloud_raw_full = [-2.921781210465183, 1.016041911057499, 0.4885386865028979, 0.23112258377954886, -0.10633935163219867, 0.6438577601088938, 0.01663965031534579, 0.06922958558123664, 1.8232196586122518, -0.09351143517611021, 0.48505394981532035]
IR0_coeffs_233cloud_bounds_full = [0.0, 1.0, 0.4885386865028979, 0.23112258377954886, 0.0, 0.6438577601088938, 0.01663965031534579, 0.06922958558123664, 1.0, 0.0, 0.48505394981532035]
IR0_coeffs_233cloud_raw_above = [-0.7354202439488421, 0.8590873579025468, 0.3881013003077521, -0.25524954583290727, 0.0763737926860092, 0.92849341619604, 0.7197598365656169, -1.3444370578623515, 1.6791637384507498, -0.7320038723206626, 0.008589622744443257]
IR0_coeffs_233cloud_bounds_above = [0.0, 0.8590873579025468, 0.3881013003077521, 0.0, 0.0763737926860092, 0.92849341619604, 0.7197598365656169, 0.0, 1.0, 0.0, 0.008589622744443257]
IR0_coeffs_233cloud_raw_below = [-3.5030287668450746, 1.0604315028719489, 0.5306636828088477, 0.39814287684194244, -0.151323368761332, 0.49210662882405387, -0.312211982967245, 0.6276951490768374, 1.8779285419590628, 0.191931378356844, 0.652267290531055]
IR0_coeffs_233cloud_bounds_below = [0.0, 1.0, 0.5306636828088477, 0.39814287684194244, 0.0, 0.49210662882405387, 0.0, 0.6276951490768374, 1.0, 0.191931378356844, 0.652267290531055]

IR1_coeffs_252cloud_raw_full = [-2.2210304242325463, 0.6359517621168526, 0.7332342669784513, 0.10746331461598115, 0.07045728242739405, -0.01098569740186419, 0.3531118523088319, 0.4236491682812797, 1.2418664985749917, -0.3312188975296008, -0.6387026797345228]
IR1_coeffs_252cloud_bounds_full = [0.0, 0.6359517621168526, 0.7332342669784513, 0.10746331461598115, 0.07045728242739405, 0.0, 0.3531118523088319, 0.4236491682812797, 1.0, 0.0, 0.0]
IR1_coeffs_252cloud_raw_above = [-0.14692650900700507, -0.05258049186680647, 0.9182479139244245, -0.29700699031969513, 1.0437268835410005, -0.38382901308400685, -0.25930315335297127, 0.5206910989605144, 1.3812335886391736, -0.43186779742706216, -2.254603252344699]
IR1_coeffs_252cloud_bounds_above = [0.0, 0.0, 0.9182479139244245, 0.0, 1.0, 0.0, 0.0, 0.5206910989605144, 1.0, 0.0, 0.0]
IR1_coeffs_252cloud_raw_below = [-2.7778405676910602, 0.8791418418599205, 0.6704410550122154, 0.19502954794216865, -0.06970832829620444, 0.06108146041384197, 0.5045073192401709, 0.3869692068211075, 1.1723492592283988, -0.3005553608083622, -0.18124885709985447]
IR1_coeffs_252cloud_bounds_below = [0.0, 0.8791418418599205, 0.6704410550122154, 0.19502954794216865, 0.0, 0.06108146041384197, 0.5045073192401709, 0.3869692068211075, 1.0, 0.0, 0.0]

if __name__ == '__main__':
    Results = xr.Dataset(data_vars=dict(incr=(["Observation"], np.array([])),
                            benefit=(["Observation"], np.array([])),
                            err_new_loc=(["Observation", "loc_function"], np.array([[] for count in ['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                            'ELF_full', 'ELF_above', 'ELF_below']]).T),
                            loc_type_test=(["Observation", "loc_function"], np.array([[] for count in ['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                            'ELF_full', 'ELF_above', 'ELF_below']]).T),
                            VIS_value=(["Observation"], np.array([])),
                            VIS_ana=(["Observation"], np.array([])),
                            VIS_fg=(["Observation"], np.array([])),
                            IR0_value=(["Observation"], np.array([])),
                            IR0_ana=(["Observation"], np.array([])),
                            IR0_fg=(["Observation"], np.array([])),
                            IR1_value=(["Observation"], np.array([])),
                            IR1_ana=(["Observation"], np.array([])),
                            IR1_fg=(["Observation"], np.array([])),
                            ), 
                        coords=dict(
                            veri_obs_type=(["Observation"], np.array([])),
                            plevel=(["Observation"], np.array([])),
                            rep_num=(["Observation"], np.array([])),
                            rep_name=(["Observation"], np.array([])),
                            day=(["Observation"], np.array([])),
                            Time = (["Observation"], np.array([])),
                            localisation=(["loc_function"], np.array(['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                            'ELF_full', 'ELF_above', 'ELF_below'])),
                            ),
        attrs=dict(description="Weather related data."),
    )
    parser = utils.define_parser()
    args = parser.parse_args()
    exp_path = args.basedir
    
    ## VIS Params for TABLE ##
    #single_peak_parameters = [[80000, 0.3], [90000, 0.1], [40000, 0.3], [35000, 1.0]]
    #Global GC, cloud GC, cloud GC, Clear ELF fit
    #multi_peak_parameters = [[87000, 0.1, 35000, 0.5], [90000, 0.1, 33000, 0.4], [90000, 0.1, 40000, 0.3], [33000, 0.3, 90000, 0.07]]
    #Global ELF fit, Cloudy ELF fit, cloud GC fit, wider cloudy ELF fit
    
    ## 1st IR channel params for Table ##
    #single_peak_parameters = [[20000, 0.3], [50000, 0.1], [25000, 0.1], [90000, 0.1]]
    #Global GC, cloud GC, cloud GC, Clear GC
    #multi_peak_parameters = [[90000, 0.05, 20000, 0.45], [90000, 0.1, 20000, 0.5], [67000, 0.1, 20000, 0.3], [33000, 0.3, 90000, 0.07]]
    #Global ELF fit, Clear ELF fit, cloud GC fit, Ignore

    ## 2nd IR channel params for Table ##
    single_peak_parameters = [[25000, 0.2], [90000, 0.1], [55000, 0.1], [25000, 0.1]]
    #Global/clear GC, cloud GC, cloud GC, Cloud GC
    multi_peak_parameters = [[87000, 0.05, 25000, 0.3], [87000, 0.05, 32000, 0.15], [67000, 0.1, 20000, 0.3], [33000, 0.3, 90000, 0.07]]
    #Global/cloudy ELF fit, Cloudy ELF fit, Ignore, Ignore
    
    read_stored_file = True
    Surface_Peak = False
    averaging_n = 6
    ELF_input_form = np.array(range(4, 65, 6))
    ELF_output_form = np.array(range(0, 65))
    start_date = int(args.ELFstartdate)
    end_date = int(args.ELFenddate)
    ana_days = [f'202306{str(i).zfill(2)}' for i in range(start_date, end_date + 1)]
    prev_days = [f'202306{str(i).zfill(2)}' for i in range(start_date - 1, end_date)]
    Real_Incr = True 
    vloc = 10
    obspres = 600
    vert_levels = np.array(range(10000, 101000, 5000))
    error_list = [[] for count in range(7)] #For reference: first/second list is to store average absolute influence for T then Q, third/fourth average abs error no obs for T then RH, fifth/sixth average error with obs for T then RH.
    RTPP = False
    RTPP_factor = 0.75
    if args.midnight == 'Midnight':
        midnight = True
    else:
        midnight = False
    #Profile to see what takes longest:
    pr = cProfile.Profile()
    pr.enable()
    inf_exp = False
    exp_path2 = '/jetfs/shared-data/ICON-LAM-DWD/exp_inf'
    ana_time2 = '20230601130000'
    for ind, ana_day in enumerate(ana_days): # 
        if ana_day == '20230601' and midnight:
            continue
        print(ana_day)
        dir_time = ana_day + args.time
        ana_time = ana_day + args.time
        if midnight:
            dir_time2 = prev_days[ind] + args.prevtime
            sonde_time = prev_days[ind] + args.sondetime
        else:
            dir_time2 = ana_day + args.prevtime
            sonde_time = ana_day + args.sondetime
        obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, ana_time, ana_time)
        obs_path_TEMP1 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time2, sonde_time)
        obs_path_TEMP2 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, ana_time, ana_time)
        if inf_exp:
            ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path2, ana_time, ana_time2)
            ana_path_inc = '{}/{}/an_R19B07.{}_inc'.format(exp_path2, ana_time, ana_time2)
            fc_path = '{}/{}/fc_R19B07.{}.mean'.format(exp_path2, ana_time, ana_time2)
        else:
            ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)
            ana_path_inc = '{}/{}/an_R19B07.{}_inc'.format(exp_path, ana_time, ana_time)
            fc_path = '{}/{}/fc_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)
        #Get list of radiosondes to test against.
        sondes_RH1 = paiut.get_ekf(obs_path_TEMP1, "RH", active = False)
        sondes_T1 = paiut.get_ekf(obs_path_TEMP1, "T", active = False)
        if os.path.isfile(obs_path_TEMP2) == True:
            sondes_RH2 = paiut.get_ekf(obs_path_TEMP2, "RH", active = False)
            sondes_T2 = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
            report_number = len(sondes_RH1.reports()) + len(sondes_RH2.reports())
        else:
            report_number = len(sondes_RH1.reports())

        vars = ['t', 'q']
        print('read analysis')
        mean = paiut.read_grib(ana_path, vars) 
        mean = mean.isel(time=0)
        if RTPP:
            if inf_exp:
                meaninc = paiut.read_grib(ana_path_inc, vars)
                meaninc = meaninc.isel(time=0)
                ensinc = paiut.read_grib_mf(ana_path_inc, ana_time2, vars, inc = True)
                ensinc = ensinc.isel(time=0)
            else:
                meaninc = paiut.read_grib(ana_path_inc, vars)
                meaninc = meaninc.isel(time=0)
                ensinc = paiut.read_grib_mf(ana_path_inc, ana_time, vars, inc = True)
                ensinc = ensinc.isel(time=0)
        else:
            if inf_exp:
                ens = paiut.read_grib_mf(ana_path, ana_time2, vars)
                ens = ens.isel(time=0)
            else:
                if read_stored_file == False:
                    ens = paiut.read_grib_mf(ana_path, ana_time, vars)
                    ens = ens.isel(time=0)
        print('analysis found')
        #if RTPP:
            #ensfc = paiut.read_grib_mf(fc_path, ana_time, vars, fc = True)
            #ensfc = ensfc.isel(time=0)
        #print('fg found')
        for i in range(report_number): #len(sondes_T1.reports()) + len(sondes_T2.reports())):
            #Retrieve information for individual report.
            if i < len(sondes_T1.reports()):
                rep = sondes_T1.reports()[i]
                obs_sonde_RH = paiut.get_ekf(obs_path_TEMP1, "RH", active = False)
                obs_sonde_T = paiut.get_ekf(obs_path_TEMP1, "T", active = False)
                
            else:
                rep = sondes_T2.reports()[i - len(sondes_T1.reports())]
                obs_sonde_RH = paiut.get_ekf(obs_path_TEMP2, "RH", active = False)
                obs_sonde_T = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
            
            obs_sonde_RH.add_filter(filter=f"report={rep}")
            obs_sonde_T.add_filter(filter=f"report={rep}")
            obslon = obs_sonde_RH.obs(param="lon")[0]
            obslat = obs_sonde_RH.obs(param="lat")[0]
            print(f"Sonde {i}, report {rep}, lat {obslat}, lon {obslon}")
            
            #Get region close to the radiosonde and restrict analysis and radiances to this region.
            rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

            infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=25.0)

            infl_reg_latlon = np.array(rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
            
            print("get obs batch")
            obs_batch = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon, obsvar="RAWBT", RAWBT_choice = 1, new_loc=False, RTPP_correction = False)
            print(len(obs_batch))
            if len(obs_batch) == 0:
                continue

            print("read data")
            var = 't'
            
            buffer = 0.04
            mean2 = find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, mean.clon, mean.clat, mean)
            if RTPP == False:
                if read_stored_file == False:
                    ens2 = find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, ens.clon, ens.clat, ens)
                    ensperturb_ana = ens2 - mean2
            if RTPP == True:
                ensperturb_ana = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(i))
            
            if RTPP:
                ensinc2 = find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, ensinc.clon, ensinc.clat, ensinc)
                meaninc2 = find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, meaninc.clon, meaninc.clat, meaninc)
                ensperturbfc = ensperturb_ana - ensinc2
                ensperturbfc = ensperturbfc - ensperturbfc.mean(dim = 'ens')
                ensperturb = (1/RTPP_factor)*(ensperturb_ana - (1 - RTPP_factor)*ensperturbfc)
                #del ensperturbfc
                del ensperturb_ana
            elif read_stored_file == False:
                ensperturb = ensperturb_ana
            print(mean2.t.shape)
            if mean2.t.shape[1] == 0:
                continue

            #Get the closest lat lon for calculations later (replace by proper interpolation)
            h_loc = 15
            print('get_distances')
            dist = get_dist_from_obs(
                np.array(mean2.clat), np.array(mean2.clon), obslat, obslon, h_loc #np.array(ana_mean.clat), np.array(ana_mean.clon), float(obs.data.lat), float(obs.data.lon), h_loc
            )

            closest = np.argmin(dist)
            print('distances found')
            if RTPP == True:
                if inf_exp:
                    ensperturbfc.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/no_inf_X_b_' + ana_time + '_' + str(i))
                    ensperturb.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/no_inf_tilde_X_a_' + ana_time + '_' + str(i))
                    del ensperturb
                    ensperturb2 = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/no_inf_tilde_X_a_' + ana_time + '_' + str(i))
                else:
                    ensperturb.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/tilde_X_a_' + ana_time + '_' + str(i))
                    del ensperturb
                    ensperturb2 = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/tilde_X_a_' + ana_time + '_' + str(i))
            else:
                if inf_exp:
                    ensperturb.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/no_inf_X_a_' + ana_time + '_' + str(i))
                    del ensperturb
                    ensperturb2 = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/no_inf_X_a_' + ana_time + '_' + str(i))
                else:
                    if read_stored_file == False:
                        ensperturb.to_netcdf('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(i))
                        del ensperturb
                    print(ana_time, i)
                    ensperturb2 = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(i))
                    print(ensperturb2.clat.values[0], ensperturb2.clon.values[0], obslat, obslon)
            #Extract values now so that we don't need to every time we call PAI.
            pressure_levels = mean2.pres.values
            lats = ensperturb2.clat.values
            lons = ensperturb2.clon.values
            perturbvars = [ensperturb2[var].values for var in vars]
            perturbvars.append(ensperturb2.pres.values)
            ensperturb2.close()
            #Get actual increment (approximated by PAI) attributed to radiances in this area (for error calculations later).
            print('get PAI increments')
            start_time = time.time()
            incr_REAL = pai_loop_over_obs_slow(
                    obs_batch,
                    ['t', 'q', 'pres'],
                    lons,
                    lats,
                    perturbvars,
                    pressure_levels,
                    40,
                )
            incr_REAL_T = incr_REAL[0][1]
            incr_REAL_Q = incr_REAL[1][1]
            incr_REAL_P = incr_REAL[2][1]
            #Change vloc and obspres settings then do it again to get the 'unlocalised' increment.
            for obs in obs_batch:
                obs['vloc'] = 20
                obs['obspres'] = 55000
            incr_noloc = pai_loop_over_obs_slow(
                    obs_batch,
                    ['t', 'q', 'pres'],
                    lons,
                    lats,
                    perturbvars,
                    pressure_levels,
                    40,
                )
            incr_noloc_T = incr_noloc[0][1]
            incr_noloc_Q = incr_noloc[1][1]
            incr_noloc_P = incr_noloc[2][1]
            print(f"{(time.time() - start_time)} seconds to run PAI")
            print('calculate errors')
            GC_no_loc = np.array([1 for count in range(65)])
            GC_no_obs = np.array([0 for count in range(65)])

            GCs = []
            for parameters in single_peak_parameters:#
                #parameters = [peak centre, peak width]
                vgc = PAI.compute_vloc(parameters[0] / 100.0, parameters[1], pressure_levels / 100.0)
                GCs.append(vgc)
            GC_1 = GCs[0]
            GC_2 = GCs[1]
            GC_3 = GCs[2]
            GC_4 = GCs[3]
            
            GC_two_peaks = []
            for parameters in multi_peak_parameters:
                #parameters = [surface peak centre, surface peak width, atmos peak centre, atmos peak width]
                surface_vgc = PAI.compute_vloc(parameters[0] / 100, parameters[1], pressure_levels / 100.0)
                atmos_vgc = PAI.compute_vloc(parameters[2] / 100.0, parameters[3], pressure_levels / 100.0)
                surface_influence_atmos = PAI.compute_vloc(parameters[0] / 100, parameters[1], np.array([parameters[2]]) / 100.0)
                atmos_influence_surface = PAI.compute_vloc(parameters[2] / 100.0, parameters[3], np.array([parameters[0]]))
                c_surface = (1 - atmos_influence_surface)/(1 - (atmos_influence_surface * surface_influence_atmos))
                c_atmos = (1 - surface_influence_atmos)/(1 - (atmos_influence_surface * surface_influence_atmos))
                GC_two_peaks.append((c_atmos * atmos_vgc) + (c_surface * surface_vgc))
            GC_two_peak_1 = GC_two_peaks[0]
            GC_two_peak_2 = GC_two_peaks[1]
            GC_two_peak_3 = GC_two_peaks[2]
            GC_two_peak_4 = GC_two_peaks[3]

            ELF_full = np.interp(ELF_output_form, ELF_input_form, IR1_coeffs_252cloud_bounds_full)
            ELF_above = np.interp(ELF_output_form, ELF_input_form, IR1_coeffs_252cloud_bounds_above)
            ELF_below = np.interp(ELF_output_form, ELF_input_form, IR1_coeffs_252cloud_bounds_below)

            vertical_loc_functions = [GC_no_loc, GC_no_obs,
                                      GC_1, GC_2, GC_3, GC_4, 
                                      GC_two_peak_1, GC_two_peak_2, GC_two_peak_3, GC_two_peak_4, 
                                      ELF_full, ELF_above, ELF_below]
            loc_errs_T = []
            loc_errs_RH = []
            # First analysis error before PAI analysis:
            ana_obsloc_t = LinearNDInterpolator(list(zip(lats, lons)), mean2.t.values.T)(obslat, obslon)
            ana_obsloc_q = LinearNDInterpolator(list(zip(lats, lons)), mean2.q.values.T)(obslat, obslon)
            interp_plevs = LinearNDInterpolator(list(zip(lats, lons)), pressure_levels.T)(obslat, obslon)
            
            ana_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), interp_plevs, ana_obsloc_t)
            ana_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), interp_plevs, ana_obsloc_q)
            ana_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), interp_plevs, ana_obsloc_t)

            ana_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i]/100 * units.hPa,
                                                                        (ana_t_obs_plevs[i] - 273.15) * units.degC,
                                                                        ana_q_obs_plevs[i] * units('kg/kg')).to('percent').magnitude / 100 for i in range(len(ana_t_obs_plevs))])

            devs_T = (ana_T_obs_plevs - np.flip(obs_sonde_T.obs()))
            errs_T = devs_T / np.flip(obs_sonde_T.obs(param = 'e_o'))
            loc_errs_T.append(np.linalg.norm(errs_T)**2)
            devs_RH = (ana_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))
            errs_RH = devs_RH / np.flip(obs_sonde_RH.obs(param = 'e_o'))
            loc_errs_RH.append(np.linalg.norm(errs_RH)**2)
            
            t_no_obs = mean2.t.values - incr_REAL_T
            q_no_obs = mean2.q.values -  incr_REAL_Q
            p_no_obs = mean2.pres.values - incr_REAL_P
            #Interpolate
            start_time = time.time()
            print('start interpolating')
            mean_t_no_obs = LinearNDInterpolator(list(zip(lats, lons)), t_no_obs.T)(obslat, obslon)
            mean_q_no_obs = LinearNDInterpolator(list(zip(lats, lons)), q_no_obs.T)(obslat, obslon)
            mean_p_no_obs = LinearNDInterpolator(list(zip(lats, lons)), p_no_obs.T)(obslat, obslon)
            print(mean_t_no_obs.shape)
            print('Finished first interpolation')
            #Now with the different localisation functions:
            for l, vert_loc in enumerate(vertical_loc_functions):
                print(['Ana_error', 'GC_no_loc', 'GC_no_obs',
                'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                'ELF_full', 'ELF_above', 'ELF_below'][l + 1])
                if l in [0, 1, 10, 11, 12]:
                    incr_T = incr_noloc_T*vert_loc[:, None]
                    incr_Q = incr_noloc_Q*vert_loc[:, None]
                    incr_P = incr_noloc_P*vert_loc[:, None]
                else:
                    incr_T = incr_REAL_T*vert_loc
                    incr_Q = incr_REAL_Q*vert_loc
                    incr_P = incr_REAL_P*vert_loc

                loc_t = mean2.t.values + incr_T - incr_REAL_T
                loc_q = mean2.q.values + incr_Q - incr_REAL_Q
                loc_p = mean2.pres.values + incr_P - incr_REAL_P
                print('start interpolating')
                with_loc_t = LinearNDInterpolator(list(zip(lats, lons)), loc_t.T)(obslat, obslon)
                with_loc_q = LinearNDInterpolator(list(zip(lats, lons)), loc_q.T)(obslat, obslon)
                with_loc_p = LinearNDInterpolator(list(zip(lats, lons)), loc_p.T)(obslat, obslon)
                print(with_loc_t.shape)
                print('Finished second interpolation')
                print(f"{(time.time() - start_time)} seconds to run interpolate")


                #Interpolate to radiosonde pressure levels. Note that we only use the top 10 pressure levels as the radiosonde doesn't need much more.
                no_obs_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs, mean_t_no_obs)
                no_obs_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs, mean_q_no_obs)

                no_obs_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[k]/100 * units.hPa, 
                                                                                        (no_obs_t_obs_plevs[k] - 273.15) * units.degC, 
                                                                                        no_obs_q_obs_plevs[k] * units('kg/kg')).to('percent').magnitude / 100 for k in range(len(no_obs_t_obs_plevs))])
                        
                no_obs_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), mean_p_no_obs, mean_t_no_obs)

                with_loc_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), with_loc_p, with_loc_t)
                with_loc_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), with_loc_p, with_loc_q)

                with_loc_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[k]/100 * units.hPa,
                                                                                        (with_loc_t_obs_plevs[k] - 273.15) * units.degC, 
                                                                                        with_loc_q_obs_plevs[k] * units('kg/kg')).to('percent').magnitude / 100 for k in range(len(with_loc_t_obs_plevs))])
                
                with_loc_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), with_loc_p, with_loc_t)
                
                
                devs_T = (with_loc_T_obs_plevs - np.flip(obs_sonde_T.obs()))
                errs_T = devs_T / np.flip(obs_sonde_T.obs(param = 'e_o'))
                loc_errs_T.append(np.linalg.norm(errs_T)**2)
                devs_RH = (with_loc_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))
                errs_RH = devs_RH / np.flip(obs_sonde_RH.obs(param = 'e_o'))
                loc_errs_RH.append(np.linalg.norm(errs_RH)**2)
                if l == 0: #If we are looking at the localisation function being 'include all observations' we can look at the full potential increment. So:
                    incr_T_to_save = with_loc_T_obs_plevs - no_obs_T_obs_plevs
                    incr_RH_to_save = with_loc_RH_obs_plevs - no_obs_RH_obs_plevs
                    devs_no_obs_T = (no_obs_T_obs_plevs - np.flip(obs_sonde_T.obs()))
                    devs_no_obs_RH = (no_obs_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))
                    benefit_T_to_save = np.absolute(devs_no_obs_T) - np.absolute(devs_T)
                    benefit_RH_to_save = np.absolute(devs_no_obs_RH) - np.absolute(devs_RH)
            #Calculate error then interpolate back to plotting pressure levels:
            
            print('get satellite values')
            ekf_RAD = paiut.get_ekf(obs_path_RAD, "REFL")
            ekf_RAWBT = paiut.get_ekf(obs_path_RAD, "RAWBT")
            dist_obs_IR = get_dist_from_obs(
                np.array(ekf_RAWBT.obs(param = 'lat')), 
                np.array(ekf_RAWBT.obs(param = 'lon')), 
                obslat, 
                obslon, 
                h_loc
            )
            closest_obs_IR = np.argmin(dist_obs_IR)
            if len(ekf_RAD.obs()) != 0:
                dist_obs_VIS = get_dist_from_obs(
                    np.array(ekf_RAD.obs(param = 'lat')), 
                    np.array(ekf_RAD.obs(param = 'lon')), 
                    obslat, 
                    obslon, 
                    h_loc
                )
                closest_obs_VIS = np.argmin(dist_obs_VIS)
                VIS_obs = ekf_RAD.obs()[closest_obs_VIS]
                VIS_fg = ekf_RAD.fg()[closest_obs_VIS]
                VIS_ana = ekf_RAD.anamean()[closest_obs_VIS]

            else:
                VIS_obs = np.nan
                VIS_fg = np.nan
                VIS_ana = np.nan

            
            
            
            gc.collect()
            loc_errs_T = np.array(loc_errs_T)
            loc_errs_RH = np.array(loc_errs_RH)
            num_T_obs = len(obs_sonde_T.obs())
            one_rep_results_T = xr.Dataset(
                data_vars=dict(incr=(["Observation"], np.array(incr_T_to_save)),
                    benefit=(["Observation"], np.array(benefit_T_to_save)),
                    err_new_loc=(["Observation", "loc_function"], np.array([loc_errs_T for count in range(num_T_obs)])),
                    loc_type_test=(["Observation", "loc_function"], np.array([['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                                            'ELF_full', 'ELF_above', 'ELF_below'] for count in range(num_T_obs)])),
                    VIS_value=(["Observation"], np.array([VIS_obs for count in range(num_T_obs)])),
                    VIS_ana=(["Observation"], np.array([VIS_ana for count in range(num_T_obs)])),
                    VIS_fg=(["Observation"], np.array([VIS_fg for count in range(num_T_obs)])),
                    IR0_value=(["Observation"], np.array([ekf_RAWBT.obs()[closest_obs_IR] for count in range(num_T_obs)])),
                    IR0_ana=(["Observation"], np.array([ekf_RAWBT.anamean()[closest_obs_IR] for count in range(num_T_obs)])),
                    IR0_fg=(["Observation"], np.array([ekf_RAWBT.fg()[closest_obs_IR] for count in range(num_T_obs)])),
                    IR1_value=(["Observation"], np.array([ekf_RAWBT.obs()[closest_obs_IR + 1] for count in range(num_T_obs)])),
                    IR1_ana=(["Observation"], np.array([ekf_RAWBT.anamean()[closest_obs_IR + 1] for count in range(num_T_obs)])),
                    IR1_fg=(["Observation"], np.array([ekf_RAWBT.fg()[closest_obs_IR + 1] for count in range(num_T_obs)])),
                ),
                coords=dict(
                    veri_obs_type=(["Observation"], np.array(['T' for count in range(num_T_obs)])),
                    plevel=(["Observation"], np.flip(obs_sonde_T.obs(param='plevel'))),
                    rep_num=(["Observation"], np.array([i for count in range(num_T_obs)])),
                    rep_name=(["Observation"], np.array([rep for count in range(num_T_obs)])),
                    day=(["Observation"], np.array([ana_day for count in range(num_T_obs)])),
                    Time = (["Observation"], np.array([args.time for count in range(num_T_obs)])),
                    localisation=(["loc_function"], np.array(['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                            'ELF_full', 'ELF_above', 'ELF_below'])),
                ),
                attrs=dict(description="Weather related data."),

            )

            num_RH_obs = len(obs_sonde_RH.obs())
            one_rep_results_RH = xr.Dataset(
                data_vars=dict(incr=(["Observation"], np.array(incr_RH_to_save)),
                    benefit=(["Observation"], np.array(benefit_RH_to_save)),
                    err_new_loc=(["Observation", "loc_function"], np.array([loc_errs_RH for count in range(num_RH_obs)])),
                    loc_type_test=(["Observation", "loc_function"], np.array([['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                                            'ELF_full', 'ELF_above', 'ELF_below'] for count in range(num_RH_obs)])),
                    VIS_value=(["Observation"], np.array([VIS_obs for count in range(num_RH_obs)])),
                    VIS_ana=(["Observation"], np.array([VIS_ana for count in range(num_RH_obs)])),
                    VIS_fg=(["Observation"], np.array([VIS_fg for count in range(num_RH_obs)])),
                    IR0_value=(["Observation"], np.array([ekf_RAWBT.obs()[closest_obs_IR] for count in range(num_RH_obs)])),
                    IR0_ana=(["Observation"], np.array([ekf_RAWBT.anamean()[closest_obs_IR] for count in range(num_RH_obs)])),
                    IR0_fg=(["Observation"], np.array([ekf_RAWBT.fg()[closest_obs_IR] for count in range(num_RH_obs)])),
                    IR1_value=(["Observation"], np.array([ekf_RAWBT.obs()[closest_obs_IR + 1] for count in range(num_RH_obs)])),
                    IR1_ana=(["Observation"], np.array([ekf_RAWBT.anamean()[closest_obs_IR + 1] for count in range(num_RH_obs)])),
                    IR1_fg=(["Observation"], np.array([ekf_RAWBT.fg()[closest_obs_IR + 1] for count in range(num_RH_obs)])),
                ),
                coords=dict(
                    veri_obs_type=(["Observation"], np.array(['RH' for count in range(num_RH_obs)])),
                    plevel=(["Observation"], np.flip(obs_sonde_RH.obs(param='plevel'))),
                    rep_num=(["Observation"], np.array([i for count in range(num_RH_obs)])),
                    rep_name=(["Observation"], np.array([rep for count in range(num_RH_obs)])),
                    day=(["Observation"], np.array([ana_day for count in range(num_RH_obs)])),
                    Time = (["Observation"], np.array([args.time for count in range(num_RH_obs)])),
                    localisation=(["loc_function"], np.array(['Ana_error', 'GC_no_loc', 'GC_no_obs',
                                                            'GC_1', 'GC_2', 'GC_3', 'GC_4', 
                                                            'GC_two_peak_1', 'GC_two_peak_2', 'GC_two_peak_3', 'GC_two_peak_4', 
                                                            'ELF_full', 'ELF_above', 'ELF_below'])),
                ),
                attrs=dict(description="Weather related data."),

            )
            Results = xr.concat([Results, one_rep_results_T, one_rep_results_RH], dim = 'Observation')
    namecode = args.ELFnamecode
    Results.to_netcdf(f'/jetfs/shared-data/ICON-LAM-DWD/exp_2/Results/Table_Computations_{namecode}.nc')

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print(s.getvalue())