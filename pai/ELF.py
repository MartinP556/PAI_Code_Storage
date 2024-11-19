import sys
sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/')
import os
import matplotlib.pyplot as plt
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
import quadprog

from kendapy.ekf import Ekf
from enstools.io import read, write

import pai.observation as oi
from pai.localization import get_dist_from_obs, find_analysis_in_area
import pai.localization as loc
import pai.pai_utils as paiut
import pai.partial_analysis_increment_np as PAI
import pai.plot_oi_utils as utils

import cartopy.crs as ccrs

import cProfile, pstats, io
from pstats import SortKey
import time

from metpy.calc import relative_humidity_from_specific_humidity, mixing_ratio_from_relative_humidity, specific_humidity_from_mixing_ratio
from metpy.units import units

#%pip install statsmodels
import statsmodels.api as sm
from scipy import stats


def get_local_regions(obslon, obslat, h_loc=25.0):
    #Get region close to the radiosonde and restrict analysis and radiances to this region.
    rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

    infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=h_loc)

    infl_reg_latlon = np.array(rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()

    return infl_reg_latlon


def rotated_pole_to_location(rlons, rlats):
    #
    rotated_pole = ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    rotated_coords = [
        ccrs.PlateCarree().transform_point(rlon, rlat, rotated_pole)
        for (rlon, rlat) in zip(rlons, rlats)
    ]

    lons, lats = zip(*rotated_coords)

    return np.array(lons), np.array(lats)

def get_local_regions(obslon, obslat, h_loc=25.0):
    #Get region close to the radiosonde and restrict analysis and radiances to this region.
    rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

    infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=h_loc)

    infl_reg_latlon = np.array(rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()

    return infl_reg_latlon


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


def C_Matrix(PAI_vals, PAI_lats, PAI_lons, ana_plevs,  obs_lat, obs_lon, obs_plevs, input_form = np.array(range(0, 65, 2))):
    #Returns the C matrix for a radiosonde location and specific PAI output.
    #PAI_vals should be a numpy array, the first axis should be horizontal location, equivalent to PAI_lons/lats and the other axis should be pressure levels.
    #Note that we interpolate onto analysis pressure levels - the p variable in the PAI vector is an increment. Could cause issues.
    #obs_lat, obs_lon, and obs_plevs should be the latitude, longitude, and pressure levels of the observation.
    #Example of inputs used in script below: 
    output_form = np.array(range(65))
    PAI_h_interp = LinearNDInterpolator(list(zip(PAI_lats, PAI_lons)), PAI_vals)(obs_lat, obs_lon)
    PAI_p_interp = LinearNDInterpolator(list(zip(PAI_lats, PAI_lons)), ana_plevs)(obs_lat, obs_lon)
    C_as_list = []
    for i in range(len(PAI_p_interp)):
        model_c = np.zeros(len(PAI_p_interp))
        model_c[i] = PAI_h_interp[i]
        #print(len(model_c), len(obs_plevs))
        #print(obs_plevs.shape, PAI_p_interp.shape, model_c.shape)
        #print(np.interp(obs_plevs, PAI_p_interp[:, 0], model_c))
        C_as_list.append(np.interp(obs_plevs, PAI_p_interp, model_c))
    thinning_interp = []
    for i in range(len(input_form)):
        id_at_point = np.zeros(len(input_form))
        id_at_point[i] = 1
        thinning_interp.append(np.interp(output_form, input_form, id_at_point))
    thinning_interp = np.array(thinning_interp).T
    C_full_length = np.array(C_as_list).T
    return (C_full_length @ thinning_interp).T

def y_equivalent(ana_vals_no_obs, PAI_lats, PAI_lons, ana_plevs,  obs_lat, obs_lon, obs_plevs):
    #Returns the observation equivalent vector for a radiosonde location and analysis with observation removed by PAI. Will go into the dy error term.
    #PAI_vals should be a numpy array, the first axis should be horizontal location, equivalent to PAI_lons/lats and the other axis should be pressure levels.
    #Note that we interpolate onto analysis pressure levels - the p variable in the PAI vector is an increment. Could cause issues.
    #obs_lat, obs_lon, and obs_plevs should be the latitude, longitude, and pressure levels of the observation.
    #Example of inputs used in script below:
    interp_h = LinearNDInterpolator(list(zip(PAI_lats, PAI_lons)), ana_vals_no_obs)
    interp_p = LinearNDInterpolator(list(zip(PAI_lats, PAI_lons)), ana_plevs)
    PAI_h_interp = interp_h(obs_lat, obs_lon)
    PAI_p_interp = interp_p(obs_lat, obs_lon)
    y = np.interp(obs_plevs, PAI_p_interp, PAI_h_interp)
    return y


if __name__ == '__main__':

    parser = utils.define_parser()
    args = parser.parse_args()
    exp_path ='/jetfs/shared-data/ICON-LAM-DWD/exp_2'
    save_date = '16_05_2024_'
    namecode = args.ELFnamecode
    save_name = '/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Results_Final/64_ELF_IR1_0612_' + namecode
    read_stored_file = True
    RTPP = False
    input_form = np.array(range(4, 65, 6))
    testing_IR = True


    #Profile to see what takes longest:
    pr = cProfile.Profile()
    pr.enable()
    rep_list_T = [] #List of report values for each C matrix - like a coordinate key.
    rep_list_Q = []
    plev_list_T = [] #List of pressure levels for each C matrix, also like a key.
    plev_list_Q = []
    C_list_T = [] #List of C matrices for each radiosonde, ready to concatenate and manipulate.
    C_list_Q = []
    dy_list_T = [] #List of dy vectors for each radiosonde, ready to concatenate and manipulate.
    dy_list_Q = []
    e_o_list = []
    incr_list_T = []
    mean_list_T = []
    incr_list_Q = []
    mean_list_Q = []
    failure_list = [] #List of radiosonde locations where there were not enough observations to analyse.#
    start_date = int(args.ELFstartdate)
    end_date = int(args.ELFenddate)
    ana_days = [f'202306{str(i).zfill(2)}' for i in range(start_date, end_date + 1)]
    prev_days = [f'202306{str(i).zfill(2)}' for i in range(start_date - 1, end_date)]
    print(ana_days)
    for ind, ana_day in enumerate(ana_days): # 
        if ana_day == '20230601':
            if testing_IR == True:
                Times = [6, 12]
            else:
                Times = [6, 9, 12, 15]
        else:
            if testing_IR == True:
                Times = [0, 6, 12]
            else:
                Times = [6, 9, 12, 15]
        for Time in Times:
            if Time == 0:
                ana_time = ana_day + '000000'
                dir_time_sonde = prev_days[ind] + '210000'
                sonde_time = prev_days[ind] + '230000'
            elif Time == 6:
                ana_time = ana_day + '060000'
                dir_time_sonde = ana_day + '030000'
                sonde_time = ana_day + '050000'
            elif Time == 9:
                ana_time = ana_day + '090000'
                dir_time_sonde = ana_day + '060000'
                sonde_time = ana_day + '080000'
            elif Time == 12:
                ana_time = ana_day + '120000'
                dir_time_sonde = ana_day + '090000'
                sonde_time = ana_day + '110000'
            else:
                ana_time = ana_day + '150000'
                dir_time_sonde = ana_day + '120000'
                sonde_time = ana_day + '140000'
            obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, ana_time, ana_time)
            #obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
            obs_path_TEMP1 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time_sonde, sonde_time)
            obs_path_TEMP2 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, ana_time, ana_time)
            ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)
            #Get list of radiosondes to test against.
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
            #Ready to store errors. Will be a list of lists, where each list corresponds to a radiosonde and the elements are the errors for each pressure level/nominal height.
            #Maybe read within loop. Depends on memory.
            print("read big data files")
            mean = paiut.read_grib(ana_path, vars) 
            #ens = paiut.read_grib_mf(ana_path, ana_time, vars)
            for i in range(report_number): #range(len(sondes_T.reports())):
                start_time = time.time()
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
                obslon = obs_sonde_T.obs(param="lon")[0]
                obslat = obs_sonde_T.obs(param="lat")[0]
            
                print(f"Day {ana_time}, Sonde {i}, report {rep}, lat {obslat}, lon {obslon}")
                infl_reg_latlon = get_local_regions(obslon, obslat, h_loc=25.0)
                #[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
                print("get obs batch")
                obs_batch = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon, obsvar="RAWBT", new_loc = False, RAWBT_choice=1)
                print(len(obs_batch))
                if len(obs_batch) == 0:
                    print('NO SATELLITE OBS HERE')
                    failure_list.append((ana_time, i))
                    continue
                print("isolate data")
                #mean = paiut.read_grib(ana_path, vars) 
                #ens = paiut.read_grib_mf(ana_path, ana_time, vars) 
                var = 't'
                buffer = 0.04
                mean2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, mean.clon, mean.clat, mean)
                mean2 = mean2.isel(time=0)
                if mean2.t.shape[1] == 0:
                    print('NO ANALYSIS HERE')
                    continue
                print(len(mean2.cell.values))
                
                if read_stored_file == False:
                    ens = paiut.read_grib_mf(ana_path, ana_time, vars)
                    ens2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, ens.clon, ens.clat, ens)
                    ens2 = ens2.isel(time=0)
                    ensperturb = ens2 - mean2
                    del ens
                    del mean
                elif RTPP == True:
                    if os.path.isfile('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/tilde_X_a_' + ana_time + '_' + str(i)) == False:
                        continue
                    else:
                        ensperturb = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/tilde_X_a_' + ana_time + '_' + str(i))
                        ensperturb = ensperturb - ensperturb.mean(dim = 'ens')
                elif RTPP == False:
                    if os.path.isfile('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(i)) == False:
                        continue
                    else:
                        ensperturb = xr.open_dataset('/jetfs/home/a12233665/pai-munich-vienna/Analysis_Forecast_Pert_Files/X_a_' + ana_time + '_' + str(i))
                        ensperturb = ensperturb - ensperturb.mean(dim = 'ens')
                #del mean
                #del ens

                #Extract values now so that we don't need to every time we call PAI.
                pressure_levels = mean2.pres.values
                lats = ensperturb.clat.values
                lons = ensperturb.clon.values
                perturbvars = [ensperturb[var].values for var in vars]
                perturbvars.append(ensperturb.pres.values)
                #Get actual increment (approximated by PAI) attributed to radiances in this area (for error calculations later).
                print('get PAI')
                #for obs in obs_batch:
                #    obs['vloc'] = 20
                #    obs['obspres'] = 70000
                incr_standard = pai_loop_over_obs_slow(
                        obs_batch,
                        ['t', 'q', 'pres'],
                        lons,
                        lats,
                        perturbvars,
                        pressure_levels,
                        40,
                    )
                incr_standard_T = incr_standard[0][1]
                incr_standard_Q = incr_standard[1][1]
                incr_standard_P = incr_standard[2][1]#incr_noloc_P = incr_noloc[2][1]

                incr_no_obs_T = mean2.t.values - incr_standard_T
                incr_no_obs_Q = mean2.q.values - incr_standard_Q
                incr_no_obs_P = mean2.pres.values - incr_standard_P

                #Now use no localisation (artificially) to get the ELF:

                for obs in obs_batch:
                    obs['vloc'] = 20
                    obs['obspres'] = 70000

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

                #Get mean value and increment for plotting
                incr_h_interp_T = LinearNDInterpolator(list(zip(lats, lons)), incr_noloc_T.T)(obslat, obslon)
                incr_list_T.append(incr_h_interp_T)

                mean_h_interp_T = LinearNDInterpolator(list(zip(lats, lons)), (mean2.t.values).T)(obslat, obslon)
                mean_list_T.append(mean_h_interp_T)

                incr_h_interp_Q = LinearNDInterpolator(list(zip(lats, lons)), incr_noloc_Q.T)(obslat, obslon)
                incr_list_Q.append(incr_h_interp_Q)

                mean_h_interp_Q = LinearNDInterpolator(list(zip(lats, lons)), (mean2.q.values).T)(obslat, obslon)
                mean_list_Q.append(mean_h_interp_Q)

                #Get y increment for observation, including retrievals of specific humidity from radiosondes.
                print('Get dy')
                y_eq_no_obs_T = y_equivalent(incr_no_obs_T.T, lats, lons, pressure_levels.T, obslat, obslon, np.flip(obs_sonde_T.obs(param = 'plevel')))
                dy = np.flip(obs_sonde_T.obs()) - y_eq_no_obs_T
                dy_list_T.append(dy)

                y_eq_no_obs_Q = y_equivalent(incr_no_obs_Q.T, lats, lons, pressure_levels.T, obslat, obslon, np.flip(obs_sonde_RH.obs(param = 'plevel'))[0:(min(len(obs_sonde_RH.obs()), len(obs_sonde_T.obs())))])
                obs_mixing_ratio = np.array([mixing_ratio_from_relative_humidity(np.flip(obs_sonde_RH.obs(param='plevel'))[i]/100 * units.hPa,
                                                                                (np.flip(obs_sonde_T.obs())[i] - 273.15) * units.degC,
                                                                                np.flip(obs_sonde_RH.obs())[i]).to('g/kg') for i in range(min(len(obs_sonde_RH.obs()), len(obs_sonde_T.obs())))]
                                                                                )
                obs_q = np.array([specific_humidity_from_mixing_ratio(obs_mr * units('kg/kg')) for obs_mr in obs_mixing_ratio])
                dy = obs_q - y_eq_no_obs_Q
                dy_list_Q.append(dy)

                #Get C matrix for radiosonde location.
                print('Get C')
                C_T = C_Matrix(incr_noloc_T.T, lats, lons, pressure_levels.T, obslat, obslon, np.flip(obs_sonde_T.obs(param = 'plevel')), input_form = input_form)
                C_list_T.append(C_T)
                C_Q = C_Matrix(incr_noloc_Q.T, lats, lons, pressure_levels.T, obslat, obslon, np.flip(obs_sonde_RH.obs(param = 'plevel')), input_form = input_form)
                C_list_Q.append(C_Q)
                rep_list_T.append([(ana_time, rep) for i in range(len(obs_sonde_T.obs()))])
                rep_list_Q.append([(ana_time, rep) for i in range(len(obs_sonde_RH.obs()))])
                plev_list_T.append(np.flip(obs_sonde_T.obs(param = 'plevel')))
                plev_list_Q.append(np.flip(obs_sonde_RH.obs(param = 'plevel')))
                print(f"{(time.time() - start_time)} seconds to work through report {rep} at time {ana_time}")

            #print("Check: should be 0 here:" + str(np.multiply(C, np.ones(pressure_levels.shape[0])) - y_eq_no_obs))
    np.save(save_name + '_C_Ts_' + save_date, np.concatenate(C_list_T, axis = 1))
    np.save(save_name + '_y_Ts_' + save_date, np.array(np.concatenate(dy_list_T)))
    np.save(save_name + '_C_Qs_' + save_date, np.concatenate(C_list_Q, axis = 1))
    np.save(save_name + '_y_Qs_' + save_date, np.array(np.concatenate(dy_list_Q)))
    np.save(save_name + '_incrs_' + save_date, np.array(incr_list_T))
    np.save(save_name + '_means_T_' + save_date, np.array(mean_list_T))
    np.save(save_name + '_incrs_Q_' + save_date, np.array(incr_list_Q))
    np.save(save_name + '_means_Q_' + save_date, np.array(mean_list_Q))
    np.save(save_name + '_reps_T_' + save_date, np.array(rep_list_T))
    np.save(save_name + '_reps_Q_' + save_date, np.array(rep_list_Q))
    np.save(save_name + '_plevs_T_' + save_date, np.array(plev_list_T))
    np.save(save_name + '_plevs_Q_' + save_date, np.array(plev_list_Q))


    #C_complete = np.concatenate(C_list_T, axis = 1)
    #non_zero_indices = np.where((C_complete != 0).any(axis=1))[0]
    #C_complete_non_zero = C_complete[non_zero_indices]
    #C_tild = np.matmul(C_complete_non_zero, C_complete_non_zero.T)
    #y_complete = np.concatenate(dy_list_T, axis = 0)


    #print('Compute optimal localisation')
    #opt_loc = np.linalg.lstsq(C_tild, np.matmul(C_complete_non_zero, y_complete))[0]
    #np.save('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Smaller_opt_loc_02_04_2024', np.array([non_zero_indices, opt_loc]))
    #opt_loc_2 = np.linalg.solve(C_tild, np.matmul(C_complete_non_zero, y_complete))
    #np.save('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Smalleropt_loc_2_02_04_2024', np.array([non_zero_indices, opt_loc_2]))
            