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

if __name__ == '__main__':
    parser = utils.define_parser()
    args = parser.parse_args()
    exp_path = args.basedir

    read_stored_file = True
    Surface_Peak = False
    averaging_n = 6
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
            obs_batch = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon, obsvar="RAWBT", RAWBT_choice = 0, new_loc=False, RTPP_correction = False)
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
            #Change vloc and obspres settings then do it again to get the retrospectively localised version.
            for obs in obs_batch:
                obs['vloc'] = vloc
                obs['obspres'] = obspres
            incr = pai_loop_over_obs_slow(
                    obs_batch,
                    ['t', 'q', 'pres'],
                    lons,
                    lats,
                    perturbvars,
                    pressure_levels,
                    40,
                )
            incr_T = incr[0][1]
            incr_Q = incr[1][1]
            incr_P = incr[2][1]
            print(f"{(time.time() - start_time)} seconds to run PAI")
            print('calculate errors')
            #Get new analysis field with PAI increment.
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
            #Calculate error then interpolate back to plotting pressure levels:
            error_list[0].append(np.interp(vert_levels, with_loc_p, incr_T[:, closest]))
            error_list[1].append(np.interp(vert_levels, with_loc_p, incr_Q[:, closest]))
            error_list[2].append(np.interp(vert_levels, np.flip(obs_sonde_T.obs(param = 'plevel')), np.abs(no_obs_T_obs_plevs - np.flip(obs_sonde_T.obs()))))
            error_list[3].append(np.interp(vert_levels, np.flip(obs_sonde_RH.obs(param = 'plevel')), np.abs(no_obs_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))))
            error_list[4].append(np.interp(vert_levels, np.flip(obs_sonde_T.obs(param = 'plevel')), np.abs(with_loc_T_obs_plevs - np.flip(obs_sonde_T.obs()))))
            error_list[5].append(np.interp(vert_levels, np.flip(obs_sonde_RH.obs(param = 'plevel')), np.abs(with_loc_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))))
            
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
            dist_obs_VIS = get_dist_from_obs(
                np.array(ekf_RAD.obs(param = 'lat')), 
                np.array(ekf_RAD.obs(param = 'lon')), 
                obslat, 
                obslon, 
                h_loc
            )
            closest_obs_IR = np.argmin(dist_obs_IR)
            closest_obs_VIS = np.argmin(dist_obs_VIS)
            
            error_list[6].append((ana_day, i, rep, 
                                ekf_RAD.obs()[closest_obs_VIS],  ekf_RAD.fgmeandep()[closest_obs_VIS], 
                                ekf_RAWBT.obs()[closest_obs_IR], ekf_RAWBT.fgmeandep()[closest_obs_IR], 
                                ekf_RAWBT.obs()[closest_obs_IR + 1], ekf_RAWBT.fgmeandep()[closest_obs_IR + 1],
                                ekf_RAD.anameandep()[closest_obs_VIS] - ekf_RAD.fgmeandep()[closest_obs_VIS],
                                ekf_RAWBT.anameandep()[closest_obs_IR] - ekf_RAWBT.fgmeandep()[closest_obs_IR], 
                                ekf_RAWBT.anameandep()[closest_obs_IR + 1] - ekf_RAWBT.fgmeandep()[closest_obs_IR + 1]))
            gc.collect()
    namecode = args.ELFnamecode
    np.save('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Results_Final/verticals_' + namecode + ana_day, error_list)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print(s.getvalue())