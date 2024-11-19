##NOTE changed the relative humidity calculation. Could throw things off.
import os
#os.system(f'source env/bin/activate')
#from numba import njit
#os.system('deactivate')
os.chdir('/jetfs/home/a12233665/pai-munich-vienna/')
#Remember to load anaconda, enstools and maybe upgrade tbb or  NUMBA_THREADING_LAYER='omp' python PAI_Optimize_GC.py ???
import sys
sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/')
#sys.path.insert(0, '/jetfs/home/a12233665/pai-munich-vienna/pai/env/lib/python3.8/site-packages')
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import xarray as xr
from pathlib import Path
import gc

from numba import njit, config
config.THREADING_LAYER = 'omp' #'threadsafe'
from kendapy.ekf import Ekf
from enstools.io import read, write

import pai.observation as oi
import pai.localization as loc
from pai.localization import get_dist_from_obs
import pai.pai_utils as paiut
import pai.partial_analysis_increment_np as PAI
import pai.plot_oi_utils as utils

import cartopy.crs as ccrs

import cProfile, pstats, io
from pstats import SortKey
import time

from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

def average(arr, n):
    remainder = len(arr) % n
    if remainder == 0:
        avg = np.mean(arr.reshape(-1, n), axis=1)
        #avg = np.repeat(avg, n)
        return avg
    else:
        #avg_head = np.mean(arr[:-remainder].reshape(-1, n), axis=1)
        #avg_tail = np.mean(arr[-remainder:])
        avg_head = np.mean(arr[:remainder])
        avg_tail = np.mean(arr[remainder:].reshape(-1, n), axis=1)
        #avg_head = np.repeat(avg_head, n)
        #avg_tail = np.repeat(avg_tail, remainder)
        return np.append(avg_head, avg_tail)

def std_average(arr, n):
    #Input a list of standard deviations and return the standard deviations of the averaged vector.
    arr_sq = arr**2 #Convert to variances
    remainder = len(arr) % n
    if remainder == 0:
        avg_sq = np.mean(arr_sq.reshape(-1, n), axis=1) #Take the average variance
        #avg_sq = np.repeat(avg_sq, n)
        return np.sqrt(avg_sq/n) #Readjust for variance scaling
    else:
        #avg_head = np.mean(arr[:-remainder].reshape(-1, n), axis=1)
        #avg_tail = np.mean(arr[-remainder:])
        avg_head = np.mean(arr_sq[:remainder])
        avg_tail = np.mean(arr_sq[remainder:].reshape(-1, n), axis=1)
        #avg_head = np.repeat(avg_head, n)
        #avg_tail = np.repeat(avg_tail, remainder)
        return np.append(np.sqrt(avg_head/remainder), np.sqrt(avg_tail/n))

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
        for l in range(len(vars)):
            incr[l][1] += PAI.compute_increment(kcol[l], obs["fgmeandep"])
    return incr

def pai_loop_over_obs_slow_surface_peak(obs_list, vars, enslon, enslat, ep_values, mean_pres, nens, surface_vloc=0.05):
    incr = [[vars[i], np.zeros((ep_values[i].shape[1], ep_values[i].shape[2]))] for i in range(len(vars))]
    #ep_values = [ens_perturb[var].values for var in vars]
    surface_vgc = PAI.compute_vloc(1000, surface_vloc, mean_pres / 100.0)
    for i, obs in enumerate(obs_list):
        hgc = PAI.compute_hloc(obs["obslat"], obs["obslon"], obs["hloc"], enslat, enslon)

        if obs["vloc"] < 10:
            surface_influence = PAI.compute_vloc(950, surface_vloc, np.array([obs["obspres"]]) / 100.0)
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

    if args.midnight == 'Midnight':
        midnight = True
    else:
        midnight = False

    read_stored_file = True
    RTPP = False
    Surface_Peak = False
    surface_vloc = 0.05
    averaging_n = 6

    exp_path = args.basedir #'/jetfs/shared-data/ICON-LAM-DWD'
    ana_time = args.startdate + args.time #'20230601120000'  # analysis time of first guess forecasts
    dir_time = args.startdate + args.time #'20230601120000'  # directory
    if midnight:
        dir_time2 = args.prevdate + args.prevtime
        sonde_time = args.prevdate + args.sondetime #Time of verifying radiosondes
    else:
        dir_time2 = args.startdate + args.prevtime
        sonde_time = args.startdate + args.sondetime #Time of verifying radiosondes
    fg_time = args.startdate + args.time #'20230601120000'   # start time of first guess forecasts

    obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, dir_time, ana_time)
    #obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
    obs_path_TEMP1 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time2, sonde_time)
    obs_path_TEMP2 = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, ana_time, ana_time)
    ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)
    fc_path = '{}/{}/fc_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)



    #Profile to see what takes longest:
    pr = cProfile.Profile()
    pr.enable()

    #Get list of radiosondes to test against.
    sondes_RH1 = paiut.get_ekf(obs_path_TEMP1, "RH", active = False)
    sondes_T1 = paiut.get_ekf(obs_path_TEMP1, "T", active = False)
    if os.path.isfile(obs_path_TEMP2) == True:
        sondes_RH2 = paiut.get_ekf(obs_path_TEMP2, "RH", active = False)
        sondes_T2 = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
        report_number = len(sondes_RH1.reports()) + len(sondes_RH2.reports())
    else:
        report_number = len(sondes_RH1.reports())
    #sondes_RH2 = paiut.get_ekf(obs_path_TEMP2, "RH", active = False)
    #sondes_T2 = paiut.get_ekf(obs_path_TEMP2, "T", active = False)
    vars = ['t', 'q']
    #Ready to store errors. Will be a list of lists, where each list corresponds to a radiosonde and the elements are the errors for each pressure level/nominal height.
    error_list = []
    error_list_no_obs = []
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
        error_list.append([])
        #Retrieve information for individual report.
        #rep = sondes_T.reports()[i]
        #obs_sonde_RH = paiut.get_ekf(obs_path_TEMP, "RH")
        #obs_sonde_T = paiut.get_ekf(obs_path_TEMP, "T")
        obs_sonde_RH.add_filter(filter=f"report={rep}")
        obs_sonde_T.add_filter(filter=f"report={rep}")
        obslon = obs_sonde_RH.obs(param="lon")[0]
        obslat = obs_sonde_RH.obs(param="lat")[0]
        
        print(f"Sonde {i}, report {rep}, lat {obslat}, lon {obslon}")

        #Get region close to the radiosonde and restrict analysis and radiances to this region.
        rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

        infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=25.0)

        infl_reg_latlon = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]
        
        print("get obs batch")
        obs_batch = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon, obsvar="RAWBT", RAWBT_choice=1, new_loc = False)#,  RTPP_correction = RTPP)
        print(len(obs_batch))
        if len(obs_batch) == 0:
            continue

        print("read data")
        buffer = 0.04
        mean = paiut.read_grib(ana_path, vars) 
        mean2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, mean.clon, mean.clat, mean)
        mean2 = mean2.isel(time=0)
        print(len(mean2.cell.values))
        if mean2.t.shape[1] == 0:
            continue
        
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
    
        #Get the closest lat lon for calculations later (replace by proper interpolation)

        #h_loc = 15
        #dist = get_dist_from_obs(
        #    np.array(mean2.clat), np.array(mean2.clon), obslat, obslon, h_loc #np.array(ana_mean.clat), np.array(ana_mean.clon), float(obs.data.lat), float(obs.data.lon), h_loc
        #)

        #closest = np.argmin(dist)
        #Extract values now so that we don't need to every time we call PAI.
        pressure_levels = mean2.pres.values
        lats = ensperturb.clat.values
        lons = ensperturb.clon.values
        print(lats, lons, [(obs['obslat'], obs['obslon']) for obs in obs_batch])
        perturbvars = [ensperturb[var].values for var in vars]
        perturbvars.append(ensperturb.pres.values)
        #Get actual increment (approximated by PAI) attributed to radiances in this area (for error calculations later).
        print('get actual increment t')
    
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
        
        print(mean2.t.values.shape, incr_REAL_T.shape)
        t_no_obs = mean2.t.values - incr_REAL_T
        q_no_obs = mean2.q.values -  incr_REAL_Q
        p_no_obs = mean2.pres.values - incr_REAL_P

        #Get error without obs for plotting
        mean_t_no_obs = LinearNDInterpolator(list(zip(lats, lons)), t_no_obs.T)(obslat, obslon)
        mean_q_no_obs = LinearNDInterpolator(list(zip(lats, lons)), q_no_obs.T)(obslat, obslon)
        mean_p_no_obs = LinearNDInterpolator(list(zip(lats, lons)), p_no_obs.T)(obslat, obslon)
        #Interpolate to radiosonde pressure levels. Note that we only use the top 10 pressure levels as the radiosonde doesn't need much more.
        no_obs_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs, mean_t_no_obs)
        no_obs_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs, mean_q_no_obs)

        no_obs_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i]/100 * units.hPa, 
                                                                                (no_obs_t_obs_plevs[i] - 273.15) * units.degC, 
                                                                                no_obs_q_obs_plevs[i] * units('kg/kg')).to('percent').magnitude / 100 for i in range(len(no_obs_t_obs_plevs))])
                
        no_obs_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), mean_p_no_obs, mean_t_no_obs)

        #Calculate error: L2 norm of difference between actual increment and PAI increment, weighted by radiosonde errors.
        #errs_RH = (no_obs_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))/np.flip(obs_sonde_RH.obs(param = 'e_o'))
        #errs_RH_averaged = np.mean(errs_RH[:(len(errs_RH)//3)*3].reshape(-1,3), axis=1)
        devs_RH = (no_obs_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))
        errs_RH = devs_RH / np.flip(obs_sonde_RH.obs(param = 'e_o'))
        devs_RH_averaged = average(devs_RH, averaging_n)
        stds_RH_averaged = std_average(np.flip(obs_sonde_RH.obs(param = 'e_o')), averaging_n)
        errs_RH_averaged = devs_RH_averaged/stds_RH_averaged
        #errs_T = (no_obs_T_obs_plevs - np.flip(obs_sonde_T.obs()))/np.flip(obs_sonde_T.obs(param = 'e_o'))
        #errs_T_averaged = np.mean(errs_T[:(len(errs_T)//3)*3].reshape(-1,3), axis=1)
        #errs_T_averaged = average(errs_T, 6)
        devs_T = (no_obs_T_obs_plevs - np.flip(obs_sonde_T.obs()))
        errs_T = devs_T / np.flip(obs_sonde_T.obs(param = 'e_o'))
        devs_T_averaged = average(devs_T, averaging_n)
        stds_T_averaged = std_average(np.flip(obs_sonde_T.obs(param = 'e_o')), averaging_n)
        errs_T_averaged = devs_T_averaged/stds_T_averaged
        error_list_no_obs.append([ana_time, 
                            rep, 
                            np.linalg.norm(errs_RH)**2,
                            np.linalg.norm(errs_T)**2,
                            np.linalg.norm(errs_RH_averaged)**2,
                            np.linalg.norm(errs_T_averaged)**2]
        )
        
        #Run over all possible vertical localisation settings and store error
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
        print(f"{(time.time() - start_time)} seconds to prepare data")
        counter = 0
        for j, vloc in enumerate(np.concatenate([0.01*np.array(range(1, 8)), 0.02*np.array(range(4, 9)), 0.05*np.array(range(4, 8)), 0.05*np.array(range(8, 30, 4))])): #enumerate(np.concatenate([0.01*np.array(range(1, 10)), 0.05*np.array(range(2, 8)), 0.05*np.array(range(8, 50, 4))])): #enumerate(np.concatenate([0.01*np.array(range(1, 10)), 0.05*np.array(range(2, 50, 4))])):#enumerate(0.05*np.concatenate([np.array(range(1, 20)), np.array(range(20, 50, 4))])):
            for k, obspres in enumerate(range(20000, 101000, 5000)):
                print(counter)
                #print(17*j + k)
                #Update the observation batch with the new vertical location and pressure level.
                #for obs in obs_batch:
                #    obs['vloc'] = vloc
                #    obs['obspres'] = obspres
                #Calculate PAI
                start_time = time.time()
                #print('PAI t')
                #if Surface_Peak:
                #    incr = pai_loop_over_obs_slow_surface_peak(
                #        obs_batch,
                #        ['t', 'q', 'pres'],
                #        lons,
                #        lats,
                #        perturbvars,
                #        pressure_levels,
                #        40,
                #    )
                #else:
                #    incr = pai_loop_over_obs_slow(
                #        obs_batch,
                #        ['t', 'q', 'pres'],
                #        lons,
                #        lats,
                #        perturbvars,
                #        pressure_levels,
                #        40,
                #    )
                if Surface_Peak:
                    surface_vgc = PAI.compute_vloc(950, surface_vloc, pressure_levels / 100.0)
                    atmos_vgc = PAI.compute_vloc(obspres / 100.0, vloc, pressure_levels / 100.0)
                    surface_influence_atmos = PAI.compute_vloc(950, surface_vloc, np.array([obspres]) / 100.0)
                    atmos_influence_surface = PAI.compute_vloc(obspres / 100.0, vloc, np.array([950]))
                    c_surface = (1 - atmos_influence_surface)/(1 - (atmos_influence_surface * surface_influence_atmos))
                    c_atmos = (1 - surface_influence_atmos)/(1 - (atmos_influence_surface * surface_influence_atmos))
                    vgc = (c_atmos * atmos_vgc) + (c_surface * surface_vgc)
                else:
                    vgc = PAI.compute_vloc(obspres / 100.0, vloc, pressure_levels / 100.0)
                incr_T = incr_noloc_T*vgc
                incr_Q = incr_noloc_Q*vgc
                incr_P = incr_noloc_P*vgc 
                #incr_T = incr[0][1]
                #incr_Q = incr[1][1]
                #incr_P = incr[2][1]
                #del incr
                #print(f"{(time.time() - start_time)} seconds to run PAI")
                loc_t = mean2.t.values + incr_T - incr_REAL_T
                loc_q = mean2.q.values + incr_Q - incr_REAL_Q
                loc_p = mean2.pres.values + incr_P - incr_REAL_P
                print('start interpolating')
                with_loc_t = LinearNDInterpolator(list(zip(lats, lons)), loc_t.T)(obslat, obslon)
                with_loc_q = LinearNDInterpolator(list(zip(lats, lons)), loc_q.T)(obslat, obslon)
                with_loc_p = LinearNDInterpolator(list(zip(lats, lons)), loc_p.T)(obslat, obslon)
                print(with_loc_t.shape)
                with_loc_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), with_loc_p, with_loc_t)
                with_loc_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), with_loc_p, with_loc_q)

                with_loc_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i]/100 * units.hPa,
                                                                                        (with_loc_t_obs_plevs[i] - 273.15) * units.degC, 
                                                                                        with_loc_q_obs_plevs[i] * units('kg/kg')).to('percent').magnitude / 100 for i in range(len(with_loc_t_obs_plevs))])
                
                with_loc_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), with_loc_p, with_loc_t)
                #Interpolate to radiosonde pressure levels.
                #Calculate error: L2 norm of difference between actual increment and PAI increment, weighted by radiosonde errors.
                #errs_RH = (with_loc_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))/np.flip(obs_sonde_RH.obs(param = 'e_o'))
                #errs_RH_averaged = np.mean(errs_RH[:(len(errs_RH)//3)*3].reshape(-1,3), axis=1)
                #errs_RH_averaged = average(errs_RH, 6)
                #errs_T = (with_loc_T_obs_plevs - np.flip(obs_sonde_T.obs()))/np.flip(obs_sonde_T.obs(param = 'e_o'))
                #errs_T_averaged = np.mean(errs_T[:(len(errs_T)//3)*3].reshape(-1,3), axis=1)
                #errs_T_averaged = average(errs_T, 6)
                devs_RH = (with_loc_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))
                errs_RH = devs_RH / np.flip(obs_sonde_RH.obs(param = 'e_o'))
                devs_RH_averaged = average(devs_RH, 6)
                stds_RH_averaged = std_average(np.flip(obs_sonde_RH.obs(param = 'e_o')), averaging_n)
                errs_RH_averaged = devs_RH_averaged/stds_RH_averaged
                devs_T = (with_loc_T_obs_plevs - np.flip(obs_sonde_T.obs()))
                errs_T = devs_T / np.flip(obs_sonde_T.obs(param = 'e_o'))
                devs_T_averaged = average(devs_T, 6)
                stds_T_averaged = std_average(np.flip(obs_sonde_T.obs(param = 'e_o')), averaging_n)
                errs_T_averaged = devs_T_averaged/stds_T_averaged
                error_list[i].append([vloc, 
                                    obspres, 
                                    np.linalg.norm(errs_RH)**2,
                                    np.linalg.norm(errs_T)**2,
                                    np.linalg.norm(errs_RH_averaged)**2,
                                    np.linalg.norm(errs_T_averaged)**2,
                                    ])
                #print(np.linalg.norm((new_mean_RH_obs_plevs - np.flip(obs_sonde_RH.obs()))/(np.flip(obs_sonde_RH.obs(param = 'e_o'))**2)),
                #      np.linalg.norm((new_mean_T_obs_plevs - np.flip(obs_sonde_T.obs()))/(np.flip(obs_sonde_T.obs(param = 'e_o'))**2)))

                counter += 1
                print(f"{(time.time() - start_time)} seconds to run PAI")
                del incr_T
                del incr_Q
                del incr_P
                gc.collect()
    np.save('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Results_Final/No_RTPP_' + str(ana_time) + 'IR1_Result_20_04_2024_', error_list)
    np.save('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Results_Final/No_RTPP_Final_no_sat_errors_IR1_' + str(ana_time), error_list_no_obs)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print(s.getvalue())