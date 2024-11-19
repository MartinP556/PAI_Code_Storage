from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

from kendapy.ekf import Ekf
from enstools.io import read, write

#import pai.localization as loc
from pai.localization import get_dist_from_obs, gaspari_cohn, vertical_dist, haversine_distance

from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from pai.pai_utils import read_grib, get_ekf
import pai.pai_utils as paiut
from pai.plot_oi_utils import compute_potential_temperature

def get_global_averages(location_start, file_code, exp_path, ana_days, no_obs_error_location, verticals, widths, failure_list=[('20230603120000', 7), ('20230602120000', 8), ('20230601120000', 8)]):
    vloc_num = len(verticals)
    obspres_num = len(widths)
    Global_Errs_Halfs = np.zeros((2, 4, obspres_num, vloc_num)) #Store the sums of all the errors here
    Global_Errs = np.zeros((4, obspres_num, vloc_num))
    for j, vloc in enumerate(verticals):
        for k, obspres in enumerate(widths):
            Global_Errs[0, k, j] = vloc
            Global_Errs[1, k, j] = obspres
            Global_Errs_Halfs[0, 0, k, j] = vloc
            Global_Errs_Halfs[1, 0, k, j] = vloc
            Global_Errs_Halfs[0, 1, k, j] = obspres
            Global_Errs_Halfs[1, 1, k, j] = obspres
    no_obs_err_total = np.zeros((2))
    no_obs_err_onoff = np.zeros((2, 2))
    no_obs_err_list = np.load(no_obs_error_location, allow_pickle = True)
    no_obs_err_counter = 0

    for ana_day in ana_days:
        dir_time = ana_day + '120000'  # directory
        veri_time = ana_day + '110000'   # time of verifying radiosonde observations
        ana_time = ana_day + '120000'  # analysis time of first guess forecasts
        print(ana_time)
        obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
        ekf_TEMP1 = get_ekf(obs_path_TEMP, 'RH')
        half_counter = 0
        for i, rep in enumerate(ekf_TEMP1.reports()):#, 32, 3, 37, 8, 11, 14, 20, 26, 29]):
            #print(i)
            if (ana_time, i) in failure_list:
                continue
            err = np.load(location_start + ana_time + file_code + str(rep) + ".npy", allow_pickle = True)
            if np.isnan(np.array(err[i])[0, 3]):
                no_obs_err_counter += 1
                print('SCHEISSE! NAN!' + ana_day + str(rep))
                continue
            for j, vloc in enumerate(verticals):
                for k, obspres in enumerate(widths):
                    Global_Errs[2, k, j] += np.array(err[i])[j*obspres_num + k, 2]#**2 #Temp then RH error - convention in all code. Not really notated anywhere.
                    Global_Errs[3, k, j] += np.array(err[i])[j*obspres_num + k, 3]#**2
                    Global_Errs_Halfs[half_counter % 2, 2, k, j] += np.array(err[i])[j*obspres_num + k, 2]#**2
                    Global_Errs_Halfs[half_counter % 2, 3, k, j] += np.array(err[i])[j*obspres_num + k, 3]#**2
                    
            no_obs_err_total[0] += float(no_obs_err_list[no_obs_err_counter][2])
            no_obs_err_total[1] += float(no_obs_err_list[no_obs_err_counter][3])
            no_obs_err_onoff[half_counter % 2, 0] += float(no_obs_err_list[no_obs_err_counter][2])
            no_obs_err_onoff[half_counter % 2, 1] += float(no_obs_err_list[no_obs_err_counter][3])
            no_obs_err_counter += 1
            half_counter += 1

    return Global_Errs, Global_Errs_Halfs, no_obs_err_total, no_obs_err_onoff
        

def plot_sensitivity(Errs, Errs_Halfs, save_name, verticals, widths, use_no_obs_err = False, no_obs_err = np.zeros((2, 2)), no_obs_err_half = np.zeros((2, 2)), half_1_name = 'Half 1', half_2_name = 'Half 2', Title = 'Sensitivity Plot'):
    vloc_num = len(widths)
    obspres_num = len(verticals)

    fig, axs = plt.subplots(3, 2, figsize = (10, 30))
    fig.tight_layout()
    for j in range(vloc_num):
        axs[0, 0].plot(np.flip(Errs[2, :, j]), 
                    np.flip(verticals)/100, 
                    color = str(j/vloc_num), 
                    label = "Width = " + str(np.round(Errs[0, 0, j], decimals = 2))
                   )
        axs[0, 0].invert_yaxis()
        axs[0, 0].set_title('Average error to T over 3 days', fontsize = 15)
        axs[0, 0].tick_params(labelsize = 13)
    
        axs[0, 1].plot(np.flip(Errs[3, :, j]), 
                    np.flip(verticals)/100, 
                    color = str(j/vloc_num), 
                   )
        axs[0, 1].invert_yaxis()
        axs[0, 1].set_title('Average error to RH over 3 days', fontsize = 15)
        axs[0, 1].tick_params(labelsize = 13)
        if use_no_obs_err:
            axs[0, 0].axvline(x = float(no_obs_err[0]),
                            color = 'g', 
                            linewidth = 2, linestyle = 'dashed'
                            )
            axs[0, 1].axvline(x = float(no_obs_err[1]),
                            color = 'g', 
                            linewidth = 2, linestyle = 'dashed'
                            )
        for half in range(2):
            axs[half + 1, 0].plot(np.flip(Errs_Halfs[half, 2, :, j]), 
                    np.flip(verticals)/100, 
                    color = str(j/vloc_num)
                   )
            axs[half + 1, 0].invert_yaxis()
            axs[half + 1, 0].set_title('Average error to T ' + [half_1_name, half_2_name][half], fontsize = 15)
            axs[half + 1, 0].tick_params(labelsize = 13)
    
            axs[half + 1, 1].plot(np.flip(Errs_Halfs[half, 3, :, j]), 
                           np.flip(verticals)/100, 
                           color = str(j/vloc_num), 
                          )
            axs[half + 1, 1].invert_yaxis()
            axs[half + 1, 1].set_title('Average error to RH half ' + [half_1_name, half_2_name][half], fontsize = 15)
            axs[half + 1, 0].tick_params(labelsize = 13)
            if use_no_obs_err:
                axs[half + 1, 0].axvline(x = float(no_obs_err_half[half, 0]),
                                color = 'g', 
                                linewidth = 2, linestyle = 'dashed'
                                )
                axs[half + 1, 1].axvline(x = float(no_obs_err_half[half, 1]),
                                color = 'g', 
                                linewidth = 2, linestyle = 'dashed'
                                )
            axs[half + 1, 0].invert_yaxis
    
    fig.legend(bbox_to_anchor=(1.2, 1), fontsize = 14)
    plt.subplots_adjust(left=0.03,
                        bottom=0.03, 
                        right=0.97, 
                        top = 0.95,
                        wspace=0.4, 
                        hspace=0.15)
    fig.suptitle(Title, fontsize = 20)
    plt.savefig('/jetfs/home/a12233665/pai-munich-vienna/Martin Scripts/Plots/' + save_name, bbox_inches='tight')

def plot_all_errors_vertical(results_location, no_obs_errs_location, exp_path, dir_time, ana_time, veri_time, vloc_num, obspres_num, output_location, fof = False, failure_list = [('20230603120000', 7), ('20230602120000', 8), ('20230601120000', 8)], temp_plot = False, day = 1, with_errors = True):
    obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, dir_time, ana_time)
    ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, dir_time, ana_time)
    if fof == True:
        obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
    else:
        obs_path_TEMP = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time, ana_time)
    ekf_TEMP1 = paiut.get_ekf(obs_path_TEMP, 'RH')
    ekf_RAD = paiut.get_ekf(obs_path_RAD, 'REFL')
    subtitle_size = 12
    label_size = 11

    fig, axs = plt.subplots((len(ekf_TEMP1.reports()) // 2) + 1, 6, figsize = (18, 18))
    fig.tight_layout()
    print('Getting analysis')
    ana_mean = read_grib(ana_path, ['t', 'q'])
    print('Found analysis')
    k = 0
    no_obs_errs = np.load(no_obs_errs_location, allow_pickle = True)
    for i, rep in enumerate(ekf_TEMP1.reports()): #ekf_TEMP1.reports()):#, 32, 3, 37, 8, 11, 14, 20, 26, 29]):
        print(i, rep)
        if (ana_time, i) in failure_list:
            print(f"Skipping {ana_time} {rep}")
            continue
        err = np.load(results_location + str(rep) + ".npy", allow_pickle = True)
        sondes_T = paiut.get_ekf(obs_path_TEMP, "T")
        sondes_T.add_filter(filter = f"report={rep}")
        sondes_RH = paiut.get_ekf(obs_path_TEMP, "RH")
        sondes_RH.add_filter(filter = f"report={rep}")
    
        h_loc = 15
        dist = get_dist_from_obs(
            np.array(ana_mean.clat), 
            np.array(ana_mean.clon), 
            sondes_T.obs(param = 'lat')[0], 
            sondes_T.obs(param = 'lon')[0], 
            h_loc
        )
        closest = np.argmin(dist)
    
        dist_obs = get_dist_from_obs(
            np.array(ekf_RAD.obs(param = 'lat')), 
            np.array(ekf_RAD.obs(param = 'lon')), 
            sondes_T.obs(param = 'lat')[0], 
            sondes_T.obs(param = 'lon')[0], 
            h_loc
        )
        closest_obs = np.argmin(dist_obs)
    
        mean_t_obs_plevs = np.interp(np.flip(sondes_T.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.t.values[0,:,closest])
    
    
        mean_t_obs_plevs2 = np.interp(np.flip(sondes_RH.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.t.values[0,:,closest])
        mean_q_obs_plevs = np.interp(np.flip(sondes_RH.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.q.values[0,:,closest])

        mean_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(sondes_RH.obs(param = 'plevel'))[i]/100 * units.hPa, 
                                                                                       (mean_t_obs_plevs2[i] - 273.15) * units.degC, 
                                                                                       mean_q_obs_plevs[i] * units('kg/kg')).to('percent').magnitude for i in range(len(mean_t_obs_plevs2))])
    
        for j in range(vloc_num):
            axs[i // 2, 3*(i % 2)].plot(np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 3]), 
                                        np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 1]/100), 
                                        color = str(j/vloc_num), 
                                        label = "Width = " + str(np.round(np.array(err[0])[j*obspres_num, 0], decimals = 2)))
            axs[i // 2, 3*(i % 2) + 1].plot(np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 2]), 
                                        np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 1]/100), 
                                        color = str(j/vloc_num), 
                                        )
        if with_errors:
            axs[i // 2, 3*(i % 2)].axvline(x = float(no_obs_errs[k + [0, 10, 19][day - 1]][3])**2,
                                        color = 'g', 
                                        linewidth = 2, linestyle = 'dashed'
                                        )
            axs[i // 2, 3*(i % 2) + 1].axvline(x = (float(no_obs_errs[k + [0, 10, 19][day - 1]][2])**2),
                                        color = 'g', 
                                        linewidth = 2, linestyle = 'dashed'
                                        )
        axs[i // 2, 3*(i % 2)].invert_yaxis()
        axs[i // 2, 3*(i % 2)].set_ylim([1000, 100])
        axs[i // 2, 3*(i % 2)].set_title(f"Report {rep}, error to temperature", fontsize = subtitle_size)
    
        axs[i // 2, 3*(i % 2) + 1].axhline(y = ekf_RAD.obs(param = 'plevel')[closest_obs]/100, 
                                       color = 'r', 
                                       linewidth = 2, linestyle = 'dashed', 
                                      )
    
        axs[i // 2, 3*(i % 2) + 1].axhline(y = ekf_RAD.obs(param = 'plevel')[closest_obs + 1]/100, 
                                       color = 'r', 
                                       linewidth = 2, linestyle = 'dashed'
                                      )
        
        axs[i // 2, 3*(i % 2) + 1].invert_yaxis()
        axs[i // 2, 3*(i % 2) + 1].set_ylim([1000, 100])
        axs[i // 2, 3*(i % 2) + 1].set_title(f"Report {rep}, error to RH", fontsize = subtitle_size)
    
        if temp_plot:
            axs[i // 2, 3*(i % 2) + 2].plot(compute_potential_temperature(sondes_T.obs(), sondes_T.obs(param = 'plevel')/100) - 273.15, 
                                            sondes_T.obs(param = 'plevel')/100, 
                                            color = 'r', label = 'Radiosonde temp')
            axs[i // 2, 3*(i % 2) + 2].plot(compute_potential_temperature(np.flip(mean_t_obs_plevs), sondes_T.obs(param = 'plevel')/100)- 273.15, 
                                            sondes_T.obs(param = 'plevel')/100, 
                                            color = 'g', label = 'Analysis temp')
        axs[i // 2, 3*(i % 2) + 2].plot(100*sondes_RH.obs(), 
                                        sondes_RH.obs(param = 'plevel')/100, 
                                        color = 'b', label = 'Radiosonde RH')
    
        
        axs[i // 2, 3*(i % 2) + 2].plot(np.flip(mean_RH_obs_plevs), 
                                        sondes_RH.obs(param = 'plevel')/100, 
                                        color = 'k', label = 'Analysis RH')
    
    
    
        axs[i // 2, 3*(i % 2) + 2].set_ylim([1000, 100])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        REFL = np.round(ekf_RAD.obs()[closest_obs], decimals = 2)
        print(REFL)
        axs[i // 2, 3*(i % 2) + 2].text(0.05, 0.95, 'REFL = ' + str(REFL), 
                                        transform=axs[i // 2, 3*(i % 2) + 2].transAxes, fontsize=14,
                                        verticalalignment='top', bbox=props)
    
        axs[i // 2, 3*(i % 2) + 2].set_title(f"rep {rep} temperature and RH profiles", fontsize = subtitle_size)
        if i == 0:
            fig.legend(bbox_to_anchor=(1.25, 1), fontsize = label_size)#loc = [1, 0.01], fontsize = 12)
            axs[i // 3, 3*(i % 2)].set_xlabel("Error in K", fontsize = label_size)
            axs[i // 3, 3*(i % 2) + 1].set_xlabel("Error in RH, %", fontsize = label_size)
            axs[i // 3, 3*(i % 2)].set_ylabel("Pressure in hPa", fontsize = label_size)
            axs[i // 3, 3*(i % 2) + 2].set_xlabel("Temp in C, RH in %", fontsize = label_size)
        
        #i += 1
        plt.subplots_adjust(left=0.03,
                        bottom=0.03, 
                        right=0.97, 
                        top = 1,
                        wspace=0.6, 
                        hspace=0.3)
        k += 1

    fig.suptitle("Error of the analysis for different vertical localization widths/heights REFL", y = 1.05, fontsize = 30)
    plt.savefig(output_location, bbox_inches='tight')

    fig.suptitle("Error of the analysis for different vertical localization widths/heights REFL", y = 1.05, fontsize = 30)
    plt.savefig(output_location, bbox_inches='tight')

def plot_all_errors_one_rep(results_location, no_obs_errs_location, exp_path, dir_time, ana_time, veri_time, rep, repnum, vloc_num, obspres_num, Title, output_location, fof = True, temp_plot = False, day = 1, with_errors = True, new = True):
    i = repnum
    k = repnum
    obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, dir_time, ana_time)
    ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, dir_time, ana_time)
    if fof == True:
        obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
    else:
        obs_path_TEMP = '{}/feedback/{}/ekfTEMP.nc.{}'.format(exp_path, dir_time, ana_time)
    ekf_TEMP1 = paiut.get_ekf(obs_path_TEMP, 'RH')
    ekf_RAD = paiut.get_ekf(obs_path_RAD, 'REFL')
    subtitle_size = 12
    label_size = 11

    fig, axs = plt.subplots(1, 3, figsize = (8, 4))
    fig.tight_layout()
    print('Getting analysis')
    ana_mean = read_grib(ana_path, ['t', 'q'])
    print('Found analysis')
    no_obs_errs = np.load(no_obs_errs_location, allow_pickle = True)
    if new:
        err = np.load(results_location + ".npy", allow_pickle = True)
    else:
        err = np.load(results_location + str(rep) + ".npy", allow_pickle = True)
    sondes_T = paiut.get_ekf(obs_path_TEMP, "T")
    sondes_T.add_filter(filter = f"report={rep}")
    sondes_RH = paiut.get_ekf(obs_path_TEMP, "RH")
    sondes_RH.add_filter(filter = f"report={rep}")

    h_loc = 15
    dist = get_dist_from_obs(
        np.array(ana_mean.clat), 
        np.array(ana_mean.clon), 
        sondes_T.obs(param = 'lat')[0], 
        sondes_T.obs(param = 'lon')[0], 
        h_loc
    )
    closest = np.argmin(dist)

    dist_obs = get_dist_from_obs(
        np.array(ekf_RAD.obs(param = 'lat')), 
        np.array(ekf_RAD.obs(param = 'lon')), 
        sondes_T.obs(param = 'lat')[0], 
        sondes_T.obs(param = 'lon')[0], 
        h_loc
    )
    closest_obs = np.argmin(dist_obs)

    mean_t_obs_plevs = np.interp(np.flip(sondes_T.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.t.values[0,:,closest])


    mean_t_obs_plevs2 = np.interp(np.flip(sondes_RH.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.t.values[0,:,closest])
    mean_q_obs_plevs = np.interp(np.flip(sondes_RH.obs(param = 'plevel')), ana_mean.pres.values[0,:,closest], ana_mean.q.values[0,:,closest])

    mean_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(sondes_RH.obs(param = 'plevel'))[i]/100 * units.hPa, 
                                                                                    (mean_t_obs_plevs2[i] - 273.15) * units.degC, 
                                                                                    mean_q_obs_plevs[i] * units('kg/kg')).to('percent').magnitude for i in range(len(mean_t_obs_plevs2))])

    for j in range(vloc_num):
        if (j % 3) == 0:
            axs[0].plot(np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 3]), 
                                        np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 1]/100), 
                                        color = str(j/vloc_num), 
                                        label = "Width = " + str(np.round(np.array(err[i])[j*obspres_num, 0], decimals = 2)))
        else:
            axs[0].plot(np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 3]), 
                                        np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 1]/100), 
                                        color = str(j/vloc_num), 
                                        )
        axs[1].plot(np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 2]), 
                                    np.flip(np.array(err[i])[j*obspres_num:(j+1)*obspres_num, 1]/100), 
                                    color = str(j/vloc_num), 
                                    )
    if with_errors:
        axs[0].axvline(x = float(no_obs_errs[repnum][3]),    #k + [0, 10, 19][day - 1]][3]),
                                    color = 'g', 
                                    linewidth = 2, linestyle = 'dashed',
                                    label = 'Error Satellites Removed'
                                    )
        axs[1].axvline(x = float(no_obs_errs[repnum][3]),     #[k + [0, 10, 19][day - 1]][2])),
                                    color = 'g', 
                                    linewidth = 2, linestyle = 'dashed'
                                    )
    axs[0].invert_yaxis()
    axs[0].set_ylim([1000, 100])
    axs[0].set_title("Eror to temperature", fontsize = subtitle_size)

    axs[1].axhline(y = ekf_RAD.obs(param = 'plevel')[closest_obs]/100, 
                                    color = 'r', 
                                    linewidth = 2, linestyle = 'dashed', 
                                    )

    axs[1].axhline(y = ekf_RAD.obs(param = 'plevel')[closest_obs + 1]/100, 
                                    color = 'r', 
                                    linewidth = 2, linestyle = 'dashed'
                                    )
    
    axs[1].invert_yaxis()
    axs[1].set_ylim([1000, 100])
    axs[1].set_title("Error to RH", fontsize = subtitle_size)

    if temp_plot:
        axs[2].plot(compute_potential_temperature(sondes_T.obs(), sondes_T.obs(param = 'plevel')/100) - 273.15, 
                                        sondes_T.obs(param = 'plevel')/100, 
                                        color = 'r', label = 'Radiosonde temp')
        axs[2].plot(compute_potential_temperature(np.flip(mean_t_obs_plevs), sondes_T.obs(param = 'plevel')/100)- 273.15, 
                                        sondes_T.obs(param = 'plevel')/100, 
                                        color = 'g', label = 'Analysis temp')
    axs[2].plot(100*sondes_RH.obs(), 
                                    sondes_RH.obs(param = 'plevel')/100, 
                                    color = 'b', label = 'Radiosonde RH')

    
    axs[2].plot(np.flip(mean_RH_obs_plevs), 
                                    sondes_RH.obs(param = 'plevel')/100, 
                                    color = 'k', label = 'Analysis RH')



    axs[2].set_ylim([1000, 100])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    REFL = np.round(ekf_RAD.obs()[closest_obs], decimals = 2)
    print(REFL)
    axs[2].text(0.05, 0.95, 'REFL = ' + str(REFL), 
                                    transform=axs[2].transAxes, fontsize=14,
                                    verticalalignment='top', bbox=props)

    axs[2].set_title("Temperature and RH profiles", fontsize = subtitle_size)
    fig.legend(bbox_to_anchor=(1.25, 1), fontsize = label_size)#loc = [1, 0.01], fontsize = 12)
    axs[0].set_xlabel("Error in K", fontsize = label_size)
    axs[1].set_xlabel("Error in RH, %", fontsize = label_size)
    axs[0].set_ylabel("Pressure in hPa", fontsize = label_size)
    axs[2].set_xlabel("Temp in C, RH in %", fontsize = label_size)

    plt.subplots_adjust(left=0.03,
                    bottom=0.03, 
                    right=0.97, 
                    top = 1,
                    wspace=0.6, 
                    hspace=0.3)

    fig.suptitle(Title, y = 1.25, fontsize = 23)
    plt.savefig(output_location, bbox_inches='tight')


ana_times = ['20230601120000', '20230602120000', '20230603120000'] # analysis time of first guess forecasts
dir_time = '20230603120000'  # directory
fg_time = '20230603120000'   # start time of first guess forecasts

def sort_cloud_on_off(ana_days, exp_path, obspres_list, vloc_list, errors_location, errors_suffix, failure_list, cloudvar = 'REFL', first_RAWBT = True, threshold = 0.4):
    Global_Errs_Clouds = np.zeros((2, 4, len(obspres_list), len(vloc_list))) #Store the sums of all the errors here
    Global_Errs = np.zeros((4, len(obspres_list), len(vloc_list))) #Store the sums of all the errors here
    h_loc = 25
    obspres_num = len(obspres_list)
    vloc_num = len(vloc_list)
    for j, vloc in enumerate(vloc_list):
        for k, obspres in enumerate(obspres_list):
            Global_Errs_Clouds[0, 0, k, j] = vloc
            Global_Errs_Clouds[1, 0, k, j] = vloc
            Global_Errs_Clouds[0, 1, k, j] = obspres
            Global_Errs_Clouds[0, 1, k, j] = obspres
            Global_Errs[0, k, j] = vloc
            Global_Errs[1, k, j] = obspres
    for ana_day in ana_days:
        dir_time = ana_day + '120000'
        ana_time = ana_day + '120000'
        veri_time = ana_day + '110000'
        obs_path_RAD = '/{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, ana_time, ana_time)
        obs_path_TEMP = '/{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
        ekf_SAT = get_ekf(obs_path_RAD, cloudvar)
        ekf_TEMP = get_ekf(obs_path_TEMP, 'RH')
        for i, rep in enumerate(ekf_TEMP.reports()):
            if (ana_time, i) in failure_list:
                continue
            print(i)
            err = np.load(errors_location + ana_time + errors_suffix + str(rep) + ".npy", allow_pickle = True)
            if np.isnan(np.array(err[i])[0, 3]):
                print('SCHEISSE! NAN!' + ana_day + str(rep))
                continue
            sondes_T = get_ekf(obs_path_TEMP, "T")
            sondes_T.add_filter(filter = f"report={rep}")
            sondes_RH = get_ekf(obs_path_TEMP, "RH")
            sondes_RH.add_filter(filter = f"report={rep}")

            dist_obs = get_dist_from_obs(
                np.array(ekf_SAT.obs(param = 'lat')), 
                np.array(ekf_SAT.obs(param = 'lon')), 
                sondes_T.obs(param = 'lat')[0], 
                sondes_T.obs(param = 'lon')[0], 
                h_loc
            )
            if cloudvar == 'REFL':
                closest_obs = np.argmin(dist_obs)
            elif first_RAWBT:
                closest_obs = np.argmin(dist_obs)
            else:
                closest_obs = np.argmin(dist_obs) + 1

            if ekf_SAT.obs()[closest_obs] > threshold:#As always 2 IR obs stored. Say cloudy if less than value, so the second one is cloudy.
                for j, vloc in enumerate(vloc_list):
                    for k, obspres in enumerate(obspres_list):
                        Global_Errs[2, k, j] += np.array(err[i])[j*obspres_num + k, 2]
                        Global_Errs[3, k, j] += np.array(err[i])[j*obspres_num + k, 3]
                        Global_Errs_Clouds[0, 2, k, j] += np.array(err[i])[j*obspres_num + k, 2]
                        Global_Errs_Clouds[0, 3, k, j] += np.array(err[i])[j*obspres_num + k, 3]
            else:
                for j, vloc in enumerate(vloc_list):
                    for k, obspres in enumerate(obspres_list):
                        Global_Errs[2, k, j] += np.array(err[i])[j*obspres_num + k, 2]
                        Global_Errs[3, k, j] += np.array(err[i])[j*obspres_num + k, 3]
                        Global_Errs_Clouds[1, 2, k, j] += np.array(err[i])[j*obspres_num + k, 2]
                        Global_Errs_Clouds[1, 3, k, j] += np.array(err[i])[j*obspres_num + k, 3]
    return Global_Errs, Global_Errs_Clouds

def explanatory_plot(exp_path, dir_time, ana_time, veri_time, rep, gc_plevel, gc_width, save_name, surface_vloc = 0.2, Surface_Peak = False, vars = ['t', 'q'], obs_var = 'T'):    
    obs_path_RAD = '{}/feedback/{}/ekfRAD.nc.{}'.format(exp_path, ana_time, ana_time)
    obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, dir_time, veri_time)
    ana_path = '{}/{}/an_R19B07.{}.mean'.format(exp_path, ana_time, ana_time)
    obs_sonde_RH = paiut.get_ekf(obs_path_TEMP, "RH")
    obs_sonde_T = paiut.get_ekf(obs_path_TEMP, "T")
    obs_sonde_RH.add_filter(filter=f"report={rep}")
    obs_sonde_T.add_filter(filter=f"report={rep}")
    obslon = obs_sonde_RH.obs(param="lon")[0]
    obslat = obs_sonde_RH.obs(param="lat")[0]
    #Get region close to the radiosonde and restrict analysis and radiances to this region.
    rot_lons, rot_lats = oi.location_to_rotated_pole([obslon, obslon], [obslat, obslat])

    infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=25.0)

    infl_reg_latlon = np.array(oi.rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])).flatten()#[list(array) for array in rotated_pole_to_location(infl_reg[0:2], infl_reg[2:4])]

    print("get obs batch")
    obs_batch = paiut.get_obs_batch(obs_path_RAD, *infl_reg_latlon, obsvar="REFL", new_loc = False)
    print(len(obs_batch))

    print("read data")
    var = 't'
    buffer = 0.075
    mean = paiut.read_grib(ana_path, vars) 
    ens = paiut.read_grib_mf(ana_path, ana_time, vars)
    ens2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, ens.clon, ens.clat, ens)
    mean2 = loc.find_analysis_in_area(obslon - buffer, obslon + buffer, obslat - buffer, obslat + buffer, mean.clon, mean.clat, mean)
    ensperturb = ens2 - mean2

    #Get the closest lat lon for calculations later (replace by proper interpolation)

    h_loc = 15
    dist = loc.get_dist_from_obs(
        np.array(mean2.clat), np.array(mean2.clon), obslat, obslon, h_loc #np.array(ana_mean.clat), np.array(ana_mean.clon), float(obs.data.lat), float(obs.data.lon), h_loc
    )

    closest = np.argmin(dist)
    #Extract values now so that we don't need to every time we call PAI.
    pressure_levels = mean2.pres.isel(time=0).values
    lats = ensperturb.clat.values
    lons = ensperturb.clon.values
    perturbvars = [ensperturb.isel(time = 0)[var].values for var in vars]
    perturbvars.append(ensperturb.isel(time = 0).pres.values)
    #Get actual increment (approximated by PAI) attributed to radiances in this area (for error calculations later).
    print('get actual increment t')
    incr_REAL = paiogc.pai_loop_over_obs_slow(
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
    #Get increment for specific Gaspari-Cohn
    for obs in obs_batch:
        obs['vloc'] = gc_width
        obs['obspres'] = gc_plevel
    #Calculate PAI
    #print('PAI t')
    if Surface_Peak:
        incr = paiogc.pai_loop_over_obs_slow_surface_peak(
            obs_batch,
            ['t', 'q', 'pres'],
            lons,
            lats,
            perturbvars,
            pressure_levels,
            40,
        )
    else:
        incr = paiogc.pai_loop_over_obs_slow(
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
    #Get new analysis field with PAI increment.
    mean_t_no_obs = mean2.t.values - incr_REAL_T
    mean_q_no_obs = mean2.q.values - incr_REAL_Q
    mean_p_no_obs = mean2.pres.values - incr_REAL_P
    #Interpolate to radiosonde pressure levels.

    incr_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), pressure_levels[:, closest], incr_T[:, closest])
    incr_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), pressure_levels[:, closest], incr_T[:, closest])
    incr_Q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), pressure_levels[:, closest], incr_Q[:, closest])

    incr_REAL_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), pressure_levels[:, closest], incr_REAL_T[:, closest])
    incr_REAL_Q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), pressure_levels[:, closest], incr_REAL_Q[:, closest])

    ana_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), pressure_levels[:, closest], mean2.t.values[0, :, closest])
    ana_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), pressure_levels[:, closest], mean2.q.values[0, :, closest])

    no_obs_t_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs[0,:,closest], mean_t_no_obs[0,:,closest])
    no_obs_q_obs_plevs = np.interp(np.flip(obs_sonde_RH.obs(param = 'plevel')), mean_p_no_obs[0,:,closest], mean_q_no_obs[0,:,closest])
    no_obs_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), mean_p_no_obs[0,:,closest], mean_t_no_obs[0,:,closest])

    with_loc_t_obs_plevs = no_obs_t_obs_plevs + incr_t_obs_plevs
    with_loc_q_obs_plevs = no_obs_q_obs_plevs + incr_Q_obs_plevs
    with_loc_T_obs_plevs = no_obs_T_obs_plevs + incr_T_obs_plevs

    no_obs_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i] * units.hPa, 
                                                                            (no_obs_t_obs_plevs[i] - 273.15) * units.degC, 
                                                                            no_obs_q_obs_plevs[i])/100 for i in range(len(no_obs_t_obs_plevs))])

    ana_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i] * units.hPa,
                                                                        (ana_t_obs_plevs[i] - 273.15) * units.degC,
                                                                        ana_q_obs_plevs[i])/100 for i in range(len(ana_t_obs_plevs))])

    with_loc_RH_obs_plevs = np.array([relative_humidity_from_specific_humidity(np.flip(obs_sonde_RH.obs(param = 'plevel'))[i] * units.hPa,
                                                                        (with_loc_t_obs_plevs[i] - 273.15) * units.degC,
                                                                        with_loc_q_obs_plevs[i])/100 for i in range(len(with_loc_t_obs_plevs))])
            
    no_obs_T_obs_plevs = np.interp(np.flip(obs_sonde_T.obs(param = 'plevel')), mean_p_no_obs[0,:,closest], mean_t_no_obs[0,:,closest])

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    if obs_var == 'T':
        axs[0].plot(incr_T_obs_plevs, np.flip(obs_sonde_T.obs(param = 'plevel'))/100, label='PAI', color = 'r')
        axs[0].plot(incr_REAL_T_obs_plevs, np.flip(obs_sonde_T.obs(param = 'plevel'))/100, label='Real', color = 'b')
        scale = max(max(incr_T_obs_plevs), max(incr_REAL_T_obs_plevs))
        sonde_error_no_obs = np.flip(obs_sonde_T.obs()) - no_obs_T_obs_plevs
        sonde_error_with_loc = np.flip(obs_sonde_T.obs()) - with_loc_T_obs_plevs
        sonde_error_real_obs = np.flip(obs_sonde_T.obs()) - ana_t_obs_plevs
        obs_sonde = obs_sonde_T
    else:
        axs[0].plot(ana_RH_obs_plevs - no_obs_RH_obs_plevs, np.flip(obs_sonde_RH.obs(param = 'plevel'))/100, label='PAI', color = 'r')
        axs[0].plot(with_loc_RH_obs_plevs - no_obs_RH_obs_plevs, np.flip(obs_sonde_RH.obs(param = 'plevel'))/100, label='Real', color = 'b')
        scale = max(max(ana_RH_obs_plevs - no_obs_RH_obs_plevs), max(with_loc_RH_obs_plevs - no_obs_RH_obs_plevs))
        sonde_error_no_obs = np.flip(obs_sonde_RH.obs()) - no_obs_RH_obs_plevs
        sonde_error_with_loc = np.flip(obs_sonde_RH.obs()) - with_loc_RH_obs_plevs
        sonde_error_real_obs = np.flip(obs_sonde_RH.obs()) - ana_RH_obs_plevs
        obs_sonde = obs_sonde_RH
    if Surface_Peak:
        surface_vgc = PAI.compute_vloc(1000, surface_vloc, np.flip(obs_sonde.obs(param = 'plevel')) / 100.0)
        if obs["vloc"] < 10:
            surface_influence = PAI.compute_vloc(1000, surface_vloc, np.array([gc_plevel]) / 100.0)
            gc = (PAI.compute_vloc(gc_plevel / 100.0, gc_width, np.flip(obs_sonde.obs(param = 'plevel')) / 100.0) + surface_vgc)/(1+surface_influence)
        else:
            gc = np.ones(len(obs_sonde.obs(param = 'plevel')))
    else:
        gc =PAI.compute_vloc(gc_plevel/100, gc_width, np.flip(obs_sonde.obs(param = 'plevel'))/100)
    axs[0].plot(gc*scale, np.flip(obs_sonde.obs(param = 'plevel'))/100, label="gc, scaled by 10")
    axs[0].legend()
    axs[0].set_title('PAI original and localised Increment')
    axs[0].set_xlabel('Increment, ' + obs_var)
    axs[0].set_ylabel('Pressure Level (hPa)')


    #axs[1].plot(sonde_error_no_obs, np.flip(obs_sonde.obs(param = 'plevel'))/100, label='Radiosonde - analysis without obs, ' + obs_var, color = 'g')
    #axs[1].plot(sonde_error_with_loc, np.flip(obs_sonde.obs(param = 'plevel'))/100, label='Radiosonde - analysis with loc, ' + obs_var, color = 'black')
    axs[1].plot(np.abs(sonde_error_no_obs) - np.abs(sonde_error_with_loc), np.flip(obs_sonde.obs(param = 'plevel'))/100, label='|Error no obs| - |Error new loc|, ' + obs_var, color = 'g')
    axs[1].plot(np.abs(sonde_error_no_obs) - np.abs(sonde_error_real_obs), np.flip(obs_sonde.obs(param = 'plevel'))/100, label='|Error no obs| - |Error old loc|, ' + obs_var, color = 'purple')
    axs[1].legend()
    axs[1].set_title('Observation Benefit')
    axs[1].set_xlabel('Error ' + obs_var)


    axs[2].plot(np.flip(obs_sonde_RH.obs()), np.flip(obs_sonde_RH.obs(param = 'plevel'))/100, label='Radiosonde RH', color = 'b')
    axs[2].plot(no_obs_RH_obs_plevs, np.flip(obs_sonde_RH.obs(param = 'plevel'))/100, label='Model RH without Satellites', color = 'r')
    axs[2].legend()
    axs[2].set_title('RH plot as often useful')
    axs[2].set_xlabel('Relative Humidity, %')

    axs[0].invert_yaxis()
    axs[0].set_ylim([1000, 100])
    axs[0].axvline(x = 0, color = 'black', alpha = 0.5)
    axs[1].invert_yaxis()
    axs[1].set_ylim([1000, 100])
    axs[1].axvline(x = 0, color = 'black', alpha = 0.5)
    axs[2].invert_yaxis()
    axs[2].set_ylim([1000, 100])
    plt.savefig(save_name, bbox_inches='tight')

