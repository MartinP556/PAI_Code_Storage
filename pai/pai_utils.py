#!/usr/bin/env python3


import numpy as np
import xarray as xr
#import proplot as pplt
import cartopy.crs as ccrs
import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
import os
from enstools.io import read
from kendapy.ekf import Ekf
from pathlib import Path
#import os
#os.system(f'source env/bin/activate')
import sys
#sys.path.insert(0, '/jetfs/home/a12233665/pai-munich-vienna/pai/env/lib/python3.8/site-packages')
from numba import jit, config
config.THREADING_LAYER = 'omp' #'threadsafe'
#os.system('deactivate')
#import localization as loc
import matplotlib
matplotlib.use('agg')
def get_regions(rlonmin, rlonmax,rlatmin, rlatmax, hloc=80.):
    # distance of one degree (rotated pole --> equator)
    dist = 2 * np.pi * 6371 / 360

    # hloc = 80.0  # TODO set the max localization length scale here!!
    cutoff = 2.0 * hloc * np.sqrt(10.0 / 3.0)

    # DEBUG
    # cutoff = hloc * 1.5

    diff = cutoff / dist

    # region that contains all observations that influence the desired area
    left_lon = rlonmin - diff
    right_lon = rlonmax + diff
    lower_lat = rlatmin - diff
    upper_lat = rlatmax + diff

    # helper region for easy PAI summation
    obs_left_lon = left_lon - diff
    obs_right_lon = right_lon + diff
    obs_lower_lat = lower_lat - diff
    obs_upper_lat = upper_lat + diff


    return [left_lon, right_lon, lower_lat, upper_lat], [obs_left_lon, obs_right_lon, obs_lower_lat, obs_upper_lat]

def slice_data_rot_pole(data, rot_lon_min, rot_lon_max, rot_lat_min, rot_lat_max):

    #get the index of the rot pole coords

    lon_min_ind = np.abs(data.lon_2 - rot_lon_min).argmin().values
    lon_max_ind = np.abs(data.lon_2 - rot_lon_max).argmin().values
    lat_min_ind = np.abs(data.lat_2 - rot_lat_min).argmin().values
    lat_max_ind = np.abs(data.lat_2 - rot_lat_max).argmin().values

    #slice
    return data.isel(x=slice(lon_min_ind, lon_max_ind),
                     lon_2=slice(lon_min_ind, lon_max_ind),
                     y=slice(lat_min_ind, lat_max_ind),
                     lat_2=slice(lat_min_ind, lat_max_ind),
                     )

def get_area_single_obs(obs_lon, obs_lat, data, hloc=80. ):

    reg_obs, _ = get_regions(obs_lon, obs_lat, obs_lon, obs_lat, hloc=hloc)

    data = slice_data_rot_pole(data, reg_obs[0], reg_obs[2], reg_obs[1], reg_obs[3])

    return data

def horizontal_localization():
    pass



def get_ens_in_hloc_area(path, time, var, obs_lon, obs_lat, members, hloc=80.):
    data_list = []
    height_path = path / f"data_const_ML.nc"
    height_data = xr.open_dataset(height_path).isel(height=slice(25,35), height_2=slice(25,35))
    for i, mem in enumerate(members):
        filename = f"data_ML_{time}.{mem}.nc"
        mem_path = path / filename
        data = xr.open_dataset(mem_path, autoclose=True).isel(height=slice(25,35), height_2=slice(25,35))

        if var == "w":
            data["w"] = interp_half_to_full_level(data.w.isel(time=0), height_data)
            data["w"] = data["w"].expand_dims({'time': data.pres.time})
        if var == "tt_lheat":
            data["tt_lheat"] = interp_half_to_full_level(data.tt_lheat.isel(time=0), height_data)
            data["tt_lheat"] = data["tt_lheat"].expand_dims({'time': data.pres.time})

        data = get_area_single_obs(obs_lon, obs_lat, data, hloc=hloc)

        data_list.append(data[[var, "lat_2", "lon_2"]])

    return xr.concat(data_list, dim="ens")


def plot_obs_regions(robs_lons, robs_lats, mask, des_llon, des_rlon, des_llat, des_ulat):

    rotated_pole = ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    fig, ax = pplt.subplots()#proj="rotpole",
                            #proj_kw={"pole_latitude": 40,
                            #         "pole_longitude": -170}
                            #)

    ax.scatter(
        robs_lons,
        robs_lats,
        color="green",
        marker="o",
        s=0.1,
        zorder=3,
        # transform=rotated_pole,
    )

    ax.scatter(
        robs_lons[mask],
        robs_lats[mask],
        color="blue",
        marker="o",
        s=0.1,
        zorder=3,
        # transform=rotated_pole,
    )
    ax.plot(
        [des_llon, des_rlon, des_rlon, des_llon, des_llon],
        [des_llat, des_llat, des_ulat, des_ulat, des_llat],
        color="black",
        zorder=4,
        # transform=rotated_pole,
    )
    ax.format(
        lonlim=(-4, 20.1),
        latlim=(42, 60),
        # lonlim=(5, 15),
        # latlim=(47.5, 52),
        labels=True,
        coast=True,
        reso="med",
    )
    return fig

@jit(nopython=True)
def interp_half_to_full_level_np(var_half, height_half, height_full):
    data_full = np.zeros_like(height_full)
    for i in np.arange(0, var_half.shape[2]):
        for j in np.arange(0, var_half.shape[1]):
            height_full_col = height_full[:, j, i][::1]
            height_half_col = height_half[:, j, i][::-1]
            data = var_half[:, j, i]
            # Find indices of missing values
            missing_indices = np.isnan(data)
            data_full[:, j, i] = np.interp(height_full_col, height_half_col, data)[::-1]
    return data_full



def interp_half_to_full_level(var_half, height):
    return xr.apply_ufunc(interp_half_to_full_level_np,
                        var_half,
                        height.z_ifc,
                        height.z_mc,
                        input_core_dims=[[ 'height_2', 'y', 'x'], ['height_2', 'y', 'x'], ['height', 'y', 'x']],
                        output_core_dims=[['height', 'y', 'x']],
                        )

def read_grib(path, vars, fc = False, grid_path="/jetfs/home/a12233665/pai-munich-vienna/assets/icon_grid_0047_R19B07_L.nc"):

    #Reads a grib file and assigns it standard lat/lon coordinates based on the grid file.
    data = read(path)#.rename(values="cell")
    data = data[[*vars, 'pres']]
    grid = xr.open_dataset(grid_path)
    if fc == True:
        data = data.assign_coords(clat=("cell2", grid.clat.values * 180 / np.pi))
        data = data.assign_coords(clon=("cell2", grid.clon.values * 180 / np.pi))
        data = data.swap_dims({"cell2": "cell"})
    else:
        data = data.assign_coords(clat=("cell", grid.clat.values * 180 / np.pi))
        data = data.assign_coords(clon=("cell", grid.clon.values * 180 / np.pi))

    return data

def read_grib_mf(path, ana_time, vars, inc = False, grid_path="/jetfs/home/a12233665/pai-munich-vienna/assets/icon_grid_0047_R19B07_L.nc"):
    #Reads an ensemble of grib files and assigns it standard lat/lon coordinates based on the grid file.
    path_reformed = Path(path)
    d = []
    ens = []
    if inc == False:
        for i, p in enumerate(path_reformed.parent.glob(f"an_R19B07.{ana_time}.0??")):
            ens.append(i)
            p = p.as_posix()
            xda = read(p)
            d.append(xda[[*vars, 'pres']])
    else:
        for i, p in enumerate(path_reformed.parent.glob(f"an_R19B07.{ana_time}_inc.0??")):
            ens.append(i)
            p = p.as_posix()
            xda = read(p)
            d.append(xda[[*vars, 'pres']])

    data = xr.concat(d, dim="ens")
    data = data.assign_coords(ens=("ens", ens))
    grid = xr.open_dataset(grid_path)

    data = data.assign_coords(clat=("cell", grid.clat.values * 180 / np.pi))

    data = data.assign_coords(clon=("cell", grid.clon.values * 180 / np.pi))

    return data


def produce_DWD_rectangle(lonl, lonr, latl, latu):
    #Codes a lat lon rectangle into a string that kendapy can read
    return "area=LATLON:{},{},{},{}".format(latl, lonl, latu, lonr)

def get_ekf(ekf_path, obsvar, llon = None, rlon = None, llat = None, ulat = None, whole_domain = True, active = True):
    #Reads an EKF file and filters out passive and rejected observations, observations of the wrong variable, and observations outside of a specified rectangle.
    if active:
        ekf = Ekf(ekf_path, filter=f"state=active")
    else:
        ekf = Ekf(ekf_path, filter=f"state=passive")
    ekf.add_filter(filter=f"varname={obsvar}")
    if whole_domain == False:
        location = produce_DWD_rectangle(llon, rlon, llat, ulat)
        print(location)
        ekf.add_filter(filter=f"{location}")
    return ekf

def get_obs_batch(ekfpath, llon, rlon, llat, ulat, obsvar="REFL", RAWBT_choice = 0,  new_loc = False, obs_pres = 30000, v_loc = 4, RTPP_correction = False, RTPP_factor = 0.75):
    #Reads an EKF file and returns a list of dictionaries, each containing the information of a single observation.
    ekf = get_ekf(ekfpath, obsvar, llon, rlon, llat, ulat, whole_domain = False)
    if new_loc == False:
        if len(ekf.obs()) != 0:
            v_loc_new = ekf.obs(param="v_loc")[0]
            obs_pres_new = ekf.obs(param="plevel")[0]
        else:
            v_loc_new = 1
            obs_pres_new = 30000
    else:
        v_loc_new = v_loc
        obs_pres_new = obs_pres
    if obsvar == 'RAWBT':
        skip = 2
    else:
        skip = 1
    if RTPP_correction:
        obs_batch = [dict(
            obsval=ekf.obs()[obsind],
            obsind=obsind,
            obslon=ekf.obs(param="lon")[obsind],
            obslat=ekf.obs(param="lat")[obsind],
            hloc=ekf.obs(param="h_loc")[obsind],
            vloc=ekf.obs(param="v_loc")[obsind],
            err=ekf.obs(param="e_o")[obsind],
            obspres=ekf.obs(param="plevel")[obsind],
            fgmeandep=0 - ekf.fgmeandep()[obsind],
            anaensT=(1/RTPP_factor)*(ekf.anaens()[:, obsind] - (1 - RTPP_factor)*ekf.fgens()[:, obsind]), #Correct for RTPP
        ) for obsind in range(RAWBT_choice, ekf.obs().shape[0], skip)]
        print('correction worked')
    elif new_loc == False:
        print('old loc')
        obs_batch = [dict(
            obsval=ekf.obs()[obsind],
            obsind=obsind,
            obslon=ekf.obs(param="lon")[obsind],
            obslat=ekf.obs(param="lat")[obsind],
            hloc=ekf.obs(param="h_loc")[obsind],
            vloc=ekf.obs(param="v_loc")[obsind],
            err=ekf.obs(param="e_o")[obsind],
            obspres=ekf.obs(param="plevel")[obsind],
            fgmeandep=0 - ekf.fgmeandep()[obsind],
            anaensT=ekf.anaens()[:, obsind],
        ) for obsind in range(RAWBT_choice, ekf.obs().shape[0], skip)]
    else:
        obs_batch = [dict(
            obsval=ekf.obs()[obsind],
            obsind=obsind,
            obslon=ekf.obs(param="lon")[obsind],
            obslat=ekf.obs(param="lat")[obsind],
            hloc=ekf.obs(param="h_loc")[obsind],
            vloc=v_loc_new,
            err=ekf.obs(param="e_o")[obsind],
            obspres=obs_pres_new,
            fgmeandep=0 - ekf.fgmeandep()[obsind],
            anaensT=ekf.anaens()[:, obsind],
        ) for obsind in range(RAWBT_choice, ekf.obs().shape[0], skip)]
    return obs_batch

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

def get_failure_list(ana_days, exp_path, start_path, save_name, save = False):
    failure_list = []
    for ana_day in ana_days:
        ana_time = ana_day + '120000'
        veri_time = ana_day + '110000'
        obs_path_TEMP = '{}/{}/fofTEMP_{}.nc'.format(exp_path, ana_time, veri_time)
        sondes_T = get_ekf(obs_path_TEMP, "T")
        for i, rep in enumerate(sondes_T.reports()):
            if os.path.isfile(start_path + str(rep) + '.npy') == False:
                failure_list.append((ana_time, i))
    if save:
        np.save(save_name, np.array(failure_list))
    return failure_list