#!/usr/bin/env python3
import numpy as np
#import os
import sys
#os.system(f'source env/bin/activate')
#sys.path.insert(0, '/jetfs/home/a12233665/pai-munich-vienna/pai/env/lib/python3.8/site-packages')
from numba import jit, config
config.THREADING_LAYER = 'omp' #'threadsafe'
#os.system('deactivate')

EARTH_RAD = 6371.0  # in km according to WGS84


def get_dist_from_obs(lats, lons, olat, olon, h_loc):
    """Get lower left corner and upper right corner lats and lons from the area
    which is influenced by the observation, confined by the cutoff distance.
    (Add +.1 degrees to each side)

    Args:
        model_state (ModelState): see model_state.py
        obs (Observation): see observation.py
        h_loc (float): localization length scale in km

    Returns:
       (llcrnrlat, llcrnrlon, urcrnrlat, urcrnrlon) (4 x float): lat lons of corners
    """

    dist = haversine_distance(lats, lons, olat, olon)
    utoff = 2.0 * h_loc * np.sqrt(10.0 / 3.0)

    return dist


@jit(fastmath=True, parallel=True, nopython=True)
def haversine_distance(lat, lon, lat_0, lon_0):
    dlat_rad = np.radians(lat - lat_0)
    dlon_rad = np.radians(lon - lon_0)
    s2_dlat = np.sin(0.5 * dlat_rad) ** 2
    s2_dlon = np.sin(0.5 * dlon_rad) ** 2
    ccs2 = np.cos(np.radians(lat)) * np.cos(np.radians(lat_0)) * s2_dlon
    return 2.0 * EARTH_RAD * np.arcsin(np.sqrt(s2_dlat + ccs2))


@jit(nopython=True, fastmath=True, parallel=True)
def gaspari_cohn(dist, l):
    """Computes the Gaspari Cohn function

    for a constant localization length scale

    Args:
        dist (numpy.ndarray): distance field
        l (float): localization length scale
    Returns:
        numpy.ndarray: Gaspari Cohn Factors, same dimensions as dist
    """
    a = l * np.sqrt(10.0 / 3.0)
    da = np.abs(dist.flatten()) / a
    gp = np.zeros_like(da)

    i = np.where(dist.flatten() <= a)

    gp[i] = np.maximum(
        0.0,
        -0.25 * da[i] ** 5
        + 0.5 * da[i] ** 4
        + 0.625 * da[i] ** 3
        - 5.0 / 3.0 * da[i] ** 2
        + 1.0,
    )

    i = np.where((dist.flatten() > a) * (dist.flatten() < 2 * a))
    gp[i] = np.maximum(
        0.0,
        1.0 / 12.0 * da[i] ** 5
        - 0.5 * da[i] ** 4
        + 0.625 * da[i] ** 3
        + 5.0 / 3.0 * da[i] ** 2
        - 5.0 * da[i]
        + 4.0
        - 2.0 / 3.0 / da[i],
    )
    return gp.reshape(dist.shape)


@jit(nopython=True, fastmath=True, parallel=True)
def vertical_dist(pres, obs_pres):
    distances = np.abs(np.log(pres) - np.log(obs_pres))
    return distances


def additional_damping(pres_model):
    pres_model = pres_model / 100.0
    p_one = 300.0
    p_zero = 100.0
    damping = np.ones_like(pres_model)
    i = np.where(pres_model < 100.0)
    damping[i] = 0.0

    i = np.where((pres_model >= 100.0) & (pres_model <= 300.0))
    dist = np.abs(np.log(pres_model[i]) - np.log(p_zero)) / np.abs(
        np.log(p_zero) - np.log(p_one)
    )
    damping[i] = 0.5 * (1 - np.cos(np.pi * dist))
    return damping[..., np.newaxis]

def find_analysis_in_area(
    left_lon,
    right_lon,
    lower_lat,
    upper_lat,
    ana_lons,
    ana_lats,
    ana_array,
):
    # get all reports within the region
    ana = ana_array.where((ana_lons >= left_lon) & (ana_lons <= right_lon) & (ana_lats >= lower_lat) & (ana_lats <= upper_lat), drop=True)
    return ana
