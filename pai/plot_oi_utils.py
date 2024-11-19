#!/usr/bin/env python3

import os
import argparse
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from numba import jit

#############
# Constants #
#############

EARTH_RAD = 6378.0  # in km


def define_parser():
    parser = argparse.ArgumentParser(
        description="Plotting Script for ICON KENDA experiments"
    )

    parser.add_argument(
        "-b", "--basedir", dest="basedir", default="", help="path to files"
    )

    parser.add_argument(
        "-s",
        "--startdate",
        dest="startdate",
        default="",
        help="YYYYMMDD of the experiment",
    )
    parser.add_argument(
        "-pd",
        "--prevdate",
        dest="prevdate",
        default="",
        help="YYYYMMDD of the day before the experiment to extract 23Uhr radiosondes.",
    )
    parser.add_argument(
        "-time",
        "--time",
        dest="time",
        default="",
        help="HHMMSS of the experiment",
    )
    parser.add_argument(
        "-prevtime",
        "--prevtime",
        dest="prevtime",
        default="",
        help="HHMMSS of the previous analysis",
    )
    parser.add_argument(
        "-sondetime",
        "--sondetime",
        dest="sondetime",
        default="",
        help="HHMMSS of the verifying radiosondes",
    )
    parser.add_argument(
        "-t",
        "--timeseries",
        dest="timeseries",
        help="get timeseries plots",
        action="store_true",
    )
    parser.add_argument(
        "-M",
        "--midnight",
        dest="midnight",
        default="1",
        help="Whether or not the analysis is at midnight.",
    )
    parser.add_argument(
        "-ES",
        "--ELFstartdate",
        dest="ELFstartdate",
        default="1",
        help="Start date for calculating ELF",
    )
    parser.add_argument(
        "-EE",
        "--ELFenddate",
        dest="ELFenddate",
        default="11",
        help="End date for calculating ELF",
    )
    parser.add_argument(
        "-EN",
        "--ELFnamecode",
        dest="ELFnamecode",
        default="01_11",
        help="Name code for saving ELF",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        default="../figures/tmp.png",
        help="Name of outfile",
    )

    parser.add_argument(
        "-e",
        "--experiments",
        dest="exps",
        default=None,
        help="comma-seperated list of experiments",
    )
    parser.add_argument(
        "-m",
        "--movie",
        dest="movie",
        help="make vertical velocity movied",
        action="store_true",
    )

    parser.add_argument(
        "-v",
        "--variable",
        dest="var",
        help="store a snapshots of var",
    )
    parser.add_argument(
        "-cmap",
        "--cmap",
        dest="cmap",
        help="cmap",
    )
    parser.add_argument(
        "-clevs",
        "--clevs",
        dest="clevs",
        help="clevs for var snapshot, Format: from,to,no.steps (used in np.linspace).",
    )

    parser.add_argument(
        "-wtg",
        "--wtg",
        dest="wtg",
        help="compute the weak temperature diagnostic",
        action="store_true",
    )

    parser.add_argument(
        "-l",
        "--lev",
        dest="lev",
        help="model level to plot",
    )
    parser.add_argument(
        "-d",
        "--diff",
        dest="diff",
        help="plot the differences in w",
        action="store_true",
    )
    parser.add_argument(
        "-vmd",
        "--vmd",
        dest="vmd",
        help="plot vertical velocity masking = vertical motion diagnostic",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--region",
        dest="region",
        help="evaluate only for a region",
        action="store_true",
    )
    parser.add_argument(
        "-obstype",
        "--obstype",
        dest="obstype",
        help="obstype for ekf file",
    )
    return parser


def get_timeseries(dir, startdate, template="data_ML_%H%M%S.001.nc"):
    ts = sorted(os.listdir(dir))
    timesteps = []
    for t in (t for t in ts if "13" in t and ".001.nc" in t and "tmp" not in t):
        date = datetime.strptime(t, template)
        date = date.replace(
            year=startdate.year, month=startdate.month, day=startdate.day
        )
        timesteps.append(date)
    return [t for t in ts if "13" in t and ".001.nc" in t], timesteps


@jit(nopython=True, fastmath=True, parallel=True)
def haversine_distance(lat, lon, lat_0, lon_0):
    dlat_rad = np.radians(lat - lat_0)
    dlon_rad = np.radians(lon - lon_0)
    s2_dlat = np.sin(0.5 * dlat_rad) ** 2
    s2_dlon = np.sin(0.5 * dlon_rad) ** 2
    ccs2 = np.cos(np.radians(lat)) * np.cos(np.radians(lat_0)) * s2_dlon
    return 2.0 * EARTH_RAD * np.arcsin(np.sqrt(s2_dlat + ccs2))


def compute_potential_temperature(data):
    CP = 1005  # specific heat capacity [J / (kg * K )] at constant pressure
    R = 287.058  # gas constant dry air
    return data.temp * (1000.0 / data.pres) ** (R / CP)
