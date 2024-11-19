#!/usr/bin/env python3
import numpy as np
import xarray as xr

from kendapy.ekf import Ekf

import cartopy.crs as ccrs


class Observation:
    def __init__(self, path, var=None, report=None, plot=False):
        self.path = path
        self.var = var
        self.ekf = Ekf(str(path), filter=f"state=active")

        self.ekf.add_filter(filter=f"varname={var}")
        self.ekf.add_filter(filter=f"report={report}")

        self.ekftoxarray()

        self.lons = self.ekf.obs(param="lon")
        self.lats = self.ekf.obs(param="lat")
        self.rot_lons, self.rot_lats = location_to_rotated_pole(self.lons, self.lats)

    def ekftoxarray(self):

        rep = self.ekf.reports()[0]
        lat = self.ekf.obs(param="lat")[0]
        lon = self.ekf.obs(param="lon")[0]

        pres = self.ekf.obs(param="plevel")[np.newaxis, np.newaxis, :, np.newaxis]
        obs = self.ekf.obs().filled()[np.newaxis, np.newaxis, :, np.newaxis]
        print(lon, lat)
        datavars = {}
        datavars[self.var] = (["x", "y", "z", "rep"], obs)
        datavars["pres"] = (["x", "y", "z", "rep"], pres)

        self.data = xr.Dataset(
            data_vars=datavars,
            coords=dict(
                lon=(["x"], [lon]),
                lat=(["y"], [lat]),
                z=(["z"], np.arange(0, pres.shape[2])),
                report=(["rep"], [rep]),
            ),
        )


def location_to_rotated_pole(lons, lats):
    rotated_pole = ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    rotated_coords = [
        rotated_pole.transform_point(lon, lat, ccrs.PlateCarree())
        for (lon, lat) in zip(lons, lats)
    ]

    rot_lons, rot_lats = zip(*rotated_coords)

    return np.array(rot_lons), np.array(rot_lats)

def rotated_pole_to_location(rlons, rlats):
    #
    rotated_pole = ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    rotated_coords = [
        ccrs.PlateCarree().transform_point(rlon, rlat, rotated_pole)
        for (rlon, rlat) in zip(rlons, rlats)
    ]

    lons, lats = zip(*rotated_coords)

    return np.array(lons), np.array(lats)


def find_reps_in_area(
    left_lon,
    right_lon,
    lower_lat,
    upper_lat,
    ekf_lons,
    ekf_lats,
    ekf_reports,
):
    # get all reports within the region
    mask = (
        (ekf_lons >= left_lon)
        & (ekf_lons <= right_lon)
        & (ekf_lats >= lower_lat)
        & (ekf_lats <= upper_lat)
    )
    if len(mask) == 0:
        print("No reports in this area")
        return False
    else:
        lons = np.unique(ekf_lons)
        lats = np.unique(ekf_lats)
        rep_mask = (
            (lons >= left_lon)
            & (lons <= right_lon)
            & (lats >= lower_lat)
            & (lats <= upper_lat)
        )

    return np.array(ekf_reports)[mask], mask
