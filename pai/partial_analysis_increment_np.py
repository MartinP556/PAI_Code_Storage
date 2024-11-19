import sys
sys.path.append('/jetfs/home/a12233665/pai-munich-vienna/pai')
#sys.path.insert(0, '/jetfs/home/a12233665/pai-munich-vienna/pai/env/lib/python3.8/site-packages')
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
#os.system(f'source env/bin/activate')
from numba import njit, config
config.THREADING_LAYER = 'omp' #'threadsafe'
#os.system('deactivate')
import gc
from joblib import Parallel, delayed
from kendapy.ekf import Ekf

import plot_oi_utils as utils
import localization as loc
import observation as oi
import pai_utils as paiut

# everything is done for only one obs
@njit
def K_column(Xa, YaTq, Rinvq):
    res = np.zeros(Xa.shape[1:])
    for m1 in range(Xa.shape[1]):  # spatial index 1
        for m2 in range(Xa.shape[2]):  # spatial index 2
            #for m3 in range(Xa.shape[3]):  # spatial index 3
            for k in range(Xa.shape[0]):  # member index
                res[m1, m2] += Xa[k, m1, m2] * YaTq[k] #MARTIN EDIT include m3 if necessary
            res[m1, m2] *= Rinvq
    return res


@njit
def compute_kalman_gain(anaensT, obserr, ens_perturb, nens, hgc=1.0, vgc=1.0):
    kcol = 1.0 / (nens - 1) * K_column(ens_perturb, anaensT, obserr ** (-2))
    return kcol * hgc * vgc


@njit
def compute_increment(kcol, fgmeandep):
    incr = kcol * fgmeandep
    return incr


@njit
def compute_hloc(obslat, obslon, obshloc, meanlat, meanlon):
    hdist = loc.haversine_distance(meanlat, meanlon, obslat, obslon)
    hgc = loc.gaspari_cohn(hdist, obshloc)
    return hgc


@njit
def compute_vloc(obspres, obsvloc, meanpres):
    vdist = np.abs(np.log(meanpres) - np.log(obspres))
    vgc = loc.gaspari_cohn(vdist, obsvloc)
    return vgc


@njit
def pai_loop_over_obs(
    anaens_list,
    obserr_list,
    fgmeandep_list,
    obslats,
    obslons,
    hlocs,
    vlocs,
    press,
    ens_perturb,
    mlats,
    mlons,
    mean_pres,
    nens,
):
    incr = np.zeros((ens_perturb.shape[1], ens_perturb.shape[2], ens_perturb.shape[3]))
    for i, anaens in enumerate(anaens_list):
        if i % 100 == 0:
            print(i)
        hgc = compute_hloc(obslats[i], obslons[i], hlocs[i], mlats, mlons)
        # vgc = compute_vloc(press[i] / 100., vlocs[i], mean_pres / 100.)
        kcol = compute_kalman_gain(
            anaens_list[i],
            obserr_list[i],
            ens_perturb,
            nens,
            hgc=hgc,
            # vgc=vgc
        )
        incr += compute_increment(kcol, fgmeandep_list[i])
    return incr


def pai_loop_over_obs_slow(obs_list, enslon, enslat, ens_perturb, mean_pres, nens):
    incr = np.zeros((ens_perturb.shape[1], ens_perturb.shape[2], ens_perturb.shape[3])) 
    for i, obs in enumerate(obs_list):
        hgc = compute_hloc(obs["obslat"], obs["obslon"], obs["hloc"], enslat, enslon)

        if obs["vloc"] < 10:
            vgc = compute_vloc(obs["obspres"] / 100.0, obs["vloc"], mean_pres / 100.0)
        else:
            vgc = 1
        kcol = compute_kalman_gain(
            obs["anaensT"],
            obs["err"],
            ens_perturb,
            nens,
            hgc=hgc,
            vgc=vgc
        )
        incr += compute_increment(kcol, obs["fgmeandep"])
    return incr


def pai_one_obs(obs, enslon, enslat, ens_perturb, mean_pres, nens):
    # incr = np.zeros((ens_perturb.shape[1], ens_perturb.shape[2], ens_perturb.shape[3]))
    hgc = compute_hloc(obs["obslat"], obs["obslon"], obs["hloc"], enslat, enslon)
    if obs["vloc"] < 10.0:
        vgc = compute_vloc(obs["obspres"] / 100.0, obs["vloc"], mean_pres / 100.0)
    else:
        vgc = 1.0
    kcol = compute_kalman_gain(
        obs["anaensT"], obs["err"], ens_perturb, nens, hgc=hgc, vgc=vgc
    )
    incr = compute_increment(kcol, obs["fgmeandep"])
    return incr


def get_ekf(ekf_path, obsvar):
    ekf = Ekf(ekf_path, filter=f"state=active")
    ekf.add_filter(filter=f"varname={obsvar}")
    return ekf


def get_one_obs(ekfpath, obsrep, obsvar="REFL", obsind=0):
    ekf = get_ekf(ekfpath, obsvar)
    ekf.add_filter(filter=f"report={obsrep}")

    obs = dict(
        obsval=ekf.obs()[obsind],
        obsrep=obsrep,
        obsind=obsind,
        obslon=ekf.obs(param="lon")[obsind],
        obslat=ekf.obs(param="lat")[obsind],
        hloc=ekf.obs(param="h_loc")[obsind],
        vloc=ekf.obs(param="v_loc")[obsind],
        err=ekf.obs(param="e_o")[obsind],
        obspres=ekf.obs(param="plevel")[obsind],
        fgmeandep=0 - ekf.fgmeandep()[obsind],
        anaensT=ekf.anaens()[:, obsind],
    )
    return obs


def get_ens(path, time, var, members):
    data_list = []
    height_path = path / f"data_const_ML.nc"
    height_data = xr.open_dataset(height_path)
    for i, mem in enumerate(members):
        filename = f"data_ML_{time}.{mem}.nc"
        mem_path = path / filename
        data = xr.open_dataset(mem_path, autoclose=True)
        data_list.append(data[[var, "lat_2", "lon_2"]])

    return xr.concat(data_list, dim="ens")


def get_ens_mean(path, time, var):
    filename = f"data_ML_{time}.mean.nc"
    mean_path = path / filename
    mean_data = xr.open_dataset(mean_path, autoclose=True).load()
    return mean_data[[var, "pres", "lon_2", "lat_2"]]


if __name__ == "__main__":

    import argparse

    parser = utils.define_parser()
    args = parser.parse_args()

    d = args.startdate
    startdate = datetime.strptime(f"{d}", "%Y%m%d")
    nice_date = startdate.strftime("%d.%m.%Y")

    exps = args.exps.split(",")

    times = [f"13{x:02d}00" for x in range(0, 31)]
    members = [f"{x:03d}" for x in range(1, 11)]

    obstype = "RAD"  # TODO: dynamic obstype?

    obspath = (
        Path(args.basedir)
        / (d + exps[0])
        / "feedback"
        / (d + "120000")
        / f"ekf{obstype}.nc.{d}{times[0]}"
    )

    ekf = get_ekf(str(obspath), "REFL")
    rep_list_all = ekf.reports()[0:]

    print("DEBUG: slice data")
    obs_lons = ekf.obs(param="lon")
    obs_lats = ekf.obs(param="lat")
    robs_lons, robs_lats = oi.location_to_rotated_pole(obs_lons, obs_lats)

    # desired area
    llon = 5
    rlon = 15
    llat = 47
    ulat = 52

    rot_lons, rot_lats = oi.location_to_rotated_pole([llon, rlon], [llat, ulat])

    infl_reg, obs_reg = paiut.get_regions(*rot_lons, *rot_lats, hloc=35.0)

    rep_list, _ = oi.find_reps_in_area(*infl_reg, robs_lons, robs_lats, rep_list_all)

    print(len(rep_list))
    path = Path(args.basedir) / (d + exps[0]) / "int_latlon"
    var = args.var
    time = args.time  # times[0]

    print("read data")
    mean = get_ens_mean(path, time, var)  # .sel(height=slice(25,35))
    ens = get_ens(path, time, var, members)  # .sel(height=slice(25,35))

    if var in ["w", "tt_lheat"]:
        # height_path = path / f"data_const_ML.nc"
        # height_data = xr.open_dataset(height_path)
        # mean = paiut.interp_half_to_full_level(var, height_data) #TODO
        print("half levels!")
        mean = mean.sel(height_2=slice(25, 35))
        ens = ens.sel(height_2=slice(25, 35))

    ensperturb = ens - mean

    # mean = paiut.slice_data_rot_pole(mean, *rot_lons, *rot_lats)
    # ens = paiut.slice_data_rot_pole(ens, *rot_lons, *rot_lats)

    # mode = "slow"
    mode = "parallel"

    if mode == "slow":
        print("compute increment slow")
        incr = pai_loop_over_obs_slow(
            obs_list,
            ensperturb.lon.values,
            ensperturb.lat.values,
            ensperturb[var].isel(time=0).values,
            mean.pres.isel(time=0).values,
            len(members),
        )

    if mode == "parallel":
        # create batches
        batch_size = 100
        nbatches = -(
            -len(rep_list) // batch_size
        )  # Round up the division to get the number of batches

        batches = [
            rep_list[i * batch_size : (i + 1) * batch_size] for i in range(nbatches)
        ]
        # incr_list = [0 for i in range(0, nbatches)]

        incr = np.zeros((mean[var].shape[1], mean[var].shape[2], mean[var].shape[3]))
        for i, rep_batch in enumerate(batches):
            print(f"compute batch {i}")
            obs_batch = Parallel(n_jobs=10, verbose=0)(
                delayed(get_one_obs)(str(obspath), rep, obsvar="REFL", obsind=0)
                for rep in rep_batch
            )
            print(f"compute pai {i}")
            incr_list = Parallel(n_jobs=10, verbose=4)(
                delayed(pai_one_obs)(
                    obs,
                    ensperturb.lon.values,
                    ensperturb.lat.values,
                    ensperturb[var].isel(time=0).values,
                    mean.pres.isel(time=0).values,
                    len(members),
                )
                for obs in obs_batch
            )
            incr += sum(incr_list)
            del incr_list
            del obs_batch
            gc.collect()

    print(incr.shape)
    ensperturb.close()
    ens.close()

    print("create xarray")
    incr = incr[np.newaxis, :]  # add the time coordinate
    incr_xr = xr.DataArray(
        incr,
        coords=[mean.time, mean.height_2, mean.lat_2, mean.lon_2],  # TODO
        dims=["time", "height_2", "y", "x"],
        name=f"pai{var}",
    )

    incr_xr = incr_xr.assign_coords(lat=mean.lat, lon=mean.lon)
    # print(incr_xr)
    incr_xr.to_netcdf(path / f"data_ML_{time}.mean.pai_{var}.nc")
    print("stored the file")

    incr_xr.close()
    ## TODO: leverage horizontal loc! and vertical localization, how deal with height_2
