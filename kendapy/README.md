
K E N D A P Y
-------------

A Python module for reading, analyzing and plotting KENDA experiments
(requires numpy, scipy, matplotlib, grib_api, NetCDF4)

**Files**

```
experiment.py........ collects information about an experiment, allows access to
                      all ekf/lff/laf files, generates cumulative statistics

plot_experiment.py... generates plots for an experiment

obs_overview.py...... observation overview plots

cosmo_state.py....... representation of the COSMO state contained in a
                      grib1 or grib2 output file

ekf.py............... reads and filters ekf files

plot_ekf.py.......... generates plots for a single ekf file

time14.py............ 14-digit time representation class

binplot.py........... used for histogram plots

visop_*py............ various script for visible satellite images
```
