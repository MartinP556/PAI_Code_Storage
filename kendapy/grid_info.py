#!/usr/bin/env python3
#
#  print most important parameters for icon netcdf grid file

import sys
import netCDF4
import numpy as np
import re

# regular expression for standard grid file names
r = re.compile('.*icon_grid_(?P<num>\d\d\d\d)_R(?P<R>\d\d)B(?P<B>\d\d)_.*nc')

for f in sys.argv[1:] :

    # parse file name
    m = r.match(f)
    if m :
        R, B = int(m.group('R')), int(m.group('B'))
        res = 5050/(R*(2**B)) # km
        num = int(m.group('num'))
        desc = '(grid #{}, resolution {:.2f}km)'.format(num,res)
    else :
        desc = ''

    # investigate content
    ds = netCDF4.Dataset(f)
    if 'vlon' in ds.variables :
        vlon = ds.variables['vlon'][...]
        vlat = ds.variables['vlat'][...]
        print( '{:32s} {:32s} has {:10d} cells and {:10d} vertices in {:7.2f} <= lon [deg] <= {:7.2f}, {:7.2f} <= lat [deg] <= {:7.2f}'.format(
                f, desc, ds.variables['clon'].size, ds.variables['vlon'].size, vlon.min()*180/np.pi, vlon.max()*180/np.pi, vlat.min()*180/np.pi, vlat.max()*180/np.pi) )
    ds.close()
