## Description and Overview
netCDF4-python is an object-oriented python interface to the NetCDF C library.  

## Loading netCDF4-python on Cori
```
module load python/2.7-anaconda
```

## Using netCDF4-python in applications

```
from netCDF4 import Dataset
fx = Dataset("mydir/test.nc", "w", format = "NETCDF4")
```
Note that netCDF4-python supports various classic netcdf versions also, e.g.,
netcdf3, netcdf3-classic, please make sure the format is consistent when you
read and write the data. 
