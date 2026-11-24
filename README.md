# pmwMRMS
Collocate NASA PPS Level-1C PMW swaths onto native MRMS rectilinear grids (~1 km). No interpolation of MRMS. Nearest valid TB footprint per channel within a spatial radius and ± time tolerance. Scan time is broadcast when beam time is unavailable. This is footprint-center matching, not footprint-area averaging.

## Install
`pip install -e '.[maps,test]'`

## Fully automatic example
```python
from pmwMRMS import automatic
results = automatic(sensor='ATMS', satellite='NPP',
    start_date='2023-01-01', end_date='2023-01-01',
    output_dir='./data', max_cases=1, plot=True, show=True)
```
Prompts for PPS registered email unless supplied via PPS_USERNAME. Public NOAA AWS MRMS needs no credentials. Downloads are cached. Selects one near-median CONUS observation time per orbit, not every MRMS timestamp. `max_cases=None` processes all eligible orbits. Phase rates use NOAA PrecipRate and PrecipFlag; hail and missing coverage are excluded. ATMS nine-channel layout follows NASA L1C documentation. Requires cfgrib and ecCodes for MRMS. Archive availability and live authentication are not guaranteed.

## Automatic PPS discovery with supplied MRMS files
```python
from pmwMRMS import automatic
results = automatic(sensor='GMI', satellite='GPM', version='V07A',
    start_date='2023-01-01', end_date='2023-01-01',
    mrms_pairs=[('rain.nc','snow.nc')], output_dir='./data',
    delta_t=7.5, max_distance_km=15, plot=True, show=True)
```
Supply actual MRMS rain and snow rate paths or direct HTTPS URLs in `mrms_pairs`. NOAA archives differ by date and product; the automatic mode above discovers the public NOAA archive. If using total precipitation rate and precipitation type, derive phase rates using the documented product codes first. Do not substitute snowfall accumulation or snow depth for snowfall rate. Units are preserved, not converted; use liquid-equivalent mm/h for both reference rates. Input files must share identical grids and valid UTC timestamps. GRIB2(.gz) and NetCDF supported. NetCDF should have latitude/longitude (or lat/lon) 1D coordinates and a scalar time or valid_time. PPS credentials via PPS_USERNAME/PPS_PASSWORD; default password is registered email as in original package. Downloads are cached.

## Local files
```python
from pmwMRMS import process
process('rain.nc','snow.nc',['PMW.HDF5'],'matched.npz',sensor='GMI')
```
GMI, SSMIS and ATMS channel layouts are provided; verify against your product version. Other sensors require explicit `frequencies={'S1':[...], 'S2':[...]}` with every actual group/channel in file order. Supports PPS HDF5-style group layout (including compatible HDF5-backed NetCDF files), not arbitrary NetCDF PMW schemas. Plot picks the channel with greatest coverage in each frequency range. Output contains full grid TB, distances, signed time offsets, reference fields, channel metadata, masks and source paths. Latitude/longitude are 1D; use meshgrid for 2D coordinates. Full CONUS arrays may require several GB RAM; process one reference time at a time. PNG is a 2x2 CONUS plot. Cartopy optional; without it axes show coordinates only.

## Tests
`pytest -q` after installation. Live archive authentication and product availability require independent verification.
