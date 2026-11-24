"""Read NOAA MRMS GRIB2 or standardized NetCDF reference grids."""
import gzip, shutil, tempfile
from pathlib import Path
import numpy as np
import xarray as xr

def read_grid(path, variable=None):
    path=Path(path)
    if path.suffix=='.gz':
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/path.stem
            with gzip.open(path,'rb') as a,p.open('wb') as b: shutil.copyfileobj(a,b)
            return read_grid(p,variable)
    kwargs={} if path.suffix.lower() in ('.nc','.nc4') else dict(engine='cfgrib',backend_kwargs={'indexpath':''})
    with xr.open_dataset(path,**kwargs) as ds:
        if variable is None:
            candidates=[k for k,v in ds.data_vars.items() if v.ndim>=2]
            if len(candidates)!=1: raise ValueError(f'Select variable from {candidates}')
            variable=candidates[0]
        a=ds[variable].squeeze(drop=True)
        lat=ds['latitude' if 'latitude' in ds else 'lat'].values
        lon=ds['longitude' if 'longitude' in ds else 'lon'].values
        if lat.ndim!=1 or lon.ndim!=1: raise ValueError('Expected rectilinear MRMS grid')
        a=a.transpose(ds['latitude' if 'latitude' in ds else 'lat'].dims[0],ds['longitude' if 'longitude' in ds else 'lon'].dims[0])
        t=ds['valid_time' if 'valid_time' in ds else 'time'].values
        t=np.asarray(t).astype('datetime64[ns]').reshape(-1)
        if len(t)!=1 or np.isnat(t[0]): raise ValueError('Expected one valid UTC time')
        v=np.asarray(a.values,dtype=np.float32); v[v<0]=np.nan
        return lat,(lon+180)%360-180,v,float(t[0].astype('int64')/1e9),str(a.attrs.get('units',''))
