"""Public NOAA MRMS archive discovery and phase separation."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
import numpy as np
import requests
import xarray as xr
from .download import download_urls
from .mrms import read_grid
from .pps import read_all, valid_geo

AWS = 'https://noaa-mrms-pds.s3.amazonaws.com'

def list_product(product, day):
    params = {'list-type':'2','prefix':f'CONUS/{product}/{day:%Y%m%d}/'}
    result = {}
    while True:
        r = requests.get(AWS, params=params, timeout=90)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for node in root.findall('.//{*}Key'):
            key = node.text
            if key and key.endswith('.grib2.gz'):
                stamp = Path(key).name.split('_')[-1].split('.')[0]
                t = datetime.strptime(stamp,'%Y%m%d-%H%M%S').replace(tzinfo=timezone.utc)
                result[t] = key
        token = root.findtext('.//{*}NextContinuationToken')
        if not token: return result
        params['continuation-token'] = token

def phase_rates(rate, flag):
    rain = np.isin(flag,[1,6,10,91,96]); snow = flag == 3
    valid = np.isfinite(rate) & (rate >= 0) & (rain | snow | (flag == 0))
    return tuple(np.where(valid,np.where(mask,rate,0),np.nan).astype('float32') for mask in (rain,snow))

def prepare_pair(rate_key, flag_key, directory):
    paths = download_urls([AWS+'/'+quote(k) for k in (rate_key,flag_key)],directory)
    lat,lon,rate,t,_ = read_grid(paths[0]); y,x,flag,s,_ = read_grid(paths[1])
    if t != s or not np.array_equal(lat,y) or not np.array_equal(lon,x):
        raise ValueError('MRMS rate and flag grid/time mismatch')
    stamp = datetime.fromtimestamp(t,timezone.utc)
    out = []
    for name,values in zip(('rain','snow'),phase_rates(rate,flag)):
        p = Path(directory)/f'MRMS_{name}_{stamp:%Y%m%d_%H%M%S}.nc'
        if not p.exists():
            ds = xr.Dataset({name+'_rate':(('latitude','longitude'),values,{'units':'mm/hr'})},coords={'latitude':lat,'longitude':lon,'time':np.datetime64(int(round(t*1000)),'ms')})
            tmp = p.with_suffix('.part.nc')
            ds.to_netcdf(tmp,engine='netcdf4',encoding={name+'_rate':{'zlib':True}})
            tmp.replace(p)
        out.append(p)
    return out

def overpass_times(file,start,end):
    chunks=[]
    for a in read_all(file).values():
        lon=(a['lon']+180)%360-180
        ok=valid_geo(a)&(a['lat']>=24)&(a['lat']<=50)&(lon>=-125)&(lon<=-66)&np.any(np.isfinite(a['data']),axis=-1)
        t=a['time'][ok]; t=t[(t>=start.timestamp())&(t<end.timestamp())]
        if t.size: chunks.append(t[::max(1,t.size//10000)])
    return np.concatenate(chunks) if chunks else np.array([])
