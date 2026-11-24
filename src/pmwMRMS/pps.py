"""Automatic PPS archive discovery and multi-swath DPR collocation."""
import os,re,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from urllib.parse import urljoin
from html.parser import HTMLParser
import numpy as np
import h5py,requests
from scipy.spatial import cKDTree
from .matching import _xyz,EARTH_KM
from .readers import _get
from .download import download_urls

BASE='https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata/'
PAT=re.compile(r'\.(\d{8})-S(\d{6})-E(\d{6})\.')
def interval(name):
    m=PAT.search(str(name))
    if not m: raise ValueError(f'Unrecognized granule name: {name}')
    a,b=[datetime.strptime(m[1]+v,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc) for v in (m[2],m[3])]
    return a,b+timedelta(days=b<a)

def orbit(name):
    return int(re.search(r'\.(\d+)\.V\d+[A-Z]?\.',str(name),re.I)[1])

def discover(session,start,end,folder,predicate,base=BASE):
    found=set(); day=start.date()
    while day<=end.date():
        url=urljoin(base,day.strftime('%Y/%m/%d/')+folder+'/')
        from urllib.parse import urlsplit
        u=urlsplit(url); listing=f'{u.scheme}://{u.netloc}/text{u.path}'
        r=session.get(listing,timeout=90)
        print(f'[PPS] {listing}: HTTP {r.status_code}',flush=True)
        if r.status_code==404: day+=timedelta(days=1); continue
        r.raise_for_status()
        if '<html' in r.text.lower(): raise RuntimeError('Expected plain-text PPS listing, received HTML')
        names=re.findall(r'[\w.+-]+\.(?:HDF5|H5|NC4|NC)(?![\w.])',r.text,re.I)
        for name in names:
            if predicate(name) and PAT.search(name):
                a,b=interval(name)
                if a<=end and b>=start: found.add(urljoin(url,name))
        print(f'[PPS] {len(names)} listed files',flush=True)
        day+=timedelta(days=1)
    unique={}
    for x in sorted(found):
        key=orbit(x); version=re.search(r'\.V(\d+)([A-Z]?)\.',x,re.I)
        rank=(int(version[1]),version[2],x)
        if key not in unique or rank>unique[key][0]: unique[key]=(rank,x)
    return sorted(v[1] for v in unique.values())

def read_group(h,g,value):
    lat,lon=(_get(h,f'{g}/{k}') for k in ('Latitude','Longitude'))
    data=_get(h,f'{g}/{value}'); shape=lat.shape
    if lat.ndim!=2 or lon.shape!=shape or data.shape[:2]!=shape: raise ValueError(f'Invalid geometry in {g}')
    s=h[f'{g}/ScanTime']; keys=['Year','Month','DayOfMonth','Hour','Minute','Second']
    arrays=np.broadcast_arrays(*[s[k][...] for k in keys]); ts=np.full(arrays[0].shape,np.nan)
    for i in np.ndindex(ts.shape):
        try: ts[i]=datetime(*[int(a[i]) for a in arrays],tzinfo=timezone.utc).timestamp()
        except (ValueError,OverflowError): pass
    if 'MilliSecond' in s: ts+=_get(s,'MilliSecond')/1000.
    mode='beam' if ts.shape==shape else 'scan_broadcast'
    if ts.shape==(shape[0],): ts=np.broadcast_to(ts[:,None],shape).copy()
    elif ts.shape==(shape[0],1): ts=np.broadcast_to(ts,shape).copy()
    if ts.shape!=shape: raise ValueError(f'Unsupported time shape {ts.shape} in {g}')
    # Optional explicitly documented beam offsets in seconds; never infer offsets.
    if 'beamTimeOffset' in s:
        ds=s['beamTimeOffset']; units=ds.attrs.get('units',''); units=units.decode() if isinstance(units,bytes) else str(units)
        if units.lower() in ('s','seconds','second') and mode=='scan_broadcast':
            ts=ts+np.broadcast_to(ds[...],shape); mode='beam_offset'
    data[(data<0)|(~np.isfinite(data))]=np.nan
    if value in ('Tc','Tb'): data[(data>400)|(data<20)]=np.nan
    return dict(lat=lat,lon=lon,time=ts,data=data,time_mode=mode)

def read_all(path):
    out={}
    with h5py.File(path,'r') as h:
        for g in sorted(h):
            if not isinstance(h[g],h5py.Group): continue
            for v in ('Tc','Tb'):
                if v in h[g] and 'Latitude' in h[g] and 'Longitude' in h[g]:
                    a=read_group(h,g,v)
                    if a['data'].ndim==2: a['data']=a['data'][...,None]
                    out[g]=a; break
    if not out: raise ValueError(f'No supported TB swaths in {path}')
    return out

def valid_geo(a):
    return np.isfinite(a['lat'])&np.isfinite(a['lon'])&np.isfinite(a['time'])&(abs(a['lat'])<=90)&(a['lon']>=-180)&(a['lon']<=360)

