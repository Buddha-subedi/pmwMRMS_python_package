"""Satellite-to-MRMS nearest-footprint collocation."""
import os
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import requests
from scipy.spatial import cKDTree
from .pps import discover, interval, read_all, valid_geo, BASE
from .matching import _xyz, EARTH_KM
from .download import download_urls
from .mrms import read_grid
from .pps import orbit
from .archive import list_product, prepare_pair, overpass_times

# Explicit channel layouts; other products require user-provided frequencies.
LAYOUTS={'GMI':{'S1':[10.65,10.65,18.7,18.7,23.8,36.5,36.5,89,89],'S2':[166,166,183.31,183.31]},'SSMIS':{'S1':[19.35,19.35,22.235],'S2':[37,37],'S3':[150,183.31,183.31,183.31],'S4':[91.665,91.665]}}

LAYOUTS['ATMS']={'S1':[23.8],'S2':[31.4],'S3':[88.2],'S4':[165.5]+[183.31]*5}

def map_group(src,lat,lon,t,delta_t,max_distance_km):
    n=src['data'].shape[-1]; shape=(len(lat),len(lon),n)
    tb=np.full(shape,np.nan,np.float32); dist=tb.copy(); dt=tb.copy()
    ok=valid_geo(src)&(abs(src['time']-t)<=delta_t*60)
    for c in range(n):
        mask=ok&np.isfinite(src['data'][...,c]); y=src['lat'][mask]; x=src['lon'][mask]; v=src['data'][...,c][mask]; times=src['time'][mask]
        if not len(v): continue
        tree=cKDTree(_xyz(y,x)); radius=2*np.sin(max_distance_km/(2*EARTH_KM))
        for i in range(0,len(lat),64):
            yy,xx=np.meshgrid(lat[i:i+64],lon,indexing='ij'); d,j=tree.query(_xyz(yy.ravel(),xx.ravel()),distance_upper_bound=radius,workers=-1)
            good=np.isfinite(d); a=tb[i:i+64,:,c].copy().ravel(); b=dist[i:i+64,:,c].copy().ravel(); e=dt[i:i+64,:,c].copy().ravel()
            a[good]=v[j[good]]; b[good]=2*EARTH_KM*np.arcsin(d[good]/2); e[good]=(times[j[good]]-t)/60
            tb[i:i+64,:,c]=a.reshape(yy.shape); dist[i:i+64,:,c]=b.reshape(yy.shape); dt[i:i+64,:,c]=e.reshape(yy.shape)
    return tb,dist,dt

def process(rain_file,snow_file,pmw_files,output_file,sensor,delta_t=7.5,max_distance_km=15,frequencies=None,plot=True,show=False,rain_variable=None,snow_variable=None):
    if not np.isfinite(delta_t) or delta_t<0 or not 0<max_distance_km<20000: raise ValueError('Invalid tolerances')
    lat,lon,rain,t,ru=read_grid(rain_file,rain_variable); y,x,snow,st,su=read_grid(snow_file,snow_variable)
    if not np.array_equal(lat,y) or not np.array_equal(lon,x) or t!=st: raise ValueError('Rain and snow grids and valid times must match exactly')
    layout=frequencies or LAYOUTS.get(sensor.upper()); merged={}
    if layout is None: raise ValueError('Provide verified frequencies={group:[GHz,...]} for this sensor')
    for file in pmw_files:
        for g,src in read_all(file).items():
            if g not in layout or len(layout[g])!=src['data'].shape[-1]: raise ValueError(f'Channel metadata mismatch in {g}; provide verified frequencies')
            a,b,c=map_group(src,lat,lon,t,delta_t,max_distance_km)
            if g not in merged: merged[g]=(a,b,c)
            else:
                old=merged[g]; take=np.isfinite(a)&(~np.isfinite(old[0])|(b<old[1]))
                for dst,new in zip(old,(a,b,c)): dst[take]=new[take]
    if not merged: raise ValueError('No PMW swaths supplied')
    groups=sorted(merged); freq=np.array([f for g in groups for f in layout[g]]); names=np.array([f'{g}_{i+1}' for g in groups for i in range(len(layout[g]))]); tb,dist,dt=[np.concatenate([merged[g][i] for g in groups],axis=-1) for i in range(3)]
    out=Path(output_file).with_suffix('.npz'); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,latitude=lat,longitude=lon,mrms_rain=rain,mrms_snow=snow,mrms_time=t,rain_units=ru,snow_units=su,pmw_tb=tb,distance_km=dist,delta_t_minutes=dt,frequency_ghz=freq,channel_names=names,coincidence_mask=np.isfinite(tb),pmw_files=np.array(list(map(str,pmw_files))))
    if plot:
        from .visualize import plot_result
        plot_result(out,show=show)
    return str(out)

def automatic(sensor,satellite,start_date,end_date,mrms_pairs=None,output_dir='./pmwMRMS_data',version=None,email=None,password=None,delta_t=7.5,max_distance_km=15,frequencies=None,plot=True,show=False,base=BASE,max_cases=1):
    """mrms_pairs: iterable of (rain path or HTTPS URL, snow path or HTTPS URL)."""
    start=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc); end=datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)+timedelta(days=1)
    email=email or os.getenv('PPS_USERNAME') or input('PPS registered email: '); password=password or os.getenv('PPS_PASSWORD') or email
    root=Path(output_dir); pad=timedelta(minutes=delta_t)
    def pred(n):
        tok=n.upper().split('.'); return tok[0] in ('1C','1C-R') and sensor.upper() in tok and satellite.upper() in tok and (not version or version.upper() in tok)
    with requests.Session() as s:
        s.auth=(email,password); urls=discover(s,start-pad-timedelta(days=1),end+pad,'1C',pred,base)
    if not urls: raise RuntimeError('No PPS granules found')
    results=[]
    if max_cases is not None and max_cases < 1: raise ValueError('max_cases must be positive or None')
    if mrms_pairs is None:
        cache={}; used=set()
        for url in urls:
            a,b=interval(url)
            if b<start or a>=end: continue
            file=download_urls([url],root/'PMW',email,password)[0]
            times=overpass_times(file,start,end)
            if not times.size: continue
            target=datetime.fromtimestamp(float(np.median(times)),timezone.utc)
            day=target.date()
            if day not in cache:
                rate=list_product('PrecipRate',target); flag=list_product('PrecipFlag',target)
                cache[day]=(rate,flag)
            rate,flag=cache[day]
            common=[t for t in rate.keys() & flag.keys() if start<=t<end and abs((t-target).total_seconds())<=delta_t*60 and (url,t) not in used]
            if not common: continue
            stamp=min(common,key=lambda t:abs((t-target).total_seconds()))
            used.add((url,stamp))
            print(f'[MRMS] Selected {stamp.isoformat()}',flush=True)
            rain,snow=prepare_pair(rate[stamp],flag[stamp],root/'MRMS')
            results.append(process(rain,snow,[file],root/'matches'/f'{satellite}_{sensor}_{stamp:%Y%m%d_%H%M%S}_{orbit(url):06d}.npz',sensor,delta_t,max_distance_km,frequencies,plot,show))
            if max_cases is not None and len(results)>=max_cases: break
        if not results: raise RuntimeError('No CONUS overpass with available MRMS rate and flag within tolerance')
        return results
    for rain,snow in mrms_pairs:
        def local(p): return download_urls([str(p)],root/'MRMS')[0] if str(p).startswith('https://') else p
        rain=local(rain); snow=local(snow); _,_,_,t,_=read_grid(rain); stamp=datetime.fromtimestamp(t,timezone.utc)
        if not start<=stamp<end: continue
        candidates=[u for u in urls if interval(u)[0]<=stamp+pad and interval(u)[1]>=stamp-pad]
        if not candidates: continue
        files=download_urls(candidates,root/'PMW',email,password)
        results.append(process(rain,snow,files,root/'matches'/f'{satellite}_{sensor}_{stamp:%Y%m%d_%H%M%S}.npz',sensor,delta_t,max_distance_km,frequencies,plot,show))
    return results
