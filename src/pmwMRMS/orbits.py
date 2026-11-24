"""Orbit-wise nearest-center matching and unweighted MRMS aggregation."""
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import numpy as np
import requests
from scipy.spatial import cKDTree
from .pps import BASE, discover, interval, orbit, read_all, valid_geo
from .matching import _xyz, EARTH_KM
from .automatic import LAYOUTS
from .archive import AWS, list_product, phase_rates
from .download import download_urls
from .mrms import read_grid

B= (20.,55.,-130.,-60.) # MRMS CONUS rectangular product domain

def domain_mask(a):
    x=(a['lon']+180)%360-180; y=a['lat']
    return valid_geo(a)&(y>=B[0])&(y<=B[1])&(x>=B[2])&(x<=B[3])

def nearest_time(t, stamps):
    j=np.searchsorted(stamps,t); a=np.clip(j-1,0,len(stamps)-1); b=np.clip(j,0,len(stamps)-1)
    return np.where(abs(t-stamps[a])<=abs(t-stamps[b]),a,b)

def assignments(src, lat, lon, radius, stamps, delta):
    """Yield grid rows and nearest native flat footprint IDs, -1 if unmatched."""
    ok=valid_geo(src)&np.any(np.isfinite(src['data']),axis=-1)
    ids=np.flatnonzero(ok); times=src['time'].ravel()
    if not len(ids): return
    tree=cKDTree(_xyz(src['lat'][ok],src['lon'][ok])); chord=2*np.sin(radius/(2*EARTH_KM))
    for i in range(0,len(lat),64):
        yy,xx=np.meshgrid(lat[i:i+64],lon,indexing='ij'); d,j=tree.query(_xyz(yy.ravel(),xx.ravel()),distance_upper_bound=chord,workers=-1)
        good=np.isfinite(d); owner=np.full(d.shape,-1,np.int32); owner[good]=ids[j[good]]
        ti=np.full(d.shape,-1,np.int32); k=np.flatnonzero(good); q=nearest_time(times[owner[k]],stamps); take=abs(times[owner[k]]-stamps[q])<=delta*60
        ti[k[take]]=q[take]; owner[k[~take]]=-1
        yield i,owner.reshape(yy.shape),ti.reshape(yy.shape)

def aggregate(owner, rain, snow, n):
    ok=(owner>=0)&np.isfinite(rain)&np.isfinite(snow); ids=owner[ok]
    count=np.bincount(ids,minlength=n); r=np.bincount(ids,weights=rain[ok],minlength=n); s=np.bincount(ids,weights=snow[ok],minlength=n)
    return count,r,s

def _run(mode,sensor,satellite,start_date,end_date,output_dir='./pmwMRMS_data',version=None,email=None,password=None,buffer_minutes=2,delta_t=2,max_distance_km=None,max_cases=None,frequencies=None,plot=True,show=True,base=BASE):
    if max_distance_km is None: max_distance_km=float(input('Maximum PMW center to MRMS pixel distance (km): '))
    if not 0<max_distance_km<20000 or not np.isfinite(delta_t) or delta_t<0 or not np.isfinite(buffer_minutes) or buffer_minutes<0: raise ValueError('Invalid distance or time tolerance')
    if max_cases is not None and max_cases<1: raise ValueError('max_cases must be positive')
    start=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc); end=datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)+timedelta(days=1)
    if end<=start: raise ValueError('Invalid date range')
    email=email or os.getenv('PPS_USERNAME') or input('PPS registered email: ').strip(); password=password or os.getenv('PPS_PASSWORD') or email
    root=Path(output_dir); layout=frequencies or LAYOUTS.get(sensor.upper()); cache={}; results=[]
    if layout is None: raise ValueError('Provide frequencies mapping for this sensor')
    def pred(n):
        tok=n.upper().split('.'); return tok[0] in ('1C','1C-R') and sensor.upper() in tok and satellite.upper() in tok and (not version or version.upper() in tok)
    with requests.Session() as session:
        session.auth=(email,password); urls=discover(session,start-timedelta(days=1),end,'1C',pred,base)
    for url in urls:
        a,b=interval(url)
        if b<start or a>=end: continue
        file=download_urls([url],root/'PMW',email,password)[0]; groups=read_all(file)
        chunks=[v['time'][domain_mask(v)] for v in groups.values()]; chunks=[x for x in chunks if x.size]
        if not chunks: continue
        lo=min(x.min() for x in chunks); hi=max(x.max() for x in chunks)
        if hi<start.timestamp() or lo>=end.timestamp(): continue
        begin=datetime.fromtimestamp(lo,timezone.utc)-timedelta(minutes=buffer_minutes); finish=datetime.fromtimestamp(hi,timezone.utc)+timedelta(minutes=buffer_minutes)
        print(f'[OVERPASS] {orbit(url)} MRMS window {begin} to {finish}',flush=True)
        rate={}; flag={}; day=begin
        while day.date()<=finish.date():
            if day.date() not in cache: cache[day.date()]=(list_product('PrecipRate',day),list_product('PrecipFlag',day))
            r,f=cache[day.date()]; rate.update(r); flag.update(f); day+=timedelta(days=1)
        keys=sorted(t for t in rate.keys()&flag.keys() if begin<=t<=finish)
        if not keys: print('[SKIP] No paired MRMS timestamps'); continue
        stamps=np.array([t.timestamp() for t in keys]); print(f'[MRMS] {len(keys)} pairs',flush=True)
        states={}; lat=lon=None
        for index,t in enumerate(keys):
            paths=download_urls([AWS+'/'+quote(rate[t]),AWS+'/'+quote(flag[t])],root/'MRMS')
            y,x,r,rt,_=read_grid(paths[0]); fy,fx,f,ft,_=read_grid(paths[1])
            if rt!=ft or abs(rt-stamps[index])>1 or not np.array_equal(y,fy) or not np.array_equal(x,fx): raise ValueError('MRMS grid/time mismatch')
            rain,snow=phase_rates(r,f); del r,f
            if lat is None:
                lat,lon=y,x
                for g,src in groups.items():
                    if g not in layout or len(layout[g])!=src['data'].shape[-1]: raise ValueError(f'Frequency metadata mismatch: {g}; supply correct frequencies')
                    owner=np.full((len(y),len(x)),-1,np.int32); ti=owner.copy()
                    for i,o,j in assignments(src,y,x,max_distance_km,stamps,delta_t): owner[i:i+len(o)]=o; ti[i:i+len(o)]=j
                    n=src['lat'].size; states[g]=dict(owner=owner,ti=ti,count=np.zeros(n,np.int64),rs=np.zeros(n),ss=np.zeros(n))
                    if mode=='PMWtoMRMS': states[g].update(rain=np.full(owner.shape,np.nan,np.float32),snow=np.full(owner.shape,np.nan,np.float32))
            elif not np.array_equal(lat,y) or not np.array_equal(lon,x): raise ValueError('MRMS grid changed within orbit')
            for state in states.values():
                select=state['ti']==index; ids=state['owner'][select]; rr=rain[select]; ss=snow[select]
                valid=np.isfinite(rr)&np.isfinite(ss); n=len(state['count']); c,r_sum,s_sum=aggregate(ids,rr,ss,n)
                state['count']+=c; state['rs']+=r_sum; state['ss']+=s_sum
                if mode=='PMWtoMRMS': state['rain'][select]=rr; state['snow'][select]=ss
                # Drop cells without a valid precipitation reference for plotting and output.
                state['owner'][select]=np.where(valid,ids,-1)
            del rain,snow
        if not any(s['count'].any() for s in states.values()): continue
        target=root/'matches'/mode/f'{satellite}_{sensor}_{keys[0]:%Y%m%d_%H%M%S}_{orbit(url):06d}.npz'; target.parent.mkdir(parents=True,exist_ok=True)
        out=dict(mode=mode,groups=np.array(list(states)),mrms_latitude=lat,mrms_longitude=lon,mrms_times=stamps,max_distance_km=max_distance_km,pmw_file=str(file))
        for g,s in states.items():
            src=groups[g]; owner=s['owner']; good=owner>=0; count=s['count']; shape=src['lat'].shape
            out[g+'_frequency_ghz']=np.array(layout[g]); out[g+'_owner']=owner
            if mode=='MRMStoPMW':
                for k in ('lat','lon','time','data'): out[g+'_'+k]=src[k]
                out[g+'_count']=count.reshape(shape)
                for k,v in [('rain',s['rs']),('snow',s['ss'])]: out[g+'_'+k]=np.divide(v,count,out=np.full(v.shape,np.nan),where=count>0).reshape(shape).astype(np.float32)
            else:
                out[g+'_rain']=s['rain']; out[g+'_snow']=s['snow']; tb=np.full((*owner.shape,src['data'].shape[-1]),np.nan,np.float32); tb[good]=src['data'].reshape(-1,tb.shape[-1])[owner[good]]; out[g+'_data']=tb
                ti=s['ti'].copy(); ti[~good]=-1; out[g+'_time_index']=ti
        np.savez_compressed(target,**out)
        if plot:
            from .orbit_plot import plot_orbit
            plot_orbit(target,show=show,states=states)
        results.append(str(target)); print('[SAVED]',target,flush=True)
        if max_cases is not None and len(results)>=max_cases: break
    return results

def PMWtoMRMS(*args,**kwargs):
    """Map TB onto MRMS grid; preserve original matched precipitation."""
    return _run('PMWtoMRMS',*args,**kwargs)

def MRMStoPMW(*args,**kwargs):
    """Average valid MRMS cells assigned to each native PMW footprint."""
    return _run('MRMStoPMW',*args,**kwargs)
