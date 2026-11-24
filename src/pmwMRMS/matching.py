"""Exact nearest spatial sample, followed by beam-level time and distance screening."""
import numpy as np
from scipy.spatial import cKDTree
EARTH_KM=6371.0088

def _xyz(lat, lon):
    phi=np.deg2rad(lat); lam=np.deg2rad(lon)
    return np.column_stack((np.cos(phi)*np.cos(lam),np.cos(phi)*np.sin(lam),np.sin(phi)))

def map_nearest(source, target, max_distance_km=214., pixel_time_minutes=7.5,
                enforce_distance=True, chunk_size=200000):
    """Map source TBs onto target pixels; never interpolate/average TBs.

    The spatial nearest source pixel is selected first, then screened in time.
    If it fails time screening, output NaN (do not switch to a farther neighbor).
    """
    shape=target['lat'].shape
    if source['lat'].shape != source['lon'].shape or source['time'].shape != source['lat'].shape: raise ValueError('Invalid source geometry/time')
    valid=(np.isfinite(source['lat']) & np.isfinite(source['lon']) & np.isfinite(source['time']) & (np.abs(source['lat'])<=90))
    vals=source['data']; nchan=vals.shape[2] if vals.ndim==3 else 1
    if vals.shape[:2]!=valid.shape: raise ValueError('Source data shape mismatch')
    valid &= np.any(np.isfinite(vals),axis=2) if vals.ndim==3 else np.isfinite(vals)
    if not valid.any(): raise ValueError('No valid source pixels')
    source_indices=np.flatnonzero(valid.ravel()); tree=cKDTree(_xyz(source['lat'][valid],source['lon'][valid]))
    output=np.full((np.prod(shape),nchan),np.nan,dtype=np.float32)
    dist=np.full(np.prod(shape),np.nan,dtype=np.float32)
    dt=np.full(np.prod(shape),np.nan,dtype=np.float32)
    accepted=np.zeros(np.prod(shape),dtype=bool)
    tl=target['lat'].ravel(); to=target['lon'].ravel(); tt=target['time'].ravel()
    indices=np.flatnonzero(np.isfinite(tl)&np.isfinite(to)&np.isfinite(tt)&(np.abs(tl)<=90))
    flat_vals=vals.reshape(-1,nchan); flat_time=source['time'].ravel()
    for start in range(0,len(indices),chunk_size):
        ids=indices[start:start+chunk_size]
        chord,idx=tree.query(_xyz(tl[ids],to[ids]),k=1,workers=-1)
        matched=source_indices[idx]
        km=2*EARTH_KM*np.arcsin(np.clip(chord/2,0,1))
        delta=np.abs(flat_time[matched]-tt[ids])/60.
        ok=(delta<=pixel_time_minutes) & ((km<=max_distance_km) if enforce_distance else True)
        dist[ids]=km; dt[ids]=delta; accepted[ids]=ok
        output[ids[ok]]=flat_vals[matched[ok]]
    return dict(tb=output.reshape(*shape,nchan), valid_mask=accepted.reshape(shape),
                distance_km=dist.reshape(shape), delta_t_minutes=dt.reshape(shape))
