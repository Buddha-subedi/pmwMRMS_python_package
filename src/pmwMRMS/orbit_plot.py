"""Four panels on native MRMS coordinates, one figure per orbit."""
import numpy as np
from pathlib import Path

def plot_orbit(path,show=True,states=None):
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cf
        proj=ccrs.PlateCarree()
    except ImportError: proj=None
    with np.load(path) as d:
        lat=d['mrms_latitude']; lon=d['mrms_longitude']; groups=list(d['groups']); mode=str(d['mode'])
        def grid(g,k,c=None):
            v=d[g+'_'+k]; v=v[...,c] if c is not None else v
            if mode=='PMWtoMRMS': return v
            if states is None: owner=d[g+'_owner']
            owner=states[g]['owner'] if states is not None else owner; good=owner>=0; result=np.full(owner.shape,np.nan,np.float32); result[good]=v.ravel()[owner[good]]; return result
        ref=max(groups,key=lambda g: (states[g]['count'].sum() if states is not None else d[g+'_count'].sum() if mode=='MRMStoPMW' else np.isfinite(d[g+'_rain']).sum()))
        panels=[(grid(ref,'rain'),'MRMS rainfall','mm/hr','turbo',0,20),(grid(ref,'snow'),'MRMS snowfall','mm/hr','Blues',0,5)]
        for low in (True,False):
            choices=[(g,i,f) for g in groups for i,f in enumerate(d[g+'_frequency_ghz']) if (f<40 if low else f>80)]
            if choices:
                g,i,f=choices[0]; panels.append((grid(g,'data',i),f'{g}: {f:g} GHz TB','K','turbo',100,320))
            else: panels.append((np.full((len(lat),len(lon)),np.nan),'TB unavailable','K','turbo',100,320))
        fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained',subplot_kw={'projection':proj} if proj else {})
        for i,(ax,(v,title,u,c,lo,hi)) in enumerate(zip(axes.flat,panels)):
            # imshow avoids allocating huge 2-D coordinate meshes on regular MRMS grid.
            dx=abs(lon[1]-lon[0])/2; dy=abs(lat[1]-lat[0])/2
            if lon[0]>lon[-1]: v=v[:,::-1]
            if lat[0]>lat[-1]: v=v[::-1]
            im=ax.imshow(v,extent=[lon.min()-dx,lon.max()+dx,lat.min()-dy,lat.max()+dy],origin='lower',interpolation='nearest',cmap=c,vmin=lo,vmax=hi,**({'transform':proj} if proj else {})); ax.set_title(f'({chr(97+i)}) {title}')
            if proj: ax.set_extent([-130,-60,20,55],crs=proj); ax.coastlines(); ax.add_feature(cf.STATES,linewidth=.4); ax.add_feature(cf.BORDERS,linewidth=.5)
            fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.8,label=u)
        fig.suptitle(f'{mode} | {Path(path).stem} | Rain/snow reference: {ref}')
        fig.savefig(Path(path).with_suffix('.png'),dpi=160)
        if show: plt.show(block=True)
        plt.close(fig)
