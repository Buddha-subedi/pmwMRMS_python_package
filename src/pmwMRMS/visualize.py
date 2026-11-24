import numpy as np

def plot_result(path,show=False):
    import matplotlib.pyplot as plt
    from pathlib import Path
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cf
        proj=ccrs.PlateCarree()
    except ImportError: proj=None
    with np.load(path) as d:
        f=d['frequency_ghz']; tb=d['pmw_tb']; names=d['channel_names']; lat=d['latitude']; lon=d['longitude']
        fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained',subplot_kw={'projection':proj} if proj else {})
        panels=[(d['mrms_rain'],'MRMS rainfall',str(d['rain_units']),'turbo',0,20),(d['mrms_snow'],'MRMS snowfall',str(d['snow_units']),'Blues',0,5)]
        for mask,label in [(f<40,'Low-frequency TB'),(f>80,'High-frequency TB')]:
            ids=np.flatnonzero(mask)
            if len(ids):
                i=max(ids,key=lambda i:np.isfinite(tb[...,i]).sum()); panels.append((tb[...,i],f'{label}: {f[i]:g} GHz ({names[i]})','K','turbo',100,320))
            else: panels.append((np.full(tb.shape[:2],np.nan),label+' (unavailable)','K','turbo',100,320))
        for i,(ax,(v,title,unit,cmap,vmin,vmax)) in enumerate(zip(axes.flat,panels)):
            m=ax.pcolormesh(lon,lat,v,shading='auto',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True,**({'transform':proj} if proj else {})); ax.set_title(f'({chr(97+i)}) {title}')
            if proj: ax.set_extent([-125,-66,24,50],crs=proj); ax.coastlines(); ax.add_feature(cf.STATES,linewidth=.4); ax.add_feature(cf.BORDERS,linewidth=.5)
            else: ax.set(xlim=(-125,-66),ylim=(24,50),xlabel='Longitude',ylabel='Latitude')
            fig.colorbar(m,ax=ax,orientation='horizontal',shrink=.8,label=unit)
        fig.savefig(Path(path).with_suffix('.png'),dpi=180)
        if show: plt.show()
        plt.close(fig)
