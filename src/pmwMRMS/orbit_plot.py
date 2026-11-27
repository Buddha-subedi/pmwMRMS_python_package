"""Consistent four-panel maps for both orbit mapping directions."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def plot_orbit(path, show=True, states=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.patches import Patch
    from matplotlib.cm import ScalarMappable
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    pc = ccrs.PlateCarree()
    # Plate Carree provides straight, accurately labeled geographic axes.
    proj = ccrs.PlateCarree()
    ocean = '#D6EAF5'
    phase_cmap = colors.ListedColormap(['gainsboro','red','blue'])
    phase_norm = colors.BoundaryNorm([-.5,.5,1.5,2.5],3)
    rain_cmap = colors.LinearSegmentedColormap.from_list('rain',
        ['gainsboro','seagreen','green','yellow','darkorange','red'])
    snow_cmap = colors.LinearSegmentedColormap.from_list('snow',
        ['gainsboro','cyan','blue','blueviolet'])
    rn, sn = colors.Normalize(0,4), colors.Normalize(0,1)
    with np.load(path) as d:
        lat, lon = d['mrms_latitude'], d['mrms_longitude']
        groups, mode = list(d['groups']), str(d['mode'])
        if 'reference_rain' not in d or 'reference_snow' not in d:
            raise ValueError('Original MRMS reference missing. Regenerate orbit with updated package.')
        fig = plt.figure(figsize=(19,11.6), facecolor='white')
        # Fixed map rectangles: preserve geographic aspect and equal panel sizes.
        left, right, gap = .065, .98, .012
        width = (right - left - gap) / 2
        height = width * fig.get_figwidth() / fig.get_figheight() * (30 / 65)
        row_gap = .045
        bottom = .18
        axes = np.array([[fig.add_axes([left+j*(width+gap), bottom+(1-i)*(height+row_gap), width, height], projection=proj) for j in range(2)] for i in range(2)])
        def setup(ax,i,title):
            ax.set_extent([-130,-65,23,53],crs=pc)
            ax.set_facecolor(ocean)
            ax.add_feature(cf.LAND.with_scale('50m'),facecolor='#FAFAFA',zorder=0)
            ax.add_feature(cf.LAKES.with_scale('50m'),facecolor=ocean,edgecolor='#8AA5B8',linewidth=.35,zorder=1)
            ax.add_feature(cf.STATES.with_scale('50m'),edgecolor='#777777',linewidth=.4,zorder=10)
            ax.add_feature(cf.BORDERS.with_scale('50m'),linewidth=.65,zorder=11)
            ax.coastlines('50m',linewidth=.75,zorder=12)
            ax.set_xticks(np.arange(-130,-64,10), crs=pc)
            ax.set_yticks(np.arange(25,54,5), crs=pc)
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(axis='both', which='major', direction='out', length=5, width=1.4, labelsize=12, labelbottom=i>=2, labelleft=i%2==0, bottom=i>=2, left=i%2==0)
            for tick in ax.get_xticklabels()+ax.get_yticklabels():
                tick.set_fontweight('bold')
            for x,y,t in [(-126.5,33.5,'Pacific\nOcean'),(-70.5,32.5,'Atlantic\nOcean'),(-90,25.5,'Gulf of Mexico')]:
                ax.text(x,y,t,transform=pc,ha='center',va='center',fontsize=10,fontstyle='italic',color='#315B80',zorder=15)
            ax.set_title(f'({chr(97+i)})',loc='left',fontweight='bold',fontsize=20, pad=7)
            ax.set_title(title,fontsize=18,fontweight='bold',pad=7)
            ax.spines['geo'].set_linewidth(1)
        def regular(ax,v,cmap,norm,z=3):
            # Render at native grid spacing without allocating coordinate meshes.
            dx=abs(lon[1]-lon[0])/2; dy=abs(lat[1]-lat[0])/2
            if lon[0]>lon[-1]: v=v[:,::-1]
            if lat[0]>lat[-1]: v=v[::-1]
            return ax.imshow(v,extent=[lon.min()-dx,lon.max()+dx,lat.min()-dy,lat.max()+dy],origin='lower',interpolation='nearest',transform=pc,cmap=cmap,norm=norm,zorder=z)
        def native(ax,v,cmap,norm,z=3):
            return ax.pcolormesh(x,y,v,transform=pc,cmap=cmap,norm=norm,shading='nearest',rasterized=True,zorder=z)
        def phase(r,s):
            return np.where(np.isfinite(r)&np.isfinite(s),np.where((r==0)&(s==0),0,np.where(r>=s,1,2)),np.nan).astype('float32')
        def pair(axp,axr,r,s,draw):
            p=phase(r,s)
            draw(axp,p,phase_cmap,phase_norm)
            draw(axr,np.where(np.isfinite(p),0,np.nan),phase_cmap,phase_norm)
            draw(axr,np.where(p==1,r,np.nan),rain_cmap,rn,4)
            draw(axr,np.where(p==2,s,np.nan),snow_cmap,sn,5)
            axp.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(['gainsboro','red','blue'],['No Precip','Rain','Snow'])],loc='lower left',prop={'size':13,'weight':'bold'},framealpha=.95,edgecolor='#999999',handlelength=1.8)
        setup(axes[0,0],0,'Original MRMS precipitation phase')
        setup(axes[0,1],1,'Original MRMS precipitation rate')
        pair(*axes[0],d['reference_rain'],d['reference_snow'],regular)
        if mode=='MRMStoPMW':
            ref=max(groups,key=lambda g: int(d[g+'_count'].sum()))
            x,y=d[ref+'_lon'],d[ref+'_lat']; x=(x+180)%360-180
            setup(axes[1,0],2,f'MRMS phase on PMW grid ({ref})')
            setup(axes[1,1],3,f'MRMS rate on PMW grid ({ref})')
            pair(*axes[1],d[ref+'_rain'],d[ref+'_snow'],native)
        elif mode=='PMWtoMRMS':
            for j,low in enumerate([True,False]):
                ax=axes[1,j]; choices=[(g,i,f) for g in groups for i,f in enumerate(d[g+'_frequency_ghz']) if (f<40 if low else f>80)]
                if choices:
                    g,i,f=choices[0]; setup(ax,j+2,f'{g}: {f:g} GHz TB on MRMS grid')
                    im=regular(ax,d[g+'_data'][...,i],'turbo',colors.Normalize(100,320))
                else: setup(ax,j+2,'TB unavailable')
        else: raise ValueError(f'Unknown mapping mode: {mode}')
        # Dedicated figure-level colorbars never resize the map axes.
        scales = [(rain_cmap,rn,r'Rain rate (mm hr$^{-1}$)',np.arange(0,4.1,1)),
                  (snow_cmap,sn,r'Snow rate (mm hr$^{-1}$)',np.arange(0,1.01,.2))]
        if mode == 'PMWtoMRMS':
            scales.append(('turbo',colors.Normalize(100,320),'Brightness temperature (K)',[100,150,200,250,300,320]))
        n = len(scales)
        width = .25 if n == 3 else .32
        gap = .055
        start = (.065+.98 - (n*width+(n-1)*gap))/2
        for j,(cmap,norm,label,ticks) in enumerate(scales):
            cax=fig.add_axes([start+j*(width+gap),.085,width,.022])
            cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=ticks,orientation='horizontal',extend='neither')
            cb.set_label(label,fontsize=13,fontweight='bold',labelpad=5)
            cb.ax.tick_params(labelsize=12,width=1.4,length=5,direction='out')
            for tick in cb.ax.get_xticklabels(): tick.set_fontweight('bold')
            cb.outline.set_linewidth(1.1)
        stamp=datetime.fromtimestamp(float(d['reference_time']),timezone.utc)
        orbit_id = f"{int(d['orbit_number']):06d}" if 'orbit_number' in d else Path(path).stem.rsplit('_', 1)[-1]
        fig.suptitle(f'MRMS time: {stamp:%Y-%m-%d %H:%M UTC} | PMW orbit: {orbit_id} | {mode}',fontsize=20,fontweight='bold',y=.985)
        fig.savefig(Path(path).with_suffix('.png'),dpi=300,bbox_inches='tight')
        if show: plt.show(block=True)
        plt.close(fig)
