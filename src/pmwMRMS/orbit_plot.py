"""Consistent four-panel maps for both orbit mapping directions."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def plot_orbit(path, show=True, states=None):
    from scipy.ndimage import binary_fill_holes
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.patches import Patch
    from matplotlib.cm import ScalarMappable
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import matplotlib.ticker as mticker
    map_extent = [-130, -65, 23, 53]
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
        if mode == 'MRMStoPMW' and ('reference_rain' not in d or 'reference_snow' not in d):
            raise ValueError('Original MRMS reference missing. Regenerate orbit with updated package.')
        fig = plt.figure(figsize=(22,16.2), facecolor='white')
        # Fixed map rectangles with a 1.4 vertical aspect multiplier.
        left, right, gap = .065, .98, .012
        width = (right - left - gap) / 2
        height = width * fig.get_figwidth() / fig.get_figheight() * (30 / 65) * 1.4
        row_gap = .040
        bottom = .105
        axes = np.array([[fig.add_axes([left+j*(width+gap), bottom+(1-i)*(height+row_gap), width, height], projection=proj) for j in range(2)] for i in range(2)])
        def setup(ax,i,title):
            ax.set_extent(map_extent,crs=pc)
            ax.set_aspect(1.4, adjustable="box")
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
            ax.set_title(f'({chr(97+i)})',loc='left',fontweight='bold',fontsize=24, pad=7)
            ax.set_title(title,fontsize=20,fontweight='bold',pad=7)
            ax.spines['geo'].set_linewidth(1)
        def regular(ax,v,cmap,norm,z=3):
            # Render at native grid spacing without allocating coordinate meshes.
            dx=abs(lon[1]-lon[0])/2; dy=abs(lat[1]-lat[0])/2
            if lon[0]>lon[-1]: v=v[:,::-1]
            if lat[0]>lat[-1]: v=v[::-1]
            return ax.imshow(v,extent=[lon.min()-dx,lon.max()+dx,lat.min()-dy,lat.max()+dy],origin='lower',interpolation='nearest',transform=pc,cmap=cmap,norm=norm,zorder=z,aspect=1.4)
        def matched_outline(ax, mask):
            # Outline the matched footprint, not individual internal TB gaps.
            mask = binary_fill_holes(np.asarray(mask, dtype=bool))
            if not mask.any():
                return
            # Pad with unmatched cells so edges at the MRMS grid limit close.
            xp = np.r_[2*lon[0]-lon[1], lon, 2*lon[-1]-lon[-2]]
            yp = np.r_[2*lat[0]-lat[1], lat, 2*lat[-1]-lat[-2]]
            ax.contour(xp, yp, np.pad(mask, 1).astype(np.uint8),
                       levels=[0.5], colors='black', linewidths=1.3,
                       transform=pc, zorder=14)
        def native_outline(ax, x, y):
            # Perimeter of the native swath's outermost beam centers.
            xx = np.r_[x[0,:], x[1:,-1], x[-1,-2::-1], x[-2::-1,0]]
            yy = np.r_[y[0,:], y[1:,-1], y[-1,-2::-1], y[-2::-1,0]]
            xx = np.r_[xx, xx[0]]; yy = np.r_[yy, yy[0]]
            invalid = ~np.isfinite(xx) | ~np.isfinite(yy) | (np.abs(yy)>90)
            xx[invalid] = np.nan; yy[invalid] = np.nan
            jumps = np.r_[False, np.abs(np.diff(xx)) > 180]
            xx[jumps] = np.nan; yy[jumps] = np.nan
            ax.plot(xx, yy, color='black', linewidth=1.3,
                    transform=pc, zorder=14)
        def native(ax,v,cmap,norm,z=3):
            v = np.asarray(v)
            if x.shape != y.shape or v.shape != x.shape:
                raise ValueError(f"Native grid shape mismatch: lon={x.shape}, lat={y.shape}, data={v.shape}")
            geo = np.isfinite(x) & np.isfinite(y) & (np.abs(y) <= 90)
            west, east, south, north = map_extent
            visible = geo & (x >= west-2) & (x <= east+2) & (y >= south-2) & (y <= north+2)
            rows = np.flatnonzero(visible.any(axis=1))
            if not rows.size:
                return None
            sl = slice(max(0, rows[0]-1), min(x.shape[0], rows[-1]+2))
            xx, yy, vv = x[sl], y[sl], v[sl]
            valid = geo[sl] & np.isfinite(vv)
            if not valid.any():
                return None
            if geo[sl].all() and min(xx.shape) >= 2 and not np.any(np.abs(np.diff(xx, axis=1)) > 180):
                return ax.pcolormesh(xx, yy, vv, transform=pc, cmap=cmap, norm=norm, shading='nearest', rasterized=True, zorder=z)
            # Invalid coordinates cannot be passed to pcolormesh. Plot only
            # genuine valid footprint centers, without inventing geolocation.
            return ax.scatter(xx[valid], yy[valid], c=vv[valid], s=5, marker='s', linewidths=0, transform=pc, cmap=cmap, norm=norm, rasterized=True, zorder=z)
        def phase(r,s):
            return np.where(np.isfinite(r)&np.isfinite(s),np.where((r==0)&(s==0),0,np.where(r>=s,1,2)),np.nan).astype('float32')
        def pair(axp,axr,r,s,draw):
            p=phase(r,s)
            draw(axp,p,phase_cmap,phase_norm)
            draw(axr,np.where(np.isfinite(p),0,np.nan),phase_cmap,phase_norm)
            draw(axr,np.where(p==1,r,np.nan),rain_cmap,rn,4)
            draw(axr,np.where(p==2,s,np.nan),snow_cmap,sn,5)
            axp.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(['gainsboro','red','blue'],['No Precip','Rain','Snow'])],loc='lower left',prop={'size':13,'weight':'bold'},framealpha=.95,edgecolor='#999999',handlelength=1.8)
        if mode=='MRMStoPMW':
            ref=max(groups,key=lambda g: int(d[g+'_count'].sum()))
            x,y=d[ref+'_lon'],d[ref+'_lat']; x=(x+180)%360-180
            rr,ss=d[ref+'_rain'],d[ref+'_snow']
            owner=d[ref+'_owner']
            overlap=owner>=0
            r=np.where(overlap,d['reference_rain'],np.nan)
            s=np.where(overlap,d['reference_snow'],np.nan)
            stamp=datetime.fromtimestamp(float(d['reference_time']),timezone.utc)
            setup(axes[0,0],0,'Coincident MRMS precipitation rate')
            setup(axes[0,1],1,f'MRMS rate on PMW grid ({ref})')
            def rate_only(ax,r,s,draw):
                p=phase(r,s)
                draw(ax,np.where(np.isfinite(p),0,np.nan),phase_cmap,phase_norm)
                draw(ax,np.where(p==1,r,np.nan),rain_cmap,rn,4)
                draw(ax,np.where(p==2,s,np.nan),snow_cmap,sn,5)
            rate_only(axes[0,0],r,s,regular)
            matched_outline(axes[0,0],overlap)
            rate_only(axes[0,1],rr,ss,native)
            native_outline(axes[0,1],x,y)
            axes[0,0].set_title(f'Coincident MRMS rate ({stamp:%H:%M} UTC reference)',fontsize=18,fontweight='bold',pad=7)
            axes[0,1].set_title(f'MRMS on PMW grid ({ref}; ref. {stamp:%H:%M} UTC)',fontsize=18,fontweight='bold',pad=7)
            tb_artists=[]; tb_min,tb_max=np.inf,-np.inf
            west,east,south,north=map_extent
            for j,low in enumerate([True,False]):
                ax=axes[1,j]
                choices=[(g,i,f) for g in groups for i,f in enumerate(d[g+'_frequency_ghz']) if (f<40 if low else f>80)]
                if not choices:
                    setup(ax,j+2,'TB unavailable'); continue
                g,i,f=choices[0]
                x,y=d[g+'_lon'],d[g+'_lat']; x=(x+180)%360-180
                tb=d[g+'_data'][...,i]
                valid=np.isfinite(tb)&np.isfinite(d[g+'_rain'])&np.isfinite(d[g+'_snow'])
                clipped=valid & (x>=west)&(x<=east)&(y>=south)&(y<=north)
                setup(ax,j+2,f'{g}: {f:g} GHz TB on PMW grid')
                if clipped.any():
                    tb_min=min(tb_min,float(tb[clipped].min())); tb_max=max(tb_max,float(tb[clipped].max()))
                artist=native(ax,np.where(valid,tb,np.nan),'turbo',colors.Normalize(100,320))
                if artist is not None: tb_artists.append(artist)
                native_outline(ax,x,y)
            tb_norm=None
            if np.isfinite(tb_min) and np.isfinite(tb_max):
                if tb_min==tb_max: tb_min-=.5; tb_max+=.5
                tb_norm=colors.Normalize(tb_min,tb_max)
                for artist in tb_artists: artist.set_norm(tb_norm)
        elif mode=='PMWtoMRMS':
            # Use the exact saved beam-time match for the displayed channels.
            # Low-frequency group takes priority in overlapping coverage;
            # high-frequency group fills gaps. Never average timestamps/rates.
            r = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
            s = np.full_like(r, np.nan)
            time_index = np.full(r.shape, -1, dtype=np.int32)
            # Shared TB scale from finite matched pixels inside the visible map.
            west, east, south, north = map_extent
            visible = ((lat >= south) & (lat <= north))[:, None] & ((lon >= west) & (lon <= east))[None, :]
            tb_min, tb_max = np.inf, -np.inf
            tb_artists = []
            for j,low in enumerate([True,False]):
                ax=axes[1,j]
                choices=[(g,i,f) for g in groups for i,f in enumerate(d[g+'_frequency_ghz']) if (f<40 if low else f>80)]
                if not choices:
                    setup(ax,j+2,'TB unavailable')
                    continue
                g,i,f=choices[0]
                tb=d[g+'_data'][...,i]
                rr,ss,ti=d[g+'_rain'],d[g+'_snow'],d[g+'_time_index']
                valid=np.isfinite(tb)&np.isfinite(rr)&np.isfinite(ss)&(rr>=0)&(ss>=0)&(ti>=0)&(ti<len(d['mrms_times']))
                setup(ax,j+2,f'{g}: {f:g} GHz TB on MRMS grid')
                clipped = valid & visible
                if clipped.any():
                    tb_min = min(tb_min, float(tb[clipped].min()))
                    tb_max = max(tb_max, float(tb[clipped].max()))
                tb_artists.append(regular(ax,np.where(valid,tb,np.nan),'turbo',colors.Normalize(100,320)))
                matched_outline(ax, valid)
                take=valid & (time_index<0)
                r[take],s[take],time_index[take]=rr[take],ss[take],ti[take]
            tb_norm = None
            if np.isfinite(tb_min) and np.isfinite(tb_max):
                if tb_min == tb_max:
                    tb_min -= 0.5
                    tb_max += 0.5
                tb_norm = colors.Normalize(tb_min, tb_max)
                for artist in tb_artists:
                    artist.set_norm(tb_norm)
            setup(axes[0,0],0,'Coincident MRMS precipitation phase')
            setup(axes[0,1],1,'Coincident MRMS precipitation rate')
            pair(*axes[0],r,s,regular)
            for ax in axes[0]:
                matched_outline(ax, time_index >= 0)
            used=np.unique(time_index[time_index>=0])
        else: raise ValueError(f'Unknown mapping mode: {mode}')
        # Dedicated figure-level colorbars never resize the map axes.
        scales = [(rain_cmap,rn,r'Rain rate (mm hr$^{-1}$)',np.arange(0,4.1,1)),
                  (snow_cmap,sn,r'Snow rate (mm hr$^{-1}$)',np.arange(0,1.01,.2))]
        if tb_norm is not None:
            ticks = np.linspace(tb_norm.vmin, tb_norm.vmax, 6)
            scales.append(('turbo',tb_norm,'Brightness temperature (K)',ticks))
        n = len(scales)
        width = .25 if n == 3 else .32
        gap = .055
        start = (.065+.98 - (n*width+(n-1)*gap))/2
        for j,(cmap,norm,label,ticks) in enumerate(scales):
            cax=fig.add_axes([start+j*(width+gap),.055,width,.018])
            cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=ticks,orientation='horizontal',extend='neither')
            if label == 'Brightness temperature (K)':
                cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            cb.set_label(label,fontsize=13,fontweight='bold',labelpad=5)
            cb.ax.tick_params(labelsize=12,width=1.4,length=5,direction='out')
            for tick in cb.ax.get_xticklabels(): tick.set_fontweight('bold')
            cb.outline.set_linewidth(1.1)
        orbit_id = f"{int(d['orbit_number']):06d}" if 'orbit_number' in d else Path(path).stem.rsplit('_', 1)[-1]
        if mode == 'PMWtoMRMS':
            if used.size:
                stamps=d['mrms_times'][used]
                start=datetime.fromtimestamp(float(stamps.min()),timezone.utc)
                end=datetime.fromtimestamp(float(stamps.max()),timezone.utc)
                end_text=end.strftime('%H:%M') if start.date()==end.date() else end.strftime('%Y-%m-%d %H:%M')
                time_text=f'MRMS: {start:%Y-%m-%d %H:%M}–{end_text} UTC ({used.size} snapshots)'
            else:
                time_text='MRMS: no valid coincident pixels'
        else:
            stamp=datetime.fromtimestamp(float(d['reference_time']),timezone.utc)
            time_text=f'MRMS reference: {stamp:%Y-%m-%d %H:%M UTC}'
        # Use actual PMW filename orbit start/end, not MRMS snapshot times.
        import re
        name=Path(str(d['pmw_file'])).name if 'pmw_file' in d else ''
        sensor=Path(path).stem.split('_')[1]
        match=re.search(r'(\d{8})-S(\d{6})-E(\d{6})',name)
        if match:
            from datetime import timedelta
            day,st,en=match.groups()
            start=datetime.strptime(day+st,'%Y%m%d%H%M%S')
            end=datetime.strptime(day+en,'%Y%m%d%H%M%S')
            if end<start: end+=timedelta(days=1)
            end_text=end.strftime('%H:%M:%S' if end.date()==start.date() else '%Y-%m-%d %H:%M:%S')
            orbit_time=f'{start:%Y-%m-%d %H:%M:%S}–{end_text} UTC'
        else:
            orbit_time='orbit time unavailable'
        for ax in axes.flat: ax.set_aspect(1.4,adjustable='box')
        fig.suptitle(f'{sensor} orbit: {orbit_id} | {orbit_time} | {mode}',fontsize=23,fontweight='bold',y=.985)
        fig.savefig(Path(path).with_suffix('.png'),dpi=300,bbox_inches='tight')
        if show: plt.show(block=True)
        plt.close(fig)
