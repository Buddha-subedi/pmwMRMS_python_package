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

    pc = ccrs.PlateCarree()
    proj = ccrs.LambertConformal(central_longitude=-97.5,
        central_latitude=38, standard_parallels=(33,45))
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
        fig, axes = plt.subplots(2,2,figsize=(19,10.5),subplot_kw={'projection':proj})
        fig.subplots_adjust(left=.055,right=.88,bottom=.08,top=.91,wspace=.08,hspace=.18)
        def setup(ax,i,title):
            ax.set_extent([-130,-65,23,53],crs=pc)
            ax.set_facecolor(ocean)
            ax.add_feature(cf.LAND.with_scale('50m'),facecolor='#FAFAFA',zorder=0)
            ax.add_feature(cf.LAKES.with_scale('50m'),facecolor=ocean,edgecolor='#8AA5B8',linewidth=.35,zorder=1)
            ax.add_feature(cf.STATES.with_scale('50m'),edgecolor='#777777',linewidth=.4,zorder=10)
            ax.add_feature(cf.BORDERS.with_scale('50m'),linewidth=.65,zorder=11)
            ax.coastlines('50m',linewidth=.75,zorder=12)
            gl=ax.gridlines(draw_labels=True,linewidth=.3,alpha=.25,linestyle='--',x_inline=False,y_inline=False,rotate_labels=False)
            gl.xlocator=mticker.FixedLocator(np.arange(-130,-59,10)); gl.ylocator=mticker.FixedLocator(np.arange(25,56,5))
            gl.top_labels=False; gl.right_labels=False; gl.left_labels=i%2==0; gl.bottom_labels=i>=2
            gl.xlabel_style={'size':10}; gl.ylabel_style={'size':10}
            for x,y,t in [(-126.5,33.5,'Pacific\nOcean'),(-70.5,32.5,'Atlantic\nOcean'),(-90,25.5,'Gulf of Mexico')]:
                ax.text(x,y,t,transform=pc,ha='center',va='center',fontsize=10,fontstyle='italic',color='#315B80',zorder=15)
            ax.set_title(f'({chr(97+i)})',loc='left',fontweight='bold',fontsize=15)
            ax.set_title(title,fontsize=13,fontweight='bold')
            ax.spines['geo'].set_linewidth(1)
        def regular(ax,v,cmap,norm,z=3):
            # Render at native grid spacing without allocating coordinate meshes.
            dx=abs(lon[1]-lon[0])/2; dy=abs(lat[1]-lat[0])/2
            if lon[0]>lon[-1]: v=v[:,::-1]
            if lat[0]>lat[-1]: v=v[::-1]
            return ax.imshow(v,extent=[lon.min()-dx,lon.max()+dx,lat.min()-dy,lat.max()+dy],origin='lower',interpolation='nearest',transform=pc,cmap=cmap,norm=norm,zorder=z,regrid_shape=1500)
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
            axp.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(['gainsboro','red','blue'],['No precipitation','Rain','Snow'])],loc='lower left',fontsize=10,framealpha=.95)
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
                    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.12,label='Brightness temperature (K)')
                else: setup(ax,j+2,'TB unavailable')
        else: raise ValueError(f'Unknown mapping mode: {mode}')
        fig.canvas.draw()
        # Both precipitation scales apply to all precipitation-rate panels.
        for j,(cmap,norm,label,ticks) in enumerate([(rain_cmap,rn,'Rain rate (mm/hr)',np.arange(0,4.1,1)),(snow_cmap,sn,'Snow rate (mm/hr)',np.arange(0,1.01,.2))]):
            pos=axes[0,1].get_position(); cax=fig.add_axes([pos.x1+.015+j*.055,pos.y0+.1*pos.height,.009,.8*pos.height])
            cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=ticks,extend='max'); cb.set_label(label,fontsize=10)
        stamp=datetime.fromtimestamp(float(d['reference_time']),timezone.utc)
        fig.suptitle(f'MRMS reference: {stamp:%Y-%m-%d %H:%M UTC} | {mode}',fontsize=13)
        fig.savefig(Path(path).with_suffix('.png'),dpi=300,bbox_inches='tight')
        if show: plt.show(block=True)
        plt.close(fig)
