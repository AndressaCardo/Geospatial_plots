# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:15:54 2024

@author: Andressa
"""

import os,sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import metpy as m
import metpy.calc as mpcalc
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from metpy.units import units
from scipy import stats
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeat

outin='D:/Doutorado/graphs'
dirin='D:/Doutorado/Resultados/Composites/RegCM_present/climatologia_media'

model = ['ERA5','RegEraI','RegHad','RegMPI']

cmap1 = mpl.colors.ListedColormap(['#FFFFFF00','#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04', '#783c00'])
cmap2 = mpl.colors.ListedColormap(['#5e4fa2','#3288bd', '#66c2a5', '#abdda4', '#e6f598','#ffffbf', '#fee08b', '#fdae61', '#f46d43','#d53e4f', '#9e0142'])


cleva = [0,1,2,3,4,5,6,7,8,9]
llev = [998,1000,1002,1004,1008,1010,1012,1014,1016,1018,1019,1020,1021,1022,1023]
nrows=1
ncols=4


xmax = 10.0
xmin = -xmax
D = 20
ymax = 10.0
ymin = -ymax
x = np.linspace(xmin, xmax, D)
y = np.linspace(ymin, ymax, D)
X, Y = np.meshgrid(x, y)
# plots the vector field for Y'=Y**3-3*Y-X
deg = np.arctan(Y ** 3 - 3 * Y - X)
widths = np.linspace(0, 2, X.size)

# Define the figure and each axis for the 3 rows and 3 columns
fig, ax = plt.subplots(nrows=nrows,ncols=ncols,
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         figsize=(15,4.5))

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
ax=ax.flatten()



for i in range(4):

    #open file
    modelb = model[i]
    
    ua = xr.open_dataset('{}/uas_{}_mean.nc'.format(dirin,modelb),engine='netcdf4', decode_times=False)
    va = xr.open_dataset('{}/vas_{}_mean.nc'.format(dirin,modelb),engine='netcdf4', decode_times=False)
    psl = xr.open_dataset('{}/psl_{}_mean.nc'.format(dirin,modelb),engine='netcdf4', decode_times=False)


    ua_m = ua.variables['uas'][0,:,:] 
    va_m = va.variables['vas'][0,:,:] 
    psl_m = psl.variables['psl']/100

    lat = ua.variables['latitude'] 
    lon = ua.variables['longitude']
    
    mag_m = np.sqrt((ua_m**2 + va_m**2))

    #Criando um array de valores que serão utilizados para os eixos de latitude e longitude do mapa
    xticks = np.arange(-90,5,20)
    yticks = np.arange(-60,10,10)


    #Funções do cartopy
    ax[i].coastlines(linewidth=0.5)
    ax[i].set_xticks(xticks, crs=ccrs.PlateCarree())
    ax[i].set_yticks(yticks, crs=ccrs.PlateCarree())
    ax[i].xaxis.set_major_formatter(LongitudeFormatter())
    ax[i].yaxis.set_major_formatter(LatitudeFormatter())
    #ax.grid(c='k', ls='--', alpha=0.3)

    ax[i].set_extent([-90, 5, -55, -10], crs=ccrs.PlateCarree())

    #Países
    ax[i].add_feature(cfeat.BORDERS,linewidth=0.5)

    #estados
    states_provinces = cfeat.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    ax[i].add_feature(states_provinces, edgecolor='0.25',linewidth=0.5)


    ax[i].set_title('{}'.format(modelb),fontsize=16)
    ax[i].tick_params(axis='both', labelsize=11)

    
    cs = ax[i].contourf(lon, lat, mag_m, cleva,cmap='rainbow',extend='both')

    cs1 = ax[i].contour(psl['longitude'], psl['latitude'], psl['psl'][0]/100, llev,colors='white', linewidths=1, linestyles='-')
    plt.clabel(cs1,llev,fontsize=11, inline=1, inline_spacing=4, fmt='%i',
               use_clabeltext=True,colors='k')

    # vector spacing
    vs = 10
    xx, yy = np.meshgrid(lon[::vs],lat[::vs])

    # setting font size and style
    sl=50
    mpl.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
    mpl.rc('xtick', labelsize=sl); mpl.rc('ytick', labelsize=sl)

    
    vec = ax[i].quiver(xx, yy, ua_m[::vs,::vs], va_m[::vs,::vs], units='width', linewidths=2, headwidth=6,headlength=5,scale=100) #larger scale number lead to smaller vectors
    #qk = ax[i].quiverkey(vec, 0.65, 0.86, 20, r'$20 \frac{m}{s}$', labelpos='E',coordinates='figure')

    
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.01, right=0.99,
                        wspace=0.2, hspace=0.2)
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.05])
    # Draw the colorbar
    cbar=plt.colorbar(cs,ticks=cleva,cax=cbar_ax,orientation='horizontal',pad=0.1)

    
    cbar.ax.tick_params(labelsize=14) 

    plt.savefig('{}/uv10m-psl_camposmedios-RegCM4.7.png'.format(outin),dpi=310,bbox_inches='tight')

