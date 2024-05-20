# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:12:55 2024

@author: Andressa
"""

import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import xarray as xr

import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from math import sqrt
import scipy as sc
from scipy import signal, misc
import scipy.ndimage
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats


lon1 = -93 #lon inicial
lon2 = 0.5     #lon final
lat1 = -60     #lat inicial
lat2 = -5      #lat final
dlon = 3 #espaçamento grade
dlat = 3
lats = np.arange(lat1,lat2,dlat) #criando array #lat ficticia
lons = np.arange(lon1,lon2,dlon) #criando array #lon ficticia
nlat = lats.size
nlon = lons.size

#Definindo uma função onde será aplicado o teste para todo o globo entre dois conjuntos de dados: ds1 e ds2
def composite_significante(ds1, ds2, equal_var=False):
    """
    Calculate composite significance using t test
    """
    mean1 = ds1.mean("time") #fazendo a média no tempo para todos os pontos de grade do conjunto ds1
    std1 = ds1.std("time") #desvio padrão de ds1
    nobs1 = len(ds1) #extraindo o número de observações da série ds1
    mean2 = ds2.mean("time") #fazendo a média no tempo para todos os pontos de grade do conjunto ds2
    std2 = ds2.std("time") #desvio padrão de ds2
    nobs2 = len(ds2) #extraindo o número de observações da série ds2

 #Aplicando a função do teste estatístico utilizando o comando ufunc. 
 #Nesse caso, a função do teste será mapeada sobre todas as variáveis e coordenadas dos arquivos.
    return xr.apply_ufunc(
        stats.ttest_ind_from_stats,
        mean1,
        std1,
        nobs1,
        mean2,
        std2,
        nobs2,
        equal_var,
        input_core_dims=[[], [], [], [], [], [], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
    )

### atribuindo 
### atribuindo 
def density_ciclones(df):
    a = 6370 #raio da terra
    a2 = a * a
    lon1 = -93 #lon inicial
    lon2 = 0.5     #lon final
    lat1 = -60     #lat inicial
    lat2 = -5      #lat final
    dlon = 3 #espaçamento grade
    dlat = 3
    lats = np.arange(lat1,lat2,dlat) #criando array #lat ficticia
    lons = np.arange(lon1,lon2,dlon) #criando array #lon ficticia
    nlat = lats.size
    nlon = lons.size
    entrada = np.array(df)
    nt=entrada[:,1].size
    
    rlats = lats * np.pi / 180         #latitudes em radianos
    drlon = dlon * np.pi / 180    
    
    den = np.zeros((nlon,nlat))
    
    for t in range(nt):
         y0=entrada[t,1]
         y=np.absolute(lats-y0)
         latloc=np.argmin(y)
         x0 = entrada[t,2]
         x=np.absolute(lons-x0)
         lonloc=np.argmin(x)
         for j in range(nlat):
             for i in range(nlon):
                 if(i==lonloc and j==latloc):
                     den[i,j]= den[i,j] + 1
                     
    aread = drlon * a2 * (np.sin(rlats[1:nlat-1]) - np.sin(rlats[0:nlat-2]))
                     
       
    for k in range(nlat-2):
        den[:,k] = (den[:,k]/aread[k])
     
    den = xr.DataArray(den, dims=['lon','lat'],
                            coords={'lat': lats,'lon': lons})                   
        
    return den
    
    

outin='C:/Users/Andressa/Documents/Doutorado/Artigos-myself/artigo_CMIP6/graphs'
dirin='C:/Users/Andressa/Documents/Doutorado/Artigos-myself/artigo_CMIP6/rastreios'

model = ['ERA5uv10m','NICAM8S-CMIP6','EC-EarthHR-CMIP6','MPI-CMIP6','HadGEM3HM-CMIP6','MRI-CMIP6','CMCC-CMIP6']

model1 = ['CFSRuv10m','NICAM7S-CMIP6','EC-Earth3P-CMIP6','MPIHR-CMIP6','HadGEM3MM-CMIP6','MRIH-CMIP6','CMCCHR4-CMIP6']

model2 = ['ERA5 - CFSR','NICAM-HR - NICAM-MR', 'ECEarth-HR - ECEarth-MR','MPI-HR - MPI-MR','Had-HR - Had-MR','MRI-HR - MRI-MR','CMCC-HR - CMCC-MR']

colorb=np.round([-4,-2,-1,-0.5,-0.2,0.2,0.5,1,2,4],decimals=1)
#colorb=[0.5,1,2,3,4,6,8,10,12,14]

anos = np.arange(1979,2015,1)

nrows=7
ncols=1

correla = np.zeros((7), dtype=np.float64)
rmse = np.zeros((7), dtype=np.float64)

       

dens = np.zeros((36,nlon,nlat))
dens1 = np.zeros((36,nlon,nlat))

# Define the figure and each axis for the 3 rows and 3 columns
fig, ax = plt.subplots(nrows=nrows,ncols=ncols,
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         figsize=(5.5,28))

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
ax=ax.flatten()

for i in range (7):
   
    modeloa=model[i]
    modelob=model1[i]
    modeloc=model2[i]
    # -- open msl file
    df = pd.read_csv('{}/genese{}_1979-2014.txt'.format(dirin,modeloa),sep=" ")
    df1 = pd.read_csv('{}/genese{}_1979-2014.txt'.format(dirin,modelob),sep=" ")
    df['data'] = pd.to_datetime(df['data'], format='%Y%m%d%H',errors='coerce')
    df1['data'] = pd.to_datetime(df1['data'], format='%Y%m%d%H',errors='coerce')
    df = df[(df['data'].dt.year >= 1979) & (df['data'].dt.year <= 2014)]
    df1 = df1[(df1['data'].dt.year >= 1979) & (df1['data'].dt.year <= 2014)]
    
    for j in range(36):
        ano = anos[j]
        
        df_new = df[(df['data'].dt.year == ano)]
        dens[j,:,:] = (density_ciclones(df_new))*1e5
                 
        df1_new = df1[(df1['data'].dt.year == ano)]
        dens1[j,:,:] = (density_ciclones(df1_new))*1e5
        
    
    
    dens_new = xr.DataArray(dens, dims=['time','lon','lat'],
                            coords={'time':anos,'lat': lats,'lon': lons})
    
    dens1_new = xr.DataArray(dens1, dims=['time','lon','lat'],
                             coords={'time':anos,'lat': lats,'lon': lons})
    


    dens1_mean  = dens1_new.mean("time") 
    dens_mean  = dens_new.mean("time")

    dif_t = (dens_mean)-(dens1_mean)

    std_1 = dens_new.std("time") #desvio padrão de ds1
    std_2 = dens1_new.std("time") #desvio padrão de ds2


    r1 = ((std_1)**2)/36
    r2 = ((std_2)**2)/36
    r = np.sqrt(r1+r2)

    ttest = (dif_t)/r


    densT = xr.DataArray(ttest, dims=['lon','lat'],
                                   coords={'lon': lons,'lat': lats})



    #Criando um array de valores que serão utilizados para os eixos de latitude e longitude do mapa
    #Criando um array de valores que serão utilizados para os eixos de latitude e longitude do mapa
    xticks = np.arange(-80,0,20)
    yticks = np.arange(-60,10,10)


    #Funções do cartopy
    ax[i].coastlines()
    ax[i].set_xticks(xticks, crs=ccrs.PlateCarree())
    ax[i].set_yticks(yticks, crs=ccrs.PlateCarree())
    ax[i].xaxis.set_major_formatter(LongitudeFormatter())
    ax[i].yaxis.set_major_formatter(LatitudeFormatter())
    ax[i].grid(c='k', ls='--', alpha=0.3)

    ax[i].set_extent([-75, -20, -55, -10], crs=ccrs.PlateCarree())

    #Países
    ax[i].add_feature(cfeat.BORDERS)



    cf = ax[i].contourf(scipy.ndimage.zoom(dif_t.lon,3),scipy.ndimage.zoom(dif_t.lat,3),scipy.ndimage.zoom(dif_t.values,3).T,colorb,transform=ccrs.PlateCarree(),extend='both', cmap='bwr')

    cs = ax[i].contourf(densT.lon,densT.lat,densT.values.T,colors='none', hatches=['....'],levels=np.array([-17,-1.96]),extend='lower',transform=ccrs.PlateCarree())
    cs = ax[i].contourf(densT.lon,densT.lat,densT.values.T,colors='none', hatches=['....'],levels=np.array([1.96,17]),extend='lower',transform=ccrs.PlateCarree())


    rmse = float(np.round(sqrt(mean_squared_error(dens_mean,dens1_mean)),2))
    correla = float(np.round(xr.corr(dens_mean,dens1_mean),2))
    
    textoR = 'rmse:{}'.format(rmse)
    textoC = 'r:{}'.format(correla)

    #ax[i].set_xlabel('Longitude',fontsize=14)
    #ax[i].set_ylabel('Latitude',fontsize=14)
    ax[i].set_title('{} '.format(modeloc),fontsize=18)
    ax[i].tick_params(axis='both', labelsize=14)
    #ax[i].text(-75, -16, letrab,style='normal', fontsize=24)
    ax[i].text(-36, -14, textoC,style='normal', fontsize=11)
    ax[i].text(-39, -18, textoR,style='normal', fontsize=11)

    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.23, top=0.9, left=0.1, right=0.90,
                        wspace=0.20, hspace=0.5)
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.1, 0.2, 0.8, 0.008])
    
    #0.85, 0.15, 0.05, 0.7
    #0.1, 0.2, 0.8, 0.009
    #[left, bottom, width, height] of the new axes.
    
    # Draw the colorbar
    cbar=plt.colorbar(cf,ticks=colorb,cax=cbar_ax,orientation='horizontal',pad=0.09,shrink=0.1,aspect=40)

    
    cbar.ax.tick_params(labelsize=12) 

    plt.savefig("{}/Densidade-genese_DIFsimulation3s_1979-2014_new_sig.png".format(outin),dpi=310,bbox_inches='tight')










