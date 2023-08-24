# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:34:11 2021

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Functions for retreiving seNorge climate data from thredds.met.no and 
correction of precipitation and temperature data from the seNorge model
elevation to the surface elevation of the glacier.

"""

#%% Libraries

# Standard libraries
import os

# External libraries
import numpy as np
import xarray as xr

# Internal libraries

#%% Function get_climate()

def get_climate(Y_coordinates, X_coordinates, mask_ca_1, year: str, 
                name_dir_files: dict, return_model_h=False):
    
    """
    The function get_climate() returns numpy arrays of raw temperature
    and precipitation data from seNorge for the given year. 
    
    If climate data for the specific area (given by glacier name) and year is
    available as local netcdf files, the file is loaded as a Dataset and 
    temperature and precipitation is stored in .npy files. This is done for 
    faster loading during modelling.
    
    If the data is not available as a local netCDF file, the netCDF file 
    containing the seNorge_2018 dataset for the particular year is read from 
    thredds.met.no as an xarray Dataset and saved locally before the data is
    retrieved as .npy files.
    
    Parameters
    ----------
    Y_coordinates : numpy.array
        Y-coordinate of cell centers.
    X_coordinates : numpy.array
        X-coordinate of cell centers.
    mask_ca_1 : numpy.array
        Mask with ones indicating the area to be selected.
    year : str
        Year to extract climate data for.
    name_dir_files : dict
        Dictionary with name of directory to store and retrieve local netCdf
        climate files.
    return_model_h : bool
        Option to return the seNorge model elevation as a numpy array.

    Saves
    -------
    ds_clim_cropped : xarray.Dataset
        Dataset of climate data the area and year containing coordinates and 
        data variables:
        X : float
            X-coordinate of cell centers.
        Y : float
            Y-coordinate of cell centers.
        time : datetime64[ns]
            Timestamp for each date in the year.
        tg (time,Y,X) : xarray DataArray
            Mean daily temperature in each cell.
        rr (time,Y,X) : xarray DataArray
            Total daily precipitation in each cell.
        tx (time,Y,X) : xarray DataArray
            Maximum daily temperature in each cell.
        tn (time,Y,X) : xarray DataArray
            Minimum daily temperature in each cell.
        model_height (Y,X) : xarray DataArray
            DEM of seNorge model. 
    temp_raw_np : numpy.array (Y,X)
        Array of seNorge temperature (variable tg) for the area given by
        Y_coordinates and X_coordinates.
    prec_raw_np : numpy.array (Y,X)
        Array of seNorge precipitation (variable rr) for the area given by
        Y_coordinates and X_coordinates.    
        
    Returns
    -------
    temp_raw_np : numpy.array (Y,X)
        Array of seNorge temperature (variable tg) for the area given by
        Y_coordinates and X_coordinates.
    prec_raw_np : numpy.array (Y,X)
        Array of seNorge precipitation (variable rr) for the area given by
        Y_coordinates and X_coordinates.
    model_h : numpy.array (Y,X)
        Optionally return array of seNorge model elevation.
    """

    # Get filepath & filename.
    
    # Filepath to file storage.
    filepath_clim_data = name_dir_files['filepath_climate_data']
    
    # Name of netCdf file to load or store.
    filename = name_dir_files['filename_climate_data'] + '_' + year + '.nc'
    
    # Name of numpy file with temperature to load or store.
    filename_temp = name_dir_files['filepath_temp_prec_raw'] + 'temp_sn_' + year + '.npy'
    
    # Name of numpy file with precipitation to load or store.
    filename_prec = name_dir_files['filepath_temp_prec_raw'] + 'prec_sn_' + year + '.npy'

    # Name of numpy file with elevation of seNorge model.
    filename_elev = name_dir_files['filepath_temp_prec_raw'] + 'elev_sn.npy'

    # Name of filepath of seNorge model DEM.
    filepath_dem = 'https://thredds.met.no/thredds/dodsC/senorge/geoinfo/seNorge2018_dem_UTM33.nc'

    # Load (and save) climate Dataset.
    
    # While the filename does not exist in filepath, get the seNorge_2018 
    # dataset for the specified year from thredds.met.no, extract the data 
    # inside coordinates of the center of the corner cells (with buffer) and 
    # save as filename in filepath.
    while not os.path.isfile(filepath_clim_data + filename):
        
        # Filepath to seNorge_2018 dataset on thredds.met.no.
        filepath= ('https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Archive/seNorge2018_' 
                   + year +'.nc')
        
        # Get the cell center coordinates of the north, south, west and east 
        # corners cell of the area from ds_geo. These coordinates are the 
        # first and last entries in X and Y. 
        bound_n = Y_coordinates.astype(int)[0]
        bound_s = Y_coordinates.astype(int)[(len(Y_coordinates)-1)]
        bound_w = X_coordinates.astype(int)[0]
        bound_e = X_coordinates.astype(int)[(len(X_coordinates)-1)]
    
        # Read the seNorge_2018 file from thredds.met.no into a Dataset.
        with xr.open_dataset(filepath_dem, engine='netcdf4') as ds_model_dem:
            
            # Rename coordinates of 'elevation' variable in DataSet to 
            # correspond to climate data coordinates.
            ds_model_dem = ds_model_dem.rename({'easting':'X','northing':'Y',
                                                'elevation':'model_height'})
            
            # Get the cellsize of the climate data.
            cellsize = ds_model_dem.X[1].values - ds_model_dem.X[0].values
            
            # Extract a cropped Dataset by the bounding box coordinates. Add 
            # an extra buffer of cellsize/2 to the coordinates. The method
            # da.sel(x=slice(x1,x2)) choses all inclusive cells (not nearest),
            # so the buffer is needed to ensure that all relevant climate 
            # cells are included. 
            ds_dem_cropped = ds_model_dem.sel(X=slice(bound_w - cellsize/2, 
                                                      bound_e + cellsize/2),
                                              Y=slice(bound_n + cellsize/2, 
                                                      bound_s - cellsize/2))
            # Close files.
            ds_model_dem.close()
        
        # Read the seNorge_2018 file from thredds.met.no into a Dataset.
        with xr.open_dataset(filepath, engine='netcdf4') as ds_clim:
            
            # Extract a cropped Dataset by the bounding box coordinates. Add 
            # an extra buffer of cellsize/2 to the coordinates. The method
            # da.sel(x=slice(x1,x2)) choses all inclusive cells (not nearest),
            # so the buffer is needed to ensure that all relevant climate 
            # cells are included. 
            ds_clim_cropped = ds_clim.sel(X=slice(bound_w - cellsize/2, 
                                                  bound_e + cellsize/2),
                                          Y=slice(bound_n + cellsize/2, 
                                                  bound_s - cellsize/2))
            
            ds_clim_cropped = xr.merge([ds_clim_cropped, ds_dem_cropped])
            
            # Save as local netCdf file.
            ds_clim_cropped.to_netcdf(filepath_clim_data + filename)

            # Close files.
            ds_clim_cropped.close()
            ds_clim.close()
            ds_model_dem.close()
    
    # If the .npy files of raw temperature and precipitation for the given
    # area and resolution are not on disk, get temperature and precipitation
    # from the netCDF climate files and save as .npy files in the given paths.
    # If the model resolution is higher than the seNorge resolution (1x1 km)
    # the nearest cell climate data is selected for a given model cell. 
    while not os.path.isfile(filename_temp):
    
        # Get Dataset from filepath and return.   
        with xr.open_dataset(filepath_clim_data + filename) as ds_clim_cropped:
        
            ds_clim_cropped.load()
        
            # Get temperature data for the entire year, resampled to DEM grid.
            da_temp_raw = ds_clim_cropped.tg.sel(X=X_coordinates,
                                                 Y=Y_coordinates,
                                                 method = "nearest")
            
            # Get precipitation data for the entire year, resampled to DEM
            # grid.
            da_prec_raw = ds_clim_cropped.rr.sel(X=X_coordinates,
                                                 Y=Y_coordinates,
                                                 method = "nearest")
        
            # Convert seNorge temperature data to numpy array.
            temp_raw_np = np.array(da_temp_raw.values)
            
            # Array of temperature masked with cells part of catchment. 
            # All other cells are nan. 
            temp_masked = np.multiply(temp_raw_np, mask_ca_1)
        
            # Create vector of temperatures with nans removed.
            temp_raw_vec = temp_masked[~np.isnan(temp_masked)]
            
            # Get the number of days of the given year (rows) and number of cells in
            # the catchment mask (columns).
            n_rows = temp_masked.shape[0]
            n_cols = int(np.nansum(mask_ca_1))
            
            # Reshape temperature vector to matrix with size n_rows, n_cols.
            temp_raw = np.reshape(temp_raw_vec, (n_rows,n_cols))
            
            # Save raw temperature data as .npy file.
            with open(filename_temp, 'wb') as f_t:
                np.save(f_t, temp_raw, allow_pickle=True)
                f_t.close()
        
            # Convert seNorge precipitation data to numpy array.
            prec_raw_np = np.array(da_prec_raw.values) 
            
            # Array of precipitation masked with cells part of catchment. All other
            # cells are nan. 
            prec_masked = np.multiply(prec_raw_np, mask_ca_1)
        
            # Create vector of precipitation with nans removed.
            prec_raw_vec = prec_masked[~np.isnan(prec_masked)]
            
            # Reshape precipitation vector to matrix with size n_rows, n_cols.
            prec_raw = np.reshape(prec_raw_vec, (n_rows,n_cols))
            
            # Save raw precipitation data as .npy file.
            with open(filename_prec, 'wb') as f_p:
                np.save(f_p, prec_raw, allow_pickle=True)
                f_p.close()
        
            # Return the (resampled) elevation of the seNorge model
            # for later elevation correction of raw temperature and 
            # precipitation to model elevation.
            while not os.path.isfile(filename_elev):
                model_h = np.array(ds_clim_cropped.model_height.sel(X=X_coordinates, 
                                                                    Y=Y_coordinates,
                                                                    method = "nearest").values)
                
                # Array of precipitation masked with cells part of catchment. All other
                # cells are nan. 
                model_h_masked = np.multiply(model_h, mask_ca_1)
        
                # Create vector of precipitation with nans removed.
                model_h_vec = model_h_masked[~np.isnan(model_h_masked)]
                
                # Save elevation data as .npy file.
                with open(filename_elev, 'wb') as f_e:
                    np.save(f_e, model_h_vec, allow_pickle=True)
                    f_e.close()                                                    
            
            ds_clim_cropped.close()
        
    # Get raw temperature data from .npy file.     
    with open(filename_temp, 'rb') as f_t_ret:
        temp_raw_ret = np.load(f_t_ret)
    
    # Get raw precipitation data form .npy file.    
    with open(filename_prec, 'rb') as f_p_ret:
        prec_raw_ret = np.load(f_p_ret)
        
        # If return_model_h is True, return seNorge model elevation,
        # otherwise return only temperature and precipitation.
        # The seNorge model elevation should only be returned in the 
        # first year of simulation.
        if return_model_h == True:
            
            # Get elevation data from .npy file.
            with open(filename_elev, 'rb') as f_e_ret:
                model_h_ret = np.load(f_e_ret)
                                                              
            return(temp_raw_ret, prec_raw_ret, model_h_ret)
       
        else:
            
            return(temp_raw_ret, prec_raw_ret, None)
        
# End of function get_climate()

#%% Function adjust_temp()

def adjust_temp(seNorge_temp, dem_elev, seNorge_elev, temp_bias, temp_w_bias, temp_lr):

    """
    The function adjust_temp() adjusts the seNorge temperature to the elevation
    of the model DEM based on the elevation (model height) of the seNorge 
    model and a temperature lapse rate.
    
    The temperature is adjusted from the seNorge elevation to the elevation of 
    the DEM cell using a temperature bias correction and a temperature lapse rate [C(100m)-1]. 
    The function returns an array with the temperature for a given year in each of the DEM 
    cells.

    Parameters
    ----------
    seNorge_temp : np.array (n_days, (Y*X))
        Numpy array of temperature data for the area for a given year.
    dem_elev : np.array (Y*X,)
        Array with elevation of mass balance model grid.
    seNorge_elev : np.array (Y*X,)
        Array with elevation of seNorge model data. 
    temp_bias : float
        Correction of temperature bias in seNorge data [C].
    temp_lr : list (12)
        Monthly temperature lapse rate [C/100m], negative upwards.

    Returns
    -------
    temp_adj : numpy.array (n_days, (Y*X))
        Array of temperature data for the area based on seNorge temperature
        adjusted to DEM from seNorge model height.
    """    
    
    # Get the number of days in each month in the current year based on the total number of days. 
    if seNorge_temp.shape[0]==366:
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # If leap year
        jan_feb = 31 + 29
        dec = 31
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        jan_feb = 31 + 28
        dec = 31

    # Make array of lapse rate for each day given the month number.
    mask_lr = np.repeat(temp_lr, days_in_month).reshape(seNorge_temp.shape[0])
    mask_lr = np.expand_dims(mask_lr, axis=(1))

    # Elevation difference between DEM and climate model DEM.
    dem_diff = np.subtract(dem_elev, seNorge_elev)
    dem_diff = np.expand_dims(dem_diff, axis=(0))
 
    # Adjust seNorge model temperature to DEM with temperature lapse rate 
    # [C(100m)-1] and elevation difference between DEM and seNorge model.
    temp_adj = seNorge_temp + temp_bias + np.multiply(mask_lr, dem_diff) / 100
    
    # With winter temperature bias correction.
    temp_adj[0:jan_feb,:] = temp_adj[0:jan_feb,:] + temp_w_bias
    temp_adj[-dec,:] = temp_adj[-dec,:] + temp_w_bias
    
    return(temp_adj)

# End of function adjust_temp()

#%% Function adjust_prec()

def adjust_prec(seNorge_prec, dem_elev, seNorge_elev, prec_cf, prec_lr):

    """
    The function adjust_prec() adjusts the seNorge precipitation to the 
    elevation of the model DEM based on the elevation (model height) of the 
    seNorge model and a precipitation lapse rate. A global correction of the
    precipitation values can also be applied.
    
    A global precipitation correction in the form of a precipitation 
    correction factor is applied. The precipitation is adjusted from the 
    seNorge elevation to the elevation of the DEM cell using a precipitation 
    lapse rate [(-)(100m)-1]. The function returns an array with the 
    precipitation for the given day in each of the DEM cells.

    Parameters
    ----------
    seNorge_prec : np.array (Y,X)
        Numpy array of precipitation data for the area on the given day.
    dem_elev : np.array (Y,X)
        Array with elevation of mass balance model grid.
    seNorge_elev : numpy.array
        Elevation of seNorge data (model height).
    prec_cf : float
        Global precipitation correction factor [-].
    prec_lr : float
        Precipitation lapse rate [100m-1], positive upwards.

    Returns
    -------
    prec_adj : numpy.array (Y,X)
        Array of precipitation data for the area based on seNorge 
        precipitation adjusted to DEM from seNorge model height.
    """    
    
    # Get the elevation difference between DEM and climate model DEM.
    dem_diff = np.subtract(dem_elev, seNorge_elev)
 
    # Apply global precipitation correction to seNorge precipitation.
    seNorge_prec = np.multiply(seNorge_prec, prec_cf)
 
    # Adjust seNorge precipitaton to DEM with precipitation lapse rate 
    # [(-)(100m)-1] and elevation difference between DEM and seNorge model.
    prec_adj = seNorge_prec + np.multiply(
        (np.multiply(prec_lr, dem_diff) / 100), seNorge_prec)
 
    # Return precipitation on the given day after correction.
    return(prec_adj)

# End of function adjust_prec()

#%% End of get_climate.py
