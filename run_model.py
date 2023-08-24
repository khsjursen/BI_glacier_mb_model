# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:28:56 2020

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Script to set up and run model simulations.

Function initialize_input() must be run first to set up DEMs and masks.

"""
    
#%% Libraries %%#

# Standard libraries
import pickle

# External libraries
import xarray as xr

# Internal libraries
from set_dirs import set_dirs
from preprocessing import initialize_dem
from preprocessing import catchment_specific_fraction
from mb_model import mass_balance

#%% Set directory

# Set main directory to local or server.
main_dir = set_dirs('server')

#%% Set model configuration

# Set configuration for model run.
# If 'get_catchment_discharge' is True, the model simulates the discharge from the catchment IDs given in the 'catchment_id' file. 
# If 'get_catchment_discharge' is False, the model does not use the outline of the catchment IDs given in the 'catchment_id' file,
# but instead run simulations based on a 'dummy' catchment created for the geographical area with a margin around the (merged) outline(s) 
# of the glacier IDs given in the 'glacier_id' file. 
config_model = {"simulation_name": 'Aalfotbreen_1km',
                "filepath_simulation_files": main_dir + 'simulation_data/', # Filepath to simulation files. 
                "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'.
                "simulation_start_date": '1957-01-01', # Start date of simulation (str)
                "simulation_end_date": '2020-12-31', # End date of simulation (str)
                "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool)
                "calculate_runoff": True, # Option to compute glacier runoff.
                "calculate_discharge": False, # Option to compute discharge from catchments.
                "calibration_start_date": '1960-01-01', # Start date for calibration period (str)
                "calibration_end_date": '1999-12-31'} # End date for calibration period (str)

#%% Declare parameter values

# Get temperature lapse rates from file.
with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
    temp_m_lr = pickle.load(fp)

# Set parameters for mass balance and discharge simulations.
parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
              "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
              "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
              "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
              "melt_factor": 3.5, # Melt factor (not used)
              "melt_factor_snow": 3.9, # Melt factor for snow (mm w.e. degC d-1)
              "melt_factor_ice": (3.9/0.7), # Melt factor for ice (mm w.e. degC d-1)
              "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
              "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
              "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
              "prec_corr_factor": 1.317, # Global precipitation correction [-]
              "prec_lapse_rate": 0.0, # Precipitation lapse rate [100m-1] (positive upwards)
              "temp_bias_corr": 0.0, # Correction for temperature bias [C]
              "temp_w_bias_corr": 0.0, # Correction for winter temperature bias [C]
              "temp_lapse_rate": temp_m_lr,#-0.55 # Temperature lapse rate [C 100m-1] (negative upwards)
              "density_water": 1000, # Density of water [kgm-3]
              "density_ice": 850} # Density of ice [kgm-3]

#%% Declare filepaths

# Filepaths and filenames for files to be used in model run.
dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/',
                  "filename_glacier_id": 'glacier_id.txt',
                  "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/',
                  "filename_catchment_id": 'catchment_id.txt',
                  "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                  "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                  "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                  "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc',
                  "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                  "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/',
                  "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/',
                  "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                  "filename_climate_data": 'svelgen',
                  "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                  "filepath_dem_base": main_dir + 'dem_base/', # Filepath to base DEM to use if not seNorge DEM
                  'filename_dem_base': 'DEM100_' + config_model['simulation_name'][:-4] + '_EPSG32633.tif',
                  "filepath_shp": main_dir + 'shape_files/', # Filepath to shape files
                  "filename_shp_overview": 'shp_overview.csv',
                  "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006_EDIT/cryoclim_GAO_NO_1999_2006_UTM_33N_EDIT.shp', # Filename of glacier shape file
                  "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of catchment shape file 

#%% Function initialize_input()

def initialize_input(config: dict, param: dict, name_dir_files: dict):
    """
    Initialize model DEMs, glacier masks, etc. 
    
    """
    
    if config['use_seNorge_dem']==True:
        # Preprocessing of 1km seNorge DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs. 
        ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=True)
    else:
        # Preprocessing of 1km seNorge DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs.
        name_dir_files['filename_high_res_dem'] = name_dir_files['filename_dem']
        name_dir_files['filename_high_res_gl_frac'] = name_dir_files['filename_gl_frac']
        
        ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=False)
        
    # Get DataArray of catchment fraction for each catchment ID.
    da_ca_spec_frac = catchment_specific_fraction(ds_dem, config, name_dir_files)

# End of function initialize_input()

#%% Function run_model()

def run_model(config: dict, param: dict, name_dir_files: dict):
    
    """
    Function run_model() runs mass balance simulations. Model simulations are
    run with the parameters specified in param dictionary.

    Parameters
    ----------
    config : dict
        Dictionary of model configuration.
    param : dict
        Dictionary of model parameters.
    name_dir_files: dict
        Dictionary containing names of shape files and directory.

    Returns
    -------
    mb_mod : Multiindex pandas.Dataframe 
        DataFrame of modeled winter, summer and annual mass balance for each
        glacier and year of simulation.
    """

    # Get DEM.
    with xr.open_dataset(name_dir_files['filepath_dem'] 
                         + name_dir_files['filename_dem']) as ds_dem_out:
        ds_dem = ds_dem_out        
        
    # Get DataArray with glacier fraction for each glacier ID. 
    with xr.open_dataset(name_dir_files['filepath_fractions'] 
                         + name_dir_files['filename_gl_frac']) as ds_gl_spec_frac:
        da_gl_spec_frac = ds_gl_spec_frac.glacier_specific_fraction
                
    # Get DataArray of catchment fraction for each catchment ID.
    with xr.open_dataset(name_dir_files['filepath_fractions'] 
                         + name_dir_files['filename_ca_frac']) as ds_ca_spec_frac:
        da_ca_spec_frac = ds_ca_spec_frac.catchment_specific_fraction
         
    # Run mass balance model.
    mb_mod = mass_balance(ds_dem, da_gl_spec_frac, da_ca_spec_frac, config, param, name_dir_files)
        
    return mb_mod

# End of function run_model()

#%% End of run_model.py