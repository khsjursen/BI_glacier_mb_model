# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:22:35 2021

Script to set up the Bayesian model and run MCMC simulations.

@author: kasj
"""
#%% Libraries

# Standard libraries
import sys
import datetime as dt
import pickle
import time

# External libraries
import arviz as az
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import pandas as pd

# Internal libraries
from set_dirs import set_dirs
from run_model import run_model
from get_observations import get_glacier_obs
from get_observations import get_hugonnet_obs

#%% Function main runs MCMC/posterior predictive

def main():
    
    main_dir = set_dirs('server')

    # Configure model.
    config_model = {"simulation_name": 'Nigardsbreen_1km', # Name of simulation case (str)
                    "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                    "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                    "simulation_start_date": '1957-01-01', # Start date (str)
                    #"simulation_start_date": '1985-01-01', # Start date of simulations (str)
                    #"simulation_end_date": '2004-12-31', # PRIOR
                    #"simulation_end_date": '2009-12-31', # End date of simulations (str)
                    "simulation_end_date": '2020-12-31', # End date (str)
                    "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                    "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                    "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                    "calculate_runoff": False, # Option to compute glacier runoff.
                    "calculate_discharge": False, # Option to compute discharge from catchments.
                    "calibration_start_date": '1960-01-01', # Start date for calibration period (str)
                    #"calibration_start_date": '1990-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '2004-12-31', # End date for calibration period (str)
                    #"calibration_end_date": '2009-12-31', # End date for calibration period (str)
                    "calibration_end_date": '2020-12-31', # End date for calibration period (str)
                    "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                    "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

    # Filepaths and filenames for files to be used in model run.
    dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/', # Path to file with glacier IDs
                      "filename_glacier_id": 'glacier_id.txt', # Name of .txt file with glacier IDs
                      "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/', # Path to file with catchment IDs
                      "filename_catchment_id": 'catchment_id.txt', # Name of .txt file with catchment IDs
                      "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                      "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                      "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                      "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                      "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                      "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                      "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                      "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                      "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                      "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                      "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                      "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                      "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                      "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                      "filename_climate_data": 'job_all_seNorge_2018_21.09', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                      "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                      "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                      "filename_dem_base": 'DEM100_' + config_model['simulation_name'][:-4] + '_EPSG32633.tif',
                      "filepath_shp": main_dir + 'shape_files/', # Filepath to glacier outline shape files
                      "filename_shp_overview": 'shp_overview.csv', # Overview of shape files used if area is updated incrementally
                      "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006_EDIT/cryoclim_GAO_NO_1999_2006_UTM_33N_EDIT.shp', # Filename of shape file with glacier IDs and outlines
                      "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of shape file with catchment IDs and outlines     

    # Get type of observations to be used in parameter estimation.
    obs_type = config_model['observation_type']

    # Start and end date for calibration.
    start_time = config_model['calibration_start_date']
    end_time = config_model['calibration_end_date']
    
    # Get start year and end year for calibration.
    calibration_year_start = dt.datetime.strptime(start_time, '%Y-%m-%d').year
    calibration_year_end = dt.datetime.strptime(end_time, '%Y-%m-%d').year
    
    # Get list of glaciers. 
    df_id = pd.read_csv(dir_file_names['filepath_glacier_id'] 
                        + dir_file_names['filename_glacier_id'], sep=';')
    id_list = df_id['BREID'].values.tolist()
            
    # Get list of calibration years. 
    yr_list = list(range(calibration_year_start, calibration_year_end + 1))
            
    # Get subsets of modeled and observed mass balance based on the list 
    # of glaciers for which observations are available and the list of 
    # calibration years.
    idx = pd.IndexSlice

    # *** USE THIS FOR GLACIOLOGICAL GLACIER-WIDE SEASONAL BALANCES ***
    if obs_type == 'glac_gw_seasonal':
        
        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # Get vectors of winter and summer balances.
        mb_obs_w = mb_obs_sub['Bw'].to_numpy(copy=True)
        mb_obs_s = mb_obs_sub['Bs'].to_numpy(copy=True)

        # Get standard deviation of seasonal balances.
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs.csv', sep=';')
        sigma_w = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_w'].values[0]
        sigma_s = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_s'].values[0]

        # If running posterior predictive, add dummy observations for years where no 
        # observations exist, e.g. 1960 to start of record. 
        if config_model['run_posterior_predictive'] == True:
            
            # Get first year of record and calculate number of years to add.
            first_obs = df_mb_obs.index[0][0]
            yrs_to_add = first_obs - calibration_year_start

            # If dummy observations are needed. 
            if yrs_to_add > 0:
                
                # Add dummy observations to record. 
                add_obs_w = np.ones((yrs_to_add,))
                add_obs_s = np.ones((yrs_to_add,))*-1
                mb_obs_w = np.hstack((add_obs_w, mb_obs_w))
                mb_obs_s = np.hstack((add_obs_s, mb_obs_s))
            
            mb_obs_a = mb_obs_w + mb_obs_s

            sigma_a = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values[0]

    # *** USE THIS FOR GLACIOLOGICAL GLACIER-WIDE ANNUAL BALANCES ***
    elif obs_type == 'glac_gw_annual':

        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # Get vector of annual balances.        
        mb_obs_a = mb_obs_sub['Ba'].to_numpy(copy=True)

        # Get standard deviation of annual balance.        
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs.csv', sep=';')
        sigma_a = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values[0]
   
    # *** USE THIS FOR GLACIER-WIDE DECADAL BALANCES ***
    elif obs_type == 'glac_gw_annual_10yr':
    
        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(config_model, dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # Get vector of annual balances.        
        mb_obs_a = mb_obs_sub['Ba'].to_numpy(copy=True)

        # Get standard deviation of annual balance.        
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs_hugonnet.csv', sep=';')
        sigma_geod = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values

        # Get 10-year rates of annual balances.
        mb_obs_10yr_rates = np.array([mb_obs_a[0:10].mean(), mb_obs_a[10:20].mean()])

    # *** USE THIS FOR GEODETIC BALANCES FROM HUGONNET ET AL. (2021) ***
    elif obs_type == 'geodetic':
    
        # Get DataFrame of observations.
        df_hugonnet_mb_obs = get_hugonnet_obs(dir_file_names)

        # Get 10-year balances.
        hugonnet_mb_obs = df_hugonnet_mb_obs['Ba'].to_numpy(copy=True)

        # Get sigma for each mass balance estimate from sigma based on 
        # Hugonnet et al. (2021).
        hugonnet_sigma = df_hugonnet_mb_obs['sigma'].to_numpy(copy=True)

    else:
        sys.exit('Observation type not found.')
    
#%% Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    def shell(p_corr, degree_day_factor_snow, temp_corr):

        main_dir = set_dirs('server')

        # Configure model. 
        # NB! Set to same as configuration above. Must be defined again in the 
        # shell function because dtype for dictionary does not exist for as_op
        config_model = {"simulation_name": 'Nigardsbreen_1km', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'.
                        #"simulation_start_date": '1985-01-01', # Start date (str)
                        #"simulation_end_date": '2009-12-31', # End date (str)
                        "simulation_start_date": '1957-01-01', # Start date (str)
                        "simulation_end_date": '2020-12-31', # End date (str)
                        "use_seNorge_dem": True,
                        "update_area_from_outline": True,
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool)
                        "calculate_runoff": False,
                        "calculate_discharge": False,
                        #"calibration_start_date": '1990-01-01', # Start date for calibration period (str)
                        #"calibration_end_date": '2009-12-31', # End date for calibration period (str)
                        "calibration_start_date": '1960-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '2020-12-31'} # End date for calibration period (str)

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
                      "temp_w_bias_corr": 0.0, # Correction for winter temperature bias [C]
                      "temp_lapse_rate": temp_m_lr, # Temperature lapse rate [C 100m-1] (negative upwards)
                      "density_water": 1000, # Density of water [kgm-3]
                      "density_ice": 850} # Density of ice [kgm-3]

        # Filepaths and filenames for files to be used in model run.
        dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/', # Path to file with glacier IDs
                          "filename_glacier_id": 'glacier_id.txt', # Name of .txt file with glacier IDs
                          "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/', # Path to file with catchment IDs
                          "filename_catchment_id": 'catchment_id.txt', # Name of .txt file with catchment IDs
                          "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to global Jostedalsbreen DEM
                          "filename_dem_base": 'DEM100_' + config_model['simulation_name'][:-4] + '_EPSG32633.tif',
                          "filepath_shp": main_dir + 'shape_files/', # Filepath to shape files
                          "filename_shp_overview": 'shp_overview.csv', # Overview of shape files used if area is updated incrementally
                          "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006_EDIT/cryoclim_GAO_NO_1999_2006_UTM_33N_EDIT.shp', # Filename of shape file with glacier IDs and outlines
                          "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of shape file with catchment IDs and outlines     

        # Run mass balance model with parameter settings.
        mb_mod = run_model(config_model, parameters, dir_file_names)

        # Get start and end time of calibration period. 
        start_time = config_model['calibration_start_date']
        end_time = config_model['calibration_end_date']
    
        # Get start year and end year for calibration.
        calibration_year_start = dt.datetime.strptime(start_time, '%Y-%m-%d').year
        calibration_year_end = dt.datetime.strptime(end_time, '%Y-%m-%d').year

        # Get list of glaciers with available observations.
        id_list = mb_mod.index.get_level_values('BREID').unique().tolist()
        
        # Make list of calibration years to crop modelled balances. 
        yr_list = list(range(calibration_year_start, calibration_year_end + 1))
            
        # Get subsets of modeled and observed mass balance based on the list 
        # of glaciers for which observations are available and the list of 
        # calibration years.
        idx = pd.IndexSlice
        mb_mod_sub = mb_mod.loc[idx[yr_list,id_list],:]
        
        # Get vectors of glacier-wide summer, winter and annual balances.
        mb_mod_winter = mb_mod_sub['Bw'].to_numpy(copy=True)
        mb_mod_summer = mb_mod_sub['Bs'].to_numpy(copy=True)
        mb_mod_annual = mb_mod_sub['Ba'].to_numpy(copy=True)

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # NB! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            calving_spec_balance = np.concatenate([np.zeros(28), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Calculate 10-year mass balance rates.
        mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        return mb_mod_winter, mb_mod_summer, mb_mod_annual, mb_mod_10yr_rates

#%% Set up PyMC model.

    # Declare model context.
    mb_model = pm.Model()
    
    # Set up model context.
    with mb_model:
        
        # Set priors for unknown model parameters.
        # Original case:
        DDF_snow = pm.TruncatedNormal("DDF_snow", mu=4.1, sigma=1.5, lower=0) 
        prec_corr = pm.TruncatedNormal("prec_corr", mu=1.25, sigma=0.8, lower=0)
        T_corr = pm.Normal("T_corr", mu=0.0, sigma=1.5)
        # High prior case:
        #DDF_snow = pm.TruncatedNormal("DDF_snow", mu=5.6, sigma=1.5, lower=0) 
        #prec_corr = pm.TruncatedNormal("prec_corr", mu=1.75, sigma=0.8, lower=0)
        #T_corr = pm.Normal("T_corr", mu=0.0, sigma=1.5)
        # Low prior case:
        #DDF_snow = pm.TruncatedNormal("DDF_snow", mu=2.6, sigma=1.5, lower=0) #low prior
        #prec_corr = pm.TruncatedNormal("prec_corr", mu=0.75, sigma=0.8, lower=0)
        #T_corr = pm.Normal("T_corr", mu=0.0, sigma=1.5)
    
        # Expected value of outcome is the modelled mass balance.
        mb_mod_w, mb_mod_s, mb_mod_a, mb_mod_geod = shell(prec_corr, DDF_snow, T_corr)

        # Observations.
        data_mb_w = pm.Data('data_mb_w', mb_obs_w) # winter glaciological
        data_mb_s = pm.Data('data_mb_s', mb_obs_s) # summer glaciological
        data_mb_a = pm.Data('data_mb_a', mb_obs_a) # annual glaciological
        #data_mb_a_10yr = pm.Data('data_mb_a_10yr', mb_obs_10yr_rates) # annual glaciological, 10-year rates

        # Expected value as deterministic RVs. Saves these to inference data file.
        mu_mb_w = pm.Deterministic('mu_mb_w', mb_mod_w) # winter
        mu_mb_s = pm.Deterministic('mu_mb_s', mb_mod_s) # summer
        mu_mb_a = pm.Deterministic('mu_mb_a', mb_mod_a) # annual
        mu_mb_geod = pm.Deterministic('mu_mb_geod', mb_mod_geod) # 10-year rates

        # Degrees of freedom of Student-t distribution.
        #dof = 4

        # Scale parameters for Student-t distribution.
        #scale_mb_s = np.sqrt((((sigma_s**2)*3)*(dof-2))/dof)
        #scale_mb_w = np.sqrt((((sigma_w**2)*3)*(dof-2))/dof)
        #scale_mb_a = np.sqrt((((sigma_a**2)*3)*(dof-2))/dof)
        
        # Likelihood (sampling distribution) of observations.
        mb_obs_winter = pm.Normal("mb_obs_winter", mu=mu_mb_w, sigma=sigma_w, observed=data_mb_w)
        mb_obs_summer = pm.Normal("mb_obs_summer", mu=mu_mb_s, sigma=sigma_s, observed=data_mb_s)
        mb_obs_annual = pm.Normal("mb_obs_annual", mu=mu_mb_a, sigma=sigma_a, observed=data_mb_a)
        #mb_obs_annual_rates = pm.Normal("mb_obs_annual_rates", mu=mu_mb_geod, sigma=sigma_geod, observed=data_mb_a_10yr)
        #mb_obs_winter = pm.StudentT("mb_obs_winter", mu=mu_mb_w, nu=dof, sigma=scale_mb_w, observed=data_mb_w)#observed=mb_obs_w)#observed=data_mb_w)
        #mb_obs_summer = pm.StudentT("mb_obs_summer", mu=mu_mb_s, nu=dof, sigma=scale_mb_s, observed=data_mb_s)#observed=mb_obs_s)#observed=data_mb_s)
        #mb_obs_annual = pm.StudentT("mb_obs_annual", mu=mu_mb_a, nu=dof, sigma=scale_mb_a, observed=data_mb_a)#observed=mb_obs_w)#observed=data_mb_w)

    # Use this to sample posterior.
    #with mb_model:
        
        # Choose sampler.
        #step = pm.Metropolis()
    #    step = pm.DEMetropolisZ()

        # Draw posterior samples.
    #    idata_post = pm.sample(draws=10000, tune=2000, step=step, return_inferencedata=True, chains=4, cores=20, progressbar=True, idata_kwargs=dict(log_likelihood=False))
        
    # Save InferenceData with posteriors to netcdf file.
    #idata_post.to_netcdf(main_dir + 'simulation_data/Nigardsbreen_1km/results/3_param/idata_DEMZ_t10000_s10000_c4_gwa_stud-t_dof4_scale3var_new.nc')

    # Use this to sample prior.
    # with mb_model:

    #     # Sample prior for given variables.
    #     start_time = time.time()
    #     prior = pm.sample_prior_predictive(10000)
    #     print("Prior predictive sampling took " + str (time.time() - start_time) + ' seconds.')

    #     # Save prior data in InferenceData object.
    #     idata_prior = az.from_pymc3(prior=prior)

    # # Save InferenceData with prior to netcdf file.
    # idata_prior.to_netcdf(main_dir + 'simulation_data/Hansebreen_1km/results/3_param/idata_DEMZ_s10000_norm_prior_pred.nc')

    # Use this to sample posterior predictive.
    with mb_model:

        # Get inference data.
        idata_sampled = az.from_netcdf(main_dir + 'simulation_data/Nigardsbreen_1km/results/3_param/idata_DEMZ_t10000_s10000_c4_gwa_stud-t_dof4_scale3var.nc')

        print("Starting posterior predictive sampling.")
         # Sample posterior predictive from trace.
        start_time = time.time()
        fast_post_pred = pm.fast_sample_posterior_predictive(trace=idata_sampled, samples=10000)
        print("Posterior predictive sampling took " + str(time.time() - start_time) + ' seconds.')

        # Save posterior predicitve data in InferenceData object.
        idata_post_pred = az.from_pymc3(posterior_predictive=fast_post_pred)

    # # Save InferenceData with posterior predictive to netcdf file.
    idata_post_pred.to_netcdf(main_dir + 'simulation_data/Nigardsbreen_1km/results/3_param/idata_DEMZ_s10000_gwa_stud-t_dof4_scale3var_post_pred.nc')

if __name__ == '__main__':
    main()
