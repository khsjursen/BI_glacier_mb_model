# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:48:18 2021

@author: kasj

Functions to get mass balance observations from file.

"""

#%% Libraries

# Standard libraries

# External libraries
import pandas as pd

# Internal libraries

#%% Get observations from Hugonnet et al. (2021) from file

def get_hugonnet_obs(name_dir_files):
    
    """
    Function get_hugonnet_obs() gets estimates of mass balance for the periods
    2000-2009 and 2010-2019 from Hugonnet et al. (2021) from .csv files. 
    
    Glaciers IDs are given by .txt file 'filename_glacier_id'.
    
    Mass balance observations are read from local files. These 
    files must be stored in the 'filepath_obs' folder. 
    
    Parameters
    ----------
    name_dir_files : dict
        Dictionary with names of files and directories. 

    Returns
    -------
    mb_obs_yearly : pandas.DataFrame
        DataFrame of mass balance.
    """

    # Filepath to observations.
    filepath_obs = name_dir_files['filepath_obs']

    # Filename of observations.
    filename_obs = 'dh_08_rgi60_pergla_rates.csv'

    # Load data
    all_gl_data = pd.read_csv(filepath_obs + filename_obs, sep=',')

    # Get list of glacier IDs.
    df_id = pd.read_csv(name_dir_files['filepath_glacier_id'] 
                        + name_dir_files['filename_glacier_id'], sep=';')

    # Get list of RGIID.
    id_list = df_id['RGIID'].values.tolist()
    
    # Crop the DataFrame by selected columns.
    mb_obs = all_gl_data.loc[:,['rgiid','period','dmdtda','err_dmdtda']]

    # The DataFrame now consists of observations of several glaciers. 
    # Crop the DataFrame by the given glacier IDs in the area of interest.
    mb_obs = mb_obs[mb_obs.rgiid.isin(id_list)]
    
    # Get 10-year balances:
    year_list = ['2000-01-01_2010-01-01', '2010-01-01_2020-01-01']

    # Crop dataframe with periods.
    mb_obs_yearly = mb_obs[mb_obs.period.isin(year_list)]
    
    # Get list of years and convert to integers.
    yrs = [int(i[0:4]) for i in list(mb_obs_yearly['period'])]
    
    # Add list of years as a column in dataframe.
    mb_obs_yearly['Year'] = yrs
    
    # Rename columns.
    mb_obs_yearly = mb_obs_yearly.rename(columns={'dmdtda':'Ba', 'err_dmdtda':'sigma'})

    # Get list of RGI ID.
    rgi_col = list(mb_obs_yearly.rgiid)
    
    # Get list of breid corresponding to list of RGI ID.
    breid_col = [df_id['BREID'].loc[df_id['RGIID']==i].values[0] for i in rgi_col]
    
    # Add BREID as column in dataframe.
    mb_obs_yearly['BREID'] = breid_col
        
    # Drop columns.
    mb_obs_yearly.drop(columns={'period','rgiid'}, inplace=True)
    
    mb_obs_yearly = mb_obs_yearly.set_index(['Year','BREID'])

    
    return(mb_obs_yearly)

#%% Get glaciological observations from file

def get_glacier_obs(name_dir_files: dict, data_type: str):
    
    """
    Function get_glacier_obs() gets observations of mass balance from 
    .csv files based on the type of observation (glacier-wide or point 
    balances) and returns observations as a Pandas DataFrame.
    
    The function gets all available mass balance measurements for all
    glaciers included in the .csv file. It then crops the observations so that
    only the observations of glacier IDs included in 'filename_glacier_id' is
    included in the final DataFrame. It does not crop observations by time
    period, but includes all available observations for the given glaciers. 
    
    Mass balance observations are read from local files. These 
    files must be stored in the 'filepath_obs' folder. 
    
    Seasonal and annual glacier wide balances for all glaciers in the area
    of interest are stored in a .csv file with name 'massbalance_gw'. Point
    balances (stake measurements) for all glaciers in the area of interest 
    are stored in a .csv file with name 'massbalance_point.csv'. The files 
    must contain the column 'GlacierId'. Glacier-wide mass balance data can be 
    downloaded from NVE glacier portal; 
    http://glacier.nve.no/glacier/viewer/ci/en/. 
    
    Parameters
    ----------
    name_dir_files : dict
        Dictionary with names of files and directories. 
    data_type : str
        Indicates the type of observation to get. Either point balance 
        ('point') or glacier-wide balances ('gw').

    Returns
    -------
    mb_obs : pandas.DataFrame
        DataFrame of observed winter, summer and annual mass balance.
    """
     
    # Filepath to observations.
    filepath_obs = name_dir_files['filepath_obs']
    
    # Get list of glacier IDs.
    df_id = pd.read_csv(name_dir_files['filepath_glacier_id'] 
                        + name_dir_files['filename_glacier_id'], sep=';')
    id_list = df_id['BREID'].values.tolist()
    
    if data_type == 'point':
        # Create pandas.DataFrame with year, BREID, location, mass balance 
        # measurements and date of measurements. 'BREID' is the glacier ID. 
        # Stake location is given by 'Y' (UTM north) and 'X' (UTM east), with
        # 'altitude' being the altitude of the stake at measurement. 'Bw' is 
        # the winter balance, 'Bs' is the summer balance and 'Ba' is the
        # annual balance. All balances are point measurements given by a 
        # location (Y, X, altitude). The unit of mass balance is m w.e.a-1.
        
        # Get name of file with point balances.
        filename_mb_obs = 'massbalance_point_UTM33.csv'
        
        # Load all glacier data as pandas.DataFrame. Cells where values are 
        # missing (eg. mass balance measurements or dates) contain nan.
        all_gl_data = pd.read_csv(filepath_obs + filename_mb_obs, sep=';')
        
        # Create column with year of measurment (Year) from column of date of
        # measurement (summer balance).
        all_gl_data['Year'] = pd.DatetimeIndex(
            all_gl_data['dt_curr_year_min_date']).year
        
        # Reorder columns so that 'Year' is the first column.
        cols = all_gl_data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        all_gl_data = all_gl_data[cols]
        
        # Rename columns.
        all_gl_data = all_gl_data.rename(columns = {'GlacierId': 'BREID',
                                                    'utm_east': 'X',
                                                    'utm_north': 'Y',
                                                    'balance_winter': 'bw',
                                                    'balance_summer': 'bs',
                                                    'balance_netto': 'ba'})
        
        # Crop the DataFrame by selected columns.
        mb_obs = all_gl_data.loc[:,['Year','BREID','X','Y','altitude',
                                    'bw','bs','ba',
                                    'dt_prev_year_min_date',
                                    'dt_curr_year_max_date',
                                    'dt_curr_year_min_date']]
        
        # The DataFrame now consists of observations of several glaciers. 
        # Crop the DataFrame by the given glacier IDs in the area of interest.
        mb_obs = mb_obs[mb_obs.BREID.isin(id_list)]
        
    elif data_type == 'gw':
        # Create pandas.DataFrame with year, BREID and mass balance 
        # measurements. # mb_obs has columns 'Year', 'BREID' (glacier ID), and 
        # 'Bw' (winter mass-balance), 'Bs' (summer mass-blance) and 'Ba' 
        # (annual mass-balance). Unit of mass balance is m w.e.a-1.
        
        # Get name of file with glacier-wide balances.
        filename_mb_obs = 'massbalanceData_reference_glaciers.csv'
        
        # Load all glacier data as pandas.DataFrame. Cells where values are 
        # missing (eg. mass balance measurements or dates) contain nan.
        all_gl_data = pd.read_csv(filepath_obs + filename_mb_obs, sep=';')
        
        # Select columns 'Year', 'GlacierId', 'Bw', 'Bs', and 'Ba' from 
        # DataFrame.
        mb_obs = all_gl_data.loc[:,['Year','GlacierId','Bw','Bs','Ba']]
        
        # Rename column GlacierId as BREID. Values in column BREID is of type
        # int.
        mb_obs = mb_obs.rename(columns = {'GlacierId': 'BREID'})
        
        # Replace ',' in values (str) with '.' and convert from string to float.
        str_cols = ['Bw','Ba','Bs']
        mb_obs[str_cols] = mb_obs[str_cols].replace(',', '.', regex=True)
        mb_obs[str_cols] = mb_obs[str_cols].astype(float)
        
        # The DataFrame now consists of observations of several glaciers. 
        # Crop the DataFrame by the given glacier IDs in the area of interest.
        mb_obs = mb_obs[mb_obs.BREID.isin(id_list)]
        
        # Set Year and BREID as index in DataFrame.
        mb_obs = mb_obs.set_index(['Year','BREID'])
    
    # Get calving data for Austdalsbreen.
    elif data_type == 'calving':
        
        # Austdalsbreen (BREID 2478) is the only glacier with calving. 
        id_list = [2478]

        # Create pandas.DataFrame with year, BREID and mass balance 
        # measurements. # mb_obs has columns 'Year', 'BREID' (glacier ID), and 
        # 'Bcalv' (calving loss). Unit of mass balance is ma-1.
        
        # Get name of file with glacier-wide balances.
        filename_mb_obs = 'massbalanceData_reference_glaciers.csv'
        
        # Load all glacier data as pandas.DataFrame. Cells where values are 
        # missing (eg. mass balance measurements or dates) contain nan.
        all_gl_data = pd.read_csv(filepath_obs + filename_mb_obs, sep=';')
        
        # Select columns 'Year', 'GlacierId', 'Bw', 'Bs', and 'Ba' from 
        # DataFrame.
        mb_obs = all_gl_data.loc[:,['Year','GlacierId','Area','Bcalv']]
        
        # Rename column GlacierId as BREID. Values in column BREID is of type
        # int.
        mb_obs = mb_obs.rename(columns = {'GlacierId': 'BREID'})
        
        # Replace ',' in values (str) with '.' and convert from string to float.
        str_cols = ['Area','Bcalv']
        mb_obs[str_cols] = mb_obs[str_cols].replace(',', '.', regex=True)
        mb_obs[str_cols] = mb_obs[str_cols].astype(float)
        
        # The DataFrame could now consist of observations of several glaciers. 
        # Crop the DataFrame by the given glacier IDs in the area of interest.
        mb_obs = mb_obs[mb_obs.BREID.isin(id_list)]
        
        # Set Year and BREID as index in DataFrame.
        mb_obs = mb_obs.set_index(['Year','BREID'])

    else:
        raise Exception('Please choose data type point (point) or ' +
              'gw (glacier-wide) for calibration.')
    
    return mb_obs
