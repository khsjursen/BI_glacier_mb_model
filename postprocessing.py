# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:43:24 2021

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Function to calculate mass balance from accumulation array.

"""

#%% Libraries

# Standard libraries

# External libraries
import numpy as np
import datetime as dt
import numexpr as ne

# Internal libraries
#%% Function get_glacier_mass_balance()

def get_glacier_mass_balance_upd(accumulation, da_gl_spec_frac, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates mass balance (winter, summer and annual balance) for each 
    glacier for the given year.
    
    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 
    
    Calculations are done for each glacier specified in BREID.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    gl_id : int
        Glacier ids.
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-10-31', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. Expand dimensions for
        # multiplication.
        glacier_acc = accumulation[idx_start:idx_end,:,:]
        glacier_acc = np.expand_dims(glacier_acc, axis=0)

        # Get all glacier masks for the given hydrological year (the 
        # hydrological year that has just ended). Expand dimensions for
        # multiplication.
        mask_glacier_all = np.array(da_gl_spec_frac.sel(time = hyd_yr).values)
        mask_glacier_all = np.expand_dims(mask_glacier_all, axis=1)
                
        # Get accumulation over each specific glacier for the given
        # hydrological year by multiplying specific glacier masks with the 
        # accumulation grid.
        gl_spec_acc = ne.evaluate("""glacier_acc * mask_glacier_all""".replace(" ",""))
        
        # Sum accumulation over all cells of each specific glacier.
        glacier_acc_d = gl_spec_acc.sum(axis=(2,3)) / mask_glacier_all.sum(axis=(2,3))
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period.
        mb[:,0] = (np.amax(glacier_acc_d[:,-365:None],axis=1) 
                    - np.amin(glacier_acc_d[:,None:(365-90)],axis=1))
        
        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year.
        mb[:,1] = (np.amin(glacier_acc_d[:,365:None], axis=1) 
                       - np.amax(glacier_acc_d[:,-365:None], axis=1))
        
        # Annual balance is sum of winter and summer balance.
        mb[:,2] = mb[:,0] + mb[:,1]
        
        # Convert balances to m w.e.
        mb = mb/1e3
        
    # Return array of mass balances. Will be nan for first year.
    return mb

# End of function get_glacier_mass_balance()

#%% End of postprocessing.py
