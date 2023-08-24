# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:09:06 2021

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Function for calculating potential direct incoming solar radiation in each
grid cell of the model DEM on each day of the year. 

"""

#%% Libraries

# Standard libraries
from math import pi, sin, cos, tan
from pyproj import Proj
import numexpr as ne

# External libraries
import numpy as np
import xrspatial as xrs

# Internal libraries

#%% Function calc_pot_rad()
def calc_pot_rad(da_dem_elevation):

    """
    The function calc_pot_rad() calculates the mean potential direct incoming 
    solar radiation in each grid cell on each day of the year.
    
    Potential direct incoming solar radiation is calculated from the mean 
    latitude of the area, solar angles and the topograpy (aspect, slope) of 
    the area. In case of leap years, solar radiation on day 366 of the year is
    set equal to solar radiation on day 365.
    
    Parameters
    ----------
    da_dem_elevation : xarray.DataArray
        DataArray for the area containing:
        res : float
            Resolution of DEM (cellsize).
        X : float
            X-coordinate of cell centers.
        Y : float
            Y-coordinate of cell centers.
        elevation (Y,X) : xarray.DataArray (float)
            Elevation in each cell of the area.

    Returns
    -------
    pot_rad_mean : numpy.array (366,Y,X)
        Array of mean daily potential direct incoming solar radiation in each 
        grid cell for each day of the year, also taking into account an extra
        day for leap years where values correspond to day 365. 
    """    

    #%% Define constants

    I_0 = 1368 # Solar constant [W m-2]
    psi_a = np.float32(0.75) # Clear-sky transmissivity, from Hock (1999)
    p_0 = 101325 # Mean pressure at sea level [Pa]

    #%% Calculate slope and aspect of each grid cell
    
    # Calculate the slope in each grid cell in radians. NB! The slope in 
    # border cells is not calculated, but set to zero. Calculate aspect 
    # (orientation of slope, measured clockwise in degrees 
    # from 0 to 360) in each grid cell. 0 is north-facing, 90 is east, 180 is 
    # south and 270 is west. NB! The slope in border cells is not calculated, 
    # but set to zero.
    def getSlopeAspect(da_elev):
        s = xrs.slope(da_elev)
        a = xrs.aspect(da_elev)
        return(np.array(s.values, dtype=np.float32) * pi / 180, 
               np.array(a.values, dtype=np.float32) * pi / 180)
    
    slope_rad, aspect_rad = getSlopeAspect(da_dem_elevation)

    #%% Get elevation
    
    elevation = np.array(da_dem_elevation.values, dtype=np.float32)

    #%% Get the mean latitude of the area
    
    # Get coordinates.
    X = np.array(da_dem_elevation.X.values, dtype=np.float32)
    Y = np.array(da_dem_elevation.Y.values, dtype=np.float32)

    # Mesh UTM33 coordinates.
    x, y = np.meshgrid(X, Y)

    # Define projection.
    myProj = Proj("+proj=utm +zone=33, \
                  +north +ellps=WGS84 +datum=WGS84 +units=m + no_defs")

    # Convert UTM33 coordinates to longitude and latitude.
    lon, lat = myProj(x, y, inverse = True)
    
    # Get mean latitude to use in calculations.
    lat_mean = lat.mean()
    
    # In radians.
    lat_mean_rad = ((pi/180)*lat_mean).astype(np.float32)
    
    #%% Find hours of the day between sunrise and sunset for each day
    
    # Initialize vectors of days and hours.    
    d = np.array(range(1,367)) # Day 1 to 366 of the year
    d[365] = 365 # Set value on day 366 (leap year) equal to 365
    h = np.array(range(1,25)) # Hour 1 to 24 of the day

    # Declination angle on the given day [radians].
    delta_init = -np.arcsin(0.397901 * np.cos(2 * pi * (d + 11) / 365.25)) # (366,)
    delta_init = delta_init.reshape(-1,1).astype(np.float32) # (366,1)
    
    # Hour angle at sunrise (negative) and sunset (positive) on the given
    # day [radians].
    hra_0 = np.arccos(-tan(lat_mean_rad) * np.tan(delta_init)) # (366,)
    
    # Hour angle [radians].
    hra_init = ((pi / 180) * 15 * (h - 12)).astype(np.float32) # (24,)
    #hra_init = hra_init.reshape(1,-1) # (1,24)
    
    # Create mask of hours between sunrise and sunset. Cells containing ones
    # are between sunrise and sunset and cells containing zeros are between
    # sunset and sunrise.
    mask_sunup = 1 - ((hra_init > hra_0) + (hra_init < -hra_0)) # (366,24)
    
    # Create vector of zeros and ones for masking day numbers and hours 
    # between sunrise and sunset.
    mask_sunup_vec = np.reshape(mask_sunup, -1)
    
    # Initialize vector of day numbers. This vector contains the day number 
    # for each hour of the year (8784 entries, including an extra day for leap
    # years). The vector contains 24 ones (indicating that the first 24 hours 
    # are part of day 1 of the year), 24 twos, 24 threes, and so on. 
    days = np.repeat(np.arange(1,(366+1)), repeats=24, axis=0)
    # Set values in last 24 entries (day 366, in case of leap year) to 365. 
    days[-24:] = 365 
    
    # Mask out the instances (combinations of day numbers and hours) that 
    # are between sunrise and sunset. The vector now contains the number of 
    # hours between sunrise and sunset on each day of the year. For example, 
    # if the number of hours between sunrise and sunset on day 1 and day 2 of 
    # the year is 5, the vector now contains 5 ones, 5 twos, and so on.
    days_sunup = days[mask_sunup_vec>0]
    
    # Initialize vector of hours. This vector contains hour numbers for each 
    # day of the year (8784 entries, including an extra day for leap years). 
    # The vector contains the numbers 1-24 repeated 366 times. 
    hrs = np.tile(np.arange(1,(24+1)), reps=366)
    
    # Mask out the instances (combinations of day numbers and hours) that 
    # are between sunrise and sunset. The vector now contains the hour numbers
    # between sunrise and sunset for each day. For example, if the sun rises
    # at 10 o'clock on day 1 and day 2 of the year and sets at 14 o'clock, 
    # the vector now contains the numbers 10, 11, 12, 13, 14, 10, 11, 12, 13,
    # 14, and similarly for the remaining days of the year. The vectors
    # days_sunup and hrs_sunup now contain the same number of entries and 
    # indicate together a day number and hour number that is between sunrise
    # and sunset and for which the potential direct solar radiation should be
    # calculated.
    hrs_sunup = hrs[mask_sunup_vec>0]
    
    #%% Calculate potential direct solar radiation between sunrise and sunset
    
    # Atmospheric pressure at altitude.
    p = (101325 * np.power((1 - 2.25577 * (1e-5) 
                           * elevation), 5.25588)).astype(np.float32) # (Y,X)
    
    # Eccentricity correction factor for each day.
    ecc_corr_factor = (1 + 0.033 * np.cos((2 * pi * days_sunup) / 365)).astype(np.float32) # (4390,)
      
    # Declination angle for each day [radians].
    delta = (-np.arcsin(0.397901 * np.cos(2 * pi * (days_sunup + 11) / 365.25))).astype(np.float32) # (4390,)
    
    # Alternative delta calculation (Spencer formula):
    # day_angle = 2 * pi * ((days_sunup-1)/365)
    # delta = (0.006918 - 0.399912*np.cos(day_angle) + 0.070257*np.sin(day_angle)
    #          - 0.006758*np.cos(2*day_angle) + 0.000907*np.sin(2*day_angle) 
    #          - 0.002697*np.cos(3*day_angle) + 0.001480*np.sin(3*day_angle))

    # Hour angle for each hour [radians].
    hra = ((pi / 180) * 15 * (hrs_sunup - 12)).astype(np.float32) # (4390,)    

    # Solar zenith angle for a given hour on a given day [radians].
    zenith_angle = (np.arccos((sin(lat_mean_rad) * np.sin(delta))
                         + (cos(lat_mean_rad) * np.cos(delta) * np.cos(hra)))) # (4390,)

    # Solar zenith angle for a given hour on a given day [degrees].
    zenith_angle_deg = zenith_angle * (180 / pi)

    # Solar azimuth angle for a given hour of a given day [radians].
    # Due to rounding errors cos_solar_azimuth is sometimes slightly
    # smaller than -1, which gives nan values when taking the inverse
    # cosine. These instances are therefore corrected manually. 
    cos_solar_azimuth = ((np.sin(delta)*cos(lat_mean_rad)-np.cos(hra)*np.cos(delta)*sin(lat_mean_rad))
                                /(np.sin(zenith_angle)))
    cos_solar_azimuth[cos_solar_azimuth<-1]=-1.0
    solar_azimuth = np.arccos(cos_solar_azimuth)
    
    # When the hour angle is positive (afternoon), the azimuth
    # angle is 360 deg - azimuth angle calculated by the above 
    # formula. 
    solar_azimuth[hra>0] = 2*pi-solar_azimuth[hra>0]

    # Alternative solar azimuth calculation:
    # With this calculation N and S surfaces are swapped.
    # cos_solar_azimuth = ((sin(lat_mean_rad) * np.cos(zenith_angle) - np.sin(delta)) 
    #                             / (cos(lat_mean_rad) * np.sin(zenith_angle)))
    # cos_solar_azimuth[cos_solar_azimuth>1]=1.0
    # solar_azimuth = np.arccos(cos_solar_azimuth)
    # solar_azimuth[hra<0] = -solar_azimuth[hra<0]

    # Expand arrays.
    slope_rad = np.expand_dims(slope_rad, axis=(0)) # Expand dims to (1,Y,X)
    aspect_rad = np.expand_dims(aspect_rad, axis=(0)) # Expand dims to (1,Y,X)
    delta = np.expand_dims(delta, axis=(1,2)) # Expand dims to (4390,1,1)
    hra = np.expand_dims(hra, axis=(1,2)) # Expand dims to (4390,1,1)
    zenith_angle = np.expand_dims(zenith_angle, axis=(1,2))
    solar_azimuth = np.expand_dims(solar_azimuth, axis=(1,2))

    # Theta is the angle of incidence between the normal and grid slope.
    # Based on formula from Hock (1999). Theta larger than 90 degrees means
    # that the sun is behind the surface, and hence theta is set to zero, which
    # gives zero incoming radiation on that surface.
    #cos_theta = np.cos(slope_rad)*np.cos(zenith_angle) + np.sin(slope_rad)*np.sin(zenith_angle)*np.cos(solar_azimuth-aspect_rad)
    cos_theta = ne.evaluate("""
                            cos(slope_rad) * cos(zenith_angle) 
                            + sin(slope_rad) * sin(zenith_angle) 
                            * cos(solar_azimuth - aspect_rad)
                            """.replace("\n", "").replace(" ","")
                            )
    theta_deg = np.arccos(cos_theta)*180/pi
    cos_theta[theta_deg>90] = 0

    # Expand arrays.
    p = np.expand_dims(p, axis=(0)) # Expand dims to (1,Y,X)
    ecc_corr_factor = np.expand_dims(ecc_corr_factor, axis=(1,2)) # (4390,1,1)
    
    # Initialize array of potential direct solar radiation in each cell for 
    # each hour and day of the year. 
    pot_rad_tot = np.zeros((366,24,
                            da_dem_elevation.shape[0], 
                            da_dem_elevation.shape[1]), dtype=np.float32)
    
    # Populate array with values based on mask of the hours and days that are
    # between sunrise and sunset.
    #pot_rad_tot[mask_sunup>0] = pot_rad_hr
    pot_rad_tot[mask_sunup>0] = ne.evaluate("I_0 * ecc_corr_factor * psi_a **(p / (p_0 * cos(zenith_angle))) * cos_theta")
    
    # Take the mean of values accross the 24 hours to get the average 
    # incoming potential solar radiation in each cell for each day. Values on 
    # day 366 correspond to those on day 365. 
    pot_rad_sum = pot_rad_tot.mean(axis=1)
    
    return(pot_rad_sum)

# End of function calc_pot_rad() 
