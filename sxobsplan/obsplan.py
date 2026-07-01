"""
sxobsplan.obsplan

Target visibility checks during astronomical night.

Public API
----------
- is_target_visible(...)
- resolve_location(...)
- is_target_visible_timegrid(...)
"""

from __future__ import annotations

import warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroplan import Observer, FixedTarget
from typing import Union, Tuple, List, Dict, Any
from astropy.utils.exceptions import AstropyWarning
from .query import fetch_with_fallback  # Assuming this exists elsewhere

# from .query import fetch_with_fallback  # Assuming this exists elsewhere

__all__ = [
    "is_target_visible",
    "resolve_location",
    "is_target_visible_timegrid",
    # "sso_target_visible_daily", # Assuming implemented elsewhere
]

LocationLike = Union[EarthLocation, str]

# Ignore polar motion warnings from astropy
warnings.filterwarnings(
    "ignore",
    category=AstropyWarning,
    message="Tried to get polar motions"
)


def resolve_location(location: LocationLike) -> EarthLocation:
    """
    Safely resolve a string location code to an astropy EarthLocation.

    Parameters
    ----------
    location : EarthLocation or str
        If an `EarthLocation` is provided, it is returned as-is. 
        If a `str` is provided, it is treated as an MPC code or IAU site 
        name and resolved via `EarthLocation.of_site()`.

    Returns
    -------
    EarthLocation
        The resolved astropy EarthLocation object.

    Raises
    ------
    ValueError
        If the string code cannot be resolved by the astropy cache.
    TypeError
        If the provided location is neither a string nor an EarthLocation.
    """
    if isinstance(location, EarthLocation):
        return location

    if isinstance(location, str):
        try:
            return EarthLocation.of_site(location, refresh_cache=True)
        except Exception as e:
            raise ValueError(f"Could not resolve observatory code '{location}': {e}")

    raise TypeError("Location must be an EarthLocation object or a valid string observatory code.")


def is_target_visible(
    ra: u.Quantity,
    dec: u.Quantity,
    date: Union[Time, str],
    location: LocationLike,
    *,
    elev_min: u.Quantity = 30 * u.deg,
    twilight_horizon: u.Quantity = -18 * u.deg,
    duration: u.Quantity = 1 * u.hour,
    dt_step: u.Quantity = 10 * u.min,
    return_block: bool = True,
) -> Union[bool, Tuple[bool, List[Dict[str, Any]]]]:
    """
    Determine if a target is visible at a given location and specific date/epoch.

    Evaluates whether the target maintains an altitude above `elev_min` 
    during astronomical nighttime (or specified `twilight_horizon`) for a 
    continuous block of time greater than `duration`.

    Parameters
    ----------
    ra : astropy.units.Quantity
        Target right ascension (angle). Must be scalar.
    dec : astropy.units.Quantity
        Target declination (angle). Must be scalar.
    date : astropy.time.Time or str
        The date or datetime (UTC) to check. Represents the ephemeris epoch.
    location : LocationLike
        Observatory location (EarthLocation object or MPC/IAU string code).
    elev_min : astropy.units.Quantity, optional
        Minimum altitude threshold. Default is 30 degrees.
    twilight_horizon : astropy.units.Quantity, optional
        Sun altitude defining the start/end of night. Default is -18 deg (Astronomical).
    duration : astropy.units.Quantity, optional
        Minimum continuous visibility time required. Default is 1 hour.
    dt_step : astropy.units.Quantity, optional
        Temporal resolution of the visibility check. Default is 10 minutes.
    return_block : bool, optional
        If True, returns a tuple of (boolean visibility, list of visibility blocks).
        If False, returns only the boolean visibility. Default is True.

    Returns
    -------
    is_visible : bool
        True if the target meets all visibility constraints.
    blocks : list of dict, optional
        Returned only if `return_block` is True. Contains dictionaries with keys:
        'start' (Time), 'end' (Time), and 'duration' (Quantity).
    """
    t0 = date if isinstance(date, Time) else Time(date)
    t0 = t0.utc

    # Guard against accidental arrays passed to the scalar function
    if not t0.isscalar:
        raise ValueError("`date` must be a scalar Time object. Use `is_target_visible_timegrid` for arrays.")

    loc = resolve_location(location)
    obs = Observer(location=loc, name=str(location), timezone="UTC")
    coord = SkyCoord(ra=ra, dec=dec)
    target = FixedTarget(coord=coord, name="Target")

    # 1. Build a time grid centered exactly on the ephemeris epoch (t0)
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_offsets = np.arange(-12 * 60, 12 * 60 + step_min, step_min) * u.min
    time_grid = t0 + time_offsets

    # 2. Vectorized darkness and altitude checks
    A_dark = obs.is_night(time_grid, horizon=twilight_horizon)
    altitudes = obs.altaz(time_grid, target).alt
    A_high = altitudes >= elev_min

    # 3. Combine masks
    A_vis = A_high & A_dark

    # 4. Detect continuous observing blocks
    padded = np.concatenate([[False], A_vis, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]

    blocks = []
    for s, e in zip(starts, stops):
        # Calculate true duration of the visible block
        dur = ((e - s) * dt_step).to(u.hour)
        
        # Keep blocks that satisfy minimum continuous duration
        if dur >= duration:
            blocks.append({
                "start": time_grid[s],
                "end":   time_grid[e - 1],  # Time of the final valid observation
                "duration": dur
            })

    is_visible = len(blocks) > 0
    return (is_visible, blocks) if return_block else is_visible


def is_target_visible_timegrid(
    ra: u.Quantity,
    dec: u.Quantity,
    dates: Union[Time, np.ndarray, list],
    location: LocationLike,
    *,
    elev_min: u.Quantity = 30 * u.deg,
    twilight_horizon: u.Quantity = -18 * u.deg,
    duration: u.Quantity = 1 * u.hour,
    dt_step: u.Quantity = 10 * u.min,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine target visibility over an array of dates using optimized 2D time grids.

    Parameters
    ----------
    ra : astropy.units.Quantity
        Right ascension (angle). Can be a scalar (1 target) or a 1D array matching `dates`.
    dec : astropy.units.Quantity
        Declination (angle). Can be a scalar (1 target) or a 1D array matching `dates`.
    dates : astropy.time.Time, numpy.ndarray, or list
        Array of UTC times or Julian Dates representing the epochs to check.
    location : LocationLike
        Observatory location (EarthLocation object or MPC/IAU string code).
    elev_min : astropy.units.Quantity, optional
        Minimum altitude threshold. Default is 30 degrees.
    twilight_horizon : astropy.units.Quantity, optional
        Sun altitude defining the start/end of night. Default is -18 deg.
    duration : astropy.units.Quantity, optional
        Minimum continuous visibility time required. Default is 1 hour.
    dt_step : astropy.units.Quantity, optional
        Temporal resolution of the visibility check. Default is 10 minutes.

    Returns
    -------
    is_visible_arr : numpy.ndarray (bool)
        1D boolean array indicating visibility success for each date.
    duration_arr : numpy.ndarray (float)
        1D array of maximum continuous visibility durations (in hours) for each date. 
        Returns NaN where `is_visible_arr` is False.
    """
    loc = resolve_location(location)
    obs = Observer(location=loc, name=str(location), timezone="UTC")

    t0 = dates if isinstance(dates, Time) else Time(dates)
    t0 = t0.utc
    
    # Ensure t0 is an array to support broadcasting 
    if t0.isscalar:
        t0 = t0.reshape(1)
    
    N = len(t0)

    # 1. Build 2D time grid (N, T)
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_offsets = np.arange(-12 * 60, 12 * 60 + step_min, step_min) * u.min
    time_grid = t0[:, np.newaxis] + time_offsets[np.newaxis, :]

    # 2. Safely broadcast RA/DEC to array size N to prevent np.newaxis IndexErrors
    ra_arr = np.broadcast_to(np.atleast_1d(ra), N)
    dec_arr = np.broadcast_to(np.atleast_1d(dec), N)
    
    coord = SkyCoord(ra=ra_arr, dec=dec_arr)[:, np.newaxis]
    target = FixedTarget(coord=coord, name="Target")

    # 3. Vectorized AltAz and Darkness checks
    A_dark = obs.is_night(time_grid, horizon=twilight_horizon)
    altitudes = obs.altaz(time_grid, target).alt
    A_high = altitudes >= elev_min

    # 4. Combine masks
    A_vis = A_high & A_dark

    # 5. Extract continuous blocks per date
    is_visible_arr = np.zeros(N, dtype=bool)
    duration_arr = np.full(N, np.nan)
    
    dur_threshold = duration.to(u.hour).value
    step_hr = step_min / 60.0

    for i in range(N):
        row_vis = A_vis[i]
        
        # Detect state changes (True <-> False)
        padded = np.concatenate([[False], row_vis, [False]])
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts, stops = changes[::2], changes[1::2]
        
        if len(starts) > 0:
            durs = (stops - starts) * step_hr
            max_dur = np.max(durs)
            
            if max_dur >= dur_threshold:
                is_visible_arr[i] = True
                duration_arr[i] = max_dur

    return is_visible_arr, duration_arr

def is_bulk_targets_visible_timegrid(
    ra: u.Quantity,                       # 2D Array of Right Ascension, shape (M, N)
    dec: u.Quantity,                      # 2D Array of Declination, shape (M, N)
    dates: Union[Time, np.ndarray],       # 1D Array of baseline epochs, shape (N,)
    location: Union[EarthLocation, str],  # Observatory location
    *,
    elev_min: u.Quantity = 30 * u.deg,
    twilight_horizon: u.Quantity = -18 * u.deg,
    duration: u.Quantity = 1 * u.hour,
    dt_step: u.Quantity = 10 * u.min,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine target visibility for bulk targets over massive time grids using 
    optimized 3D matrix broadcasting.

    Parameters
    ----------
    ra : astropy.units.Quantity
        2D array of Right Ascensions of shape (M, N), where M is the number of 
        targets and N is the number of dates.
    dec : astropy.units.Quantity
        2D array of Declinations of shape (M, N).
    dates : astropy.time.Time or numpy.ndarray
        1D array of unique observation dates of shape (N,).
    location : EarthLocation or str
        Observatory location identifier or object.
    elev_min : astropy.units.Quantity, optional
        Minimum acceptable altitude. Default is 30 degrees.
    twilight_horizon : astropy.units.Quantity, optional
        Sun altitude marking nighttime limits. Default is -18 degrees.
    duration : astropy.units.Quantity, optional
        Minimum required continuous observation window. Default is 1 hour.
    dt_step : astropy.units.Quantity, optional
        Sampling step resolution. Default is 10 minutes.

    Returns
    -------
    is_visible_matrix : numpy.ndarray (bool)
        2D boolean mask of shape (M, N) indicating visibility success.
    max_duration_matrix : numpy.ndarray (float)
        2D float matrix of shape (M, N) containing the maximum continuous 
        observing hours for each target on each day (NaN if not visible).
    """
    # 1. Standardize inputs and initialize tracking metrics
    if isinstance(location, str):
        loc = EarthLocation.of_site(location, refresh_cache=True)
    else:
        loc = location
        
    obs = Observer(location=loc, timezone="UTC")
    t0 = dates if isinstance(dates, Time) else Time(dates)
    t0 = t0.utc
    
    M, N = ra.shape
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_offsets = np.arange(-12 * 60, 12 * 60 + step_min, step_min) * u.min
    T = len(time_offsets)
    
    # 2. Build 2D baseline time grid (N, T) and compute unique Night Mask
    time_grid_2d = t0[:, np.newaxis] + time_offsets[np.newaxis, :]
    A_dark_2d = obs.is_night(time_grid_2d, horizon=twilight_horizon)
    
    # Broadcast night mask up to 3D: Shape becomes (1, N, T)
    A_dark_3d = A_dark_2d[np.newaxis, :, :]

    # 3. Prepare 3D Shapes for Vectorized Astropy Frame Transformation
    # Reshape coordinates to (M, N, 1)
    coord_3d = SkyCoord(ra=ra, dec=dec)[:, :, np.newaxis]
    # Reshape times to (1, N, T)
    time_grid_3d = time_grid_2d[np.newaxis, :, :]
    
    # Execute full 3D matrix transformation via C-compiled ERFA pipeline
    altaz_frame = AltAz(obstime=time_grid_3d, location=loc)
    altitudes = coord_3d.transform_to(altaz_frame).alt
    A_high_3d = altitudes >= elev_min

    # 4. Synthesize logical criteria into a uniform visibility matrix
    A_vis_3d = A_high_3d & A_dark_3d  # Shape: (M, N, T)

    # 5. Extract continuous windows using an efficient Sliding Window filter
    steps_required = int(np.ceil((duration / dt_step).decompose().value))
    
    if T >= steps_required:
        # Create sliding windows along the fine time axis (axis=2)
        # Resulting window shape: (M, N, T - steps_required + 1, steps_required)
        windows = np.lib.stride_tricks.sliding_window_view(A_vis_3d, window_shape=steps_required, axis=2)
        # Target is visible if ALL elements in ANY window evaluate to True
        is_visible_matrix = windows.all(axis=3).any(axis=2)
    else:
        is_visible_matrix = np.zeros((M, N), dtype=bool)

    # 6. Calculate Maximum Continuous Durations
    # To maintain optimal speed, loop only over rows with confirmed visibility
    max_duration_matrix = np.full((M, N), np.nan)
    step_hr = step_min / 60.0
    
    active_indices = np.argwhere(is_visible_matrix)
    for m, n in active_indices:
        row_vis = A_vis_3d[m, n]
        padded = np.concatenate([[False], row_vis, [False]])
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts, stops = changes[::2], changes[1::2]
        
        if len(starts) > 0:
            max_duration_matrix[m, n] = np.max(stops - starts) * step_hr

    return is_visible_matrix, max_duration_matrix

def _get_best_obstime(ephem, min_alt=20.0):
    """
    Supportive function to evaluate target visibility, best UT time, and magnitude.
    
    Parameters
    ----------
    ephem : astropy.table.Table
        Ephemeris table returned from JPL Horizons.
    min_alt : float
        Minimum altitude (elevation) in degrees for the target to be considered visible.
        
    Returns
    -------
    is_target_visible : bool
    best_time_ut : str or None
    mag : float or None
    """
    # 1. Check for Night Time
    # 'solar_presence' column: '*' (daylight), 'C' (civil twilight), 'N' (nautical), 'A' (astronomical), '' (dark)
    # We consider it "night" if it is not daylight and not any twilight.
    if 'solar_presence' in ephem.colnames:
        is_night = ~np.isin(ephem['solar_presence'], ['*', 'C', 'N', 'A'])
    else:
        # Fallback if the column is missing
        is_night = np.ones(len(ephem), dtype=bool)

    # 2. Check for Altitude (Elevation)
    if 'EL' in ephem.colnames:
        visible = (ephem['EL'] > min_alt) & is_night
    else:
        visible = np.zeros(len(ephem), dtype=bool)

    # If the target never breaches the minimum altitude during the night, it's not visible
    if not np.any(visible):
        return False, None, None

    # Filter down to only the visible night-time rows
    vis_ephem = ephem[visible]
    
    # 3. Find Best Time (Highest Altitude)
    best_idx = np.argmax(vis_ephem['EL'])
    best_row = vis_ephem[best_idx]
    
    # Extract UT time in HH:MM format from 'datetime_str'
    best_time_str = best_row['datetime_str']
    best_time_ut = best_time_str.split(' ')[1][:5]
    
    # 4. Extract Magnitude
    # Comets usually return 'Tmag' (Total magnitude) while asteroids return 'V'
    if 'Tmag' in best_row.colnames and not np.ma.is_masked(best_row['Tmag']):
        mag = float(best_row['Tmag'])
    elif 'V' in best_row.colnames and not np.ma.is_masked(best_row['V']):
        mag = float(best_row['V'])
    else:
        mag = np.nan
        
    return True, best_time_ut, round(mag, 1) if not np.isnan(mag) else np.nan


def sso_target_visible_daily(target_name, date, location, min_alt=20.0):
    """
    Checks the visibility of a Solar System Object on a specific date, 
    calculating the best observation time and magnitude.
    
    Parameters
    ----------
    target_name : str
        Target designation (e.g., '24P', 'C/2023 C2').
    date : str
        The date to check in 'YYYY-MM-DD' format.
    location : str
        Observatory location code (based on MPC observatory code standards).
    min_alt : float
        Minimum altitude in degrees to consider the target visible.
        
    Returns
    -------
    is_target_visible : bool
        True if the target is visible at night above the minimum altitude.
    best_time_UT : str
        The UT time (HH:MM) when the target reaches maximum altitude during the night.
    mag : float
        The Tmag (or Vmag) of the target at the best observation time.
    """
    # Construct a 24-hour UT time window for the given date
    start_time = f"{date} 00:00:00"
    t_start = Time(start_time, format='iso', scale='utc')
    t_stop = t_start + 1.0*u.day
    
    epochs = {
        'start': t_start.strftime('%Y-%m-%d %H:%M'),
        'stop': t_stop.strftime('%Y-%m-%d %H:%M'),
        'step': '30m'
    }
    
    try:
        # Use the imported function from query.py instead of calling Horizons directly
        ephem = fetch_with_fallback(target_id=target_name, location=location, epochs=epochs)
        
        # Guard clause in case the query fails or returns None
        if ephem is None or len(ephem) == 0:
            return False, None, None
            
        return _get_best_obstime(ephem, min_alt=min_alt)
        
    except Exception as e:
        print(f"Error processing visibility for {target_name}: {e}")
        return False, None, None