"""
sxobsplan.obsplan

Target visibility checks during astronomical night.

Public API
----------
- is_target_visible(...)
- resolve_location(...)
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer, FixedTarget
from typing import Union
from .query import fetch_with_fallback

__all__ = [
    "is_target_visible",
    "resolve_location",
]

LocationLike = Union[EarthLocation, str]  # EarthLocation object or observatory name

# Ignore polar motion warnings from astropy
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings(
    "ignore",
    category=AstropyWarning,
    message="Tried to get polar motions"
)

def resolve_location(location: LocationLike) -> EarthLocation:
    """
    Resolve a location to an EarthLocation.

    Parameters
    ----------
    location : EarthLocation | str
        - EarthLocation: returned as-is.
        - str: assumed to be resolvable by EarthLocation.of_site
               (MPC code or IAU site name).

    Returns
    -------
    EarthLocation

    Raises
    ------
    ValueError
        If the string code cannot be resolved.
    TypeError
        If `location` is of unsupported type.
    """
    if isinstance(location, EarthLocation):
        return location

    if isinstance(location, str):
        try:
            return EarthLocation.of_site(location, refresh_cache=True)
        except Exception as e:
            raise ValueError(f"Could not resolve observatory code '{location}': {e}")

    raise TypeError("location must be EarthLocation or str (observatory name).")

# def _normalize_date_to_noon_utc(date) -> Time:
#     """
#     Convert input to Time in UTC; if it's a bare date (00:00),
#     shift to 12:00 UTC to ensure proper twilight bracketing.
#     """
#     t0 = date if isinstance(date, Time) else Time(date)
#     t0 = t0.utc

#     midnight = Time(t0.iso.split(" ")[0] + " 00:00", scale="utc")
#     if abs((t0 - midnight).to(u.second).value) < 1.0:
#         return Time(f"{t0.utc.iso.split(' ')[0]} 12:00", scale="utc")
#     return t0


# def is_target_visible(
#     ra: u.Quantity,                       # Right ascension [angle], e.g., 120*u.deg or 8*u.hourangle
#     dec: u.Quantity,                      # Declination [angle]
#     date,                                 # Time | datetime/date | str (UTC)
#     location: LocationLike,               # EarthLocation | MPC code
#     *,
#     elev_min: u.Quantity = 30 * u.deg,    # Minimum altitude
#     duration: u.Quantity = 1 * u.hour,    # Required continuous observing time
#     dt_step: u.Quantity = 10 * u.min,     # Sampling cadence
#     return_block: bool = True,            # Whether to return visibility blocks
# ):
#     """
#     Determine target visibility.

#     Parameters
#     ----------
#     ra, dec : Quantity[angle]
#         Target coordinates (RA may be in hourangle).
#     date : Time | datetime/date | str
#         Date or datetime (UTC). If only a date is given (00:00),
#         internally shifted to 12:00 UTC.
#     location : EarthLocation | str
#         Observatory location. If string, treated as MPC/IAU observatory code.
#     elev_min : Quantity[angle], default 30 deg
#         Minimum altitude threshold.
#     duration : Quantity[time], default 1 hour
#         Minimum continuous visibility required.
#     dt_step : Quantity[time], default 2 min
#         Altitude sampling cadence.
#     return_block : bool, default True
#         If True, return (is_visible, blocks).
#         If False, return only is_visible (bool).

#     Returns
#     -------
#     bool or (bool, list[dict])
#         - If return_block=False → only `is_visible` (bool).
#         - If return_block=True  → (is_visible, blocks),
#           where blocks is a list of dicts:
#             {"start": Time, "end": Time, "duration": Quantity[time]}.
#     """
#     t0 = _normalize_date_to_noon_utc(date)
#     loc = resolve_location(location)
#     coord = SkyCoord(ra=ra, dec=dec)

#     # Observer and target
#     obs = Observer(location=loc, name=str(location), timezone="UTC")
#     target = FixedTarget(coord=coord, name="Target")

#     # Twilight bounds
#     try:
#         e_twilight = obs.twilight_evening_astronomical(t0, which="next")
#         m_twilight = obs.twilight_morning_astronomical(e_twilight, which="next")
#         if (e_twilight is None) or (m_twilight is None):
#             return (False, []) if return_block else False
#     except Exception:
#         return (False, []) if return_block else False

#     # Build time grid
#     step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
#     time_grid = t0 + np.arange(0, 24 * 60, step_min) * u.min

#     # Altitudes & masks
#     altitudes = obs.altaz(time_grid, target).alt
#     A_high = altitudes >= elev_min
#     A_dark = (time_grid >= e_twilight) & (time_grid <= m_twilight)
#     A_vis = A_high & A_dark

#     # Detect blocks
#     padded = np.concatenate([[False], A_vis, [False]])
#     changes = np.flatnonzero(padded[1:] != padded[:-1])
#     starts, stops = changes[::2], changes[1::2]

#     blocks = []
#     for s, e in zip(starts, stops):
#         if e - s <= 1:
#             continue
#         dur = (time_grid[e - 1] - time_grid[s]).to(u.hour)
#         if dur >= duration:
#             blocks.append({
#                 "start": time_grid[s],
#                 "end":   time_grid[e - 1],
#                 "duration": dur
#             })

#     is_visible = len(blocks) > 0
#     return (is_visible, blocks) if return_block else is_visible


def is_target_visible(
    ra: u.Quantity,                       # Right ascension [angle]
    dec: u.Quantity,                      # Declination [angle]
    date,                                 # Time | datetime/date | str (UTC)
    location: LocationLike,               # EarthLocation | MPC code
    *,
    elev_min: u.Quantity = 30 * u.deg,    # Minimum altitude
    duration: u.Quantity = 1 * u.hour,    # Required continuous observing time
    dt_step: u.Quantity = 10 * u.min,     # Sampling cadence
    return_block: bool = True,            # Whether to return visibility blocks
):
    """
    Determine target visibility.

    Parameters
    ----------
    ra, dec : Quantity[angle]
        Target coordinates (RA may be in hourangle).
    date : Time | datetime/date | str
        Date or datetime (UTC). Represents the exact epoch of the coordinates.
    location : EarthLocation | str
        Observatory location. If string, treated as MPC/IAU observatory code.
    elev_min : Quantity[angle], default 30 deg
        Minimum altitude threshold.
    duration : Quantity[time], default 1 hour
        Minimum continuous visibility required.
    dt_step : Quantity[time], default 10 min
        Altitude sampling cadence.
    return_block : bool, default True
        If True, return (is_visible, blocks).
        If False, return only is_visible (bool).
    """
    t0 = date if isinstance(date, Time) else Time(date)
    t0 = t0.utc

    loc = resolve_location(location)
    obs = Observer(location=loc, name=str(location), timezone="UTC")
    coord = SkyCoord(ra=ra, dec=dec)
    target = FixedTarget(coord=coord, name="Target")

    # 1. Build a time grid centered exactly on the ephemeris epoch (t0)
    # Spanning +/- 12 hours ensures we capture the nearest local night globally
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_offsets = np.arange(-12 * 60, 12 * 60 + step_min, step_min) * u.min
    time_grid = t0 + time_offsets

    # 2. Vectorized darkness check using Astronomical twilight (-18 deg horizon)
    # This replaces the buggy `twilight_evening_astronomical(which="next")`
    A_dark = obs.is_night(time_grid, horizon=-18 * u.deg)

    # 3. Calculate target altitudes
    altitudes = obs.altaz(time_grid, target).alt
    A_high = altitudes >= elev_min

    # 4. Combine masks
    A_vis = A_high & A_dark

    # Detect continuous observing blocks
    padded = np.concatenate([[False], A_vis, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]

    blocks = []
    for s, e in zip(starts, stops):
        if e - s <= 1:
            continue
        
        # Calculate true duration of the visible block
        dur = ((e - s) * dt_step).to(u.hour)
        if dur >= duration:
            blocks.append({
                "start": time_grid[s],
                "end":   time_grid[e - 1],
                "duration": dur
            })

    is_visible = len(blocks) > 0
    return (is_visible, blocks) if return_block else is_visible



def is_target_visible_timegrid(
    ra: u.Quantity,                       # 1D Array of Right ascension [angle]
    dec: u.Quantity,                      # 1D Array of Declination [angle]
    dates,                                # 1D Array of Time | float (JD) (UTC)
    location: LocationLike,               # EarthLocation | str
    *,
    elev_min: u.Quantity = 30 * u.deg,    # Minimum altitude
    duration: u.Quantity = 1 * u.hour,    # Required continuous observing time
    dt_step: u.Quantity = 10 * u.min,     # Sampling cadence
):
    """
    Determine target visibility over an array of dates using 2D time grids.
    Returns boolean array of visibility and an array of max continuous durations.
    """
    loc = resolve_location(location)
    obs = Observer(location=loc, name=str(location), timezone="UTC")

    t0 = dates if isinstance(dates, Time) else Time(dates)
    t0 = t0.utc
    N = len(t0)

    # 1. Build 2D time grid
    # time_offsets shape: (T,)
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_offsets = np.arange(-12 * 60, 12 * 60 + step_min, step_min) * u.min
    
    # Broadcast t0 (N, 1) + offsets (1, T) -> time_grid (N, T)
    time_grid = t0[:, np.newaxis] + time_offsets[np.newaxis, :]

    # 2. Vectorized darkness check: returns shape (N, T)
    A_dark = obs.is_night(time_grid, horizon=-18 * u.deg)

    # 3. Vectorized AltAz check
    # Broadcast coordinates to shape (N, 1) so they align with the rows of time_grid
    coord = SkyCoord(ra=ra, dec=dec)[:, np.newaxis]
    target = FixedTarget(coord=coord, name="Target")
    
    # Calculate altitudes for all dates and times at once: returns shape (N, T)
    altitudes = obs.altaz(time_grid, target).alt
    A_high = altitudes >= elev_min

    # 4. Combine masks: returns shape (N, T)
    A_vis = A_high & A_dark

    # 5. Extract continuous blocks per date
    is_visible_arr = np.zeros(N, dtype=bool)
    duration_arr = np.full(N, np.nan)
    
    dur_threshold = duration.to(u.hour).value
    step_hr = step_min / 60.0

    # Iterate over the N rows (dates) to find continuous blocks.
    # Since N is just the number of ephemeris rows, this 1D loop over booleans is extremely fast.
    for i in range(N):
        row_vis = A_vis[i]
        
        # Detect state changes (True <-> False)
        padded = np.concatenate([[False], row_vis, [False]])
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts, stops = changes[::2], changes[1::2]
        
        if len(starts) > 0:
            # Calculate duration of all visible blocks for this date
            durs = (stops - starts) * step_hr
            max_dur = np.max(durs)
            
            if max_dur >= dur_threshold:
                is_visible_arr[i] = True
                duration_arr[i] = max_dur

    return is_visible_arr, duration_arr


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
    # We consider it "night" if it is not daylight and not civil twilight.
    if 'solar_presence' in ephem.colnames:
        is_night = ~np.isin(ephem['solar_presence'], ['*', 'C'])
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
    t_stop = t_start + 1.0  
    
    epochs = {
        'start': t_start.strftime('%Y-%m-%d %H:%M'),
        'stop': t_stop.strftime('%Y-%m-%d %H:%M'),
        'step': '30m'  
    }
    
    try:
        # Use the imported function from query.py instead of calling Horizons directly
        ephem = fetch_with_fallback(target=target_name, location=location, epochs=epochs)
        
        # Guard clause in case the query fails or returns None
        if ephem is None or len(ephem) == 0:
            return False, None, None
            
        return _get_best_obstime(ephem, min_alt=min_alt)
        
    except Exception as e:
        print(f"Error processing visibility for {target_name}: {e}")
        return False, None, None