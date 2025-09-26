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

__all__ = [
    "is_target_visible",
    "resolve_location",
]

LocationLike = Union[EarthLocation, str]  # EarthLocation object or MPC/IAU code

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
            return EarthLocation.of_site(location)
        except Exception as e:
            raise ValueError(f"Could not resolve observatory code '{location}': {e}")

    raise TypeError("location must be EarthLocation or str (MPC/IAU observatory code).")


def _normalize_date_to_noon_utc(date) -> Time:
    """
    Convert input to Time in UTC; if it's a bare date (00:00),
    shift to 12:00 UTC to ensure proper twilight bracketing.
    """
    t0 = date if isinstance(date, Time) else Time(date)
    t0 = t0.utc

    midnight = Time(t0.iso.split(" ")[0] + " 00:00", scale="utc")
    if abs((t0 - midnight).to(u.second).value) < 1.0:
        return Time(f"{t0.utc.iso.split(' ')[0]} 12:00", scale="utc")
    return t0


def is_target_visible(
    ra: u.Quantity,                       # Right ascension [angle], e.g., 120*u.deg or 8*u.hourangle
    dec: u.Quantity,                      # Declination [angle]
    date,                                 # Time | datetime/date | str (UTC)
    location: LocationLike,               # EarthLocation | MPC code
    *,
    height_min: u.Quantity = 30 * u.deg,  # Minimum altitude
    duration:  u.Quantity = 1 * u.hour,   # Required continuous observing time
    dt_step:   u.Quantity = 2 * u.min,    # Sampling cadence
    return_block: bool = True,            # Whether to return visibility blocks
):
    """
    Determine whether a target is observable for at least `duration`
    above `height_min` during the astronomical night bracketing `date`.

    Parameters
    ----------
    ra, dec : Quantity[angle]
        Target coordinates (RA may be in hourangle).
    date : Time | datetime/date | str
        Date or datetime (UTC). If only a date is given (00:00),
        internally shifted to 12:00 UTC.
    location : EarthLocation | str
        Observatory location. If string, treated as MPC/IAU observatory code.
    height_min : Quantity[angle], default 30 deg
        Minimum altitude threshold.
    duration : Quantity[time], default 1 hour
        Minimum continuous visibility required.
    dt_step : Quantity[time], default 2 min
        Altitude sampling cadence.
    return_block : bool, default True
        If True, return (is_visible, blocks).
        If False, return only is_visible (bool).

    Returns
    -------
    bool or (bool, list[dict])
        - If return_block=False → only `is_visible` (bool).
        - If return_block=True  → (is_visible, blocks),
          where blocks is a list of dicts:
            {"start": Time, "end": Time, "duration": Quantity[time]}.
    """
    t0 = _normalize_date_to_noon_utc(date)
    loc = resolve_location(location)
    coord = SkyCoord(ra=ra, dec=dec)

    # Observer and target
    obs = Observer(location=loc, name=str(location), timezone="UTC")
    target = FixedTarget(coord=coord, name="Target")

    # Twilight bounds
    try:
        e_twilight = obs.twilight_evening_astronomical(t0, which="next")
        m_twilight = obs.twilight_morning_astronomical(e_twilight, which="next")
        if (e_twilight is None) or (m_twilight is None):
            return (False, []) if return_block else False
    except Exception:
        return (False, []) if return_block else False

    # Build time grid
    step_min = max(1, int(np.floor(dt_step.to(u.min).value)))
    time_grid = t0 + np.arange(0, 24 * 60, step_min) * u.min

    # Altitudes & masks
    altitudes = obs.altaz(time_grid, target).alt
    A_high = altitudes >= height_min
    A_dark = (time_grid >= e_twilight) & (time_grid <= m_twilight)
    A_vis = A_high & A_dark

    # Detect blocks
    padded = np.concatenate([[False], A_vis, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]

    blocks = []
    for s, e in zip(starts, stops):
        if e - s <= 1:
            continue
        dur = (time_grid[e - 1] - time_grid[s]).to(u.hour)
        if dur >= duration:
            blocks.append({
                "start": time_grid[s],
                "end":   time_grid[e - 1],
                "duration": dur
            })

    is_visible = len(blocks) > 0
    return (is_visible, blocks) if return_block else is_visible
