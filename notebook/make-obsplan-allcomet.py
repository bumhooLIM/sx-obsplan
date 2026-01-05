import sxobsplan

from pathlib import Path
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import _rcparams

# Ephemeris
EPHDIR = Path("../eph")

# # Small-Body Data Source: JPL SBDB Query
# # https://ssd.jpl.nasa.gov/tools/sbdb_query.html
# fpath_sbdb = Path("../data/sbdb_query_results_comet_ver260105.csv")
# df_sbdb = pd.read_csv(fpath_sbdb)

# # Filter database before queries
# mask  = (df_sbdb.prefix != "D") # Remove destroyed comets
# mask &= ~((df_sbdb.prefix == "C") & (df_sbdb.epoch_cal <= "2015-01-01")) # Old comets

# df_sbdb_filtered = df_sbdb[mask]
# df_sbdb_filtered.head()

# Save result directory
VISDIR = Path("../visibility_ver260105_allcomet")
VISDIR.mkdir(exist_ok=True, parents=True)

# SBDB
fpath_sbdb = Path("../data/sbdb_query_results_comet_ver260105.csv") # SBDB comet query results (latest ver)
df_sbdb = pd.read_csv(fpath_sbdb)
# df_sbdb.head()

# Observatory locations

# LSGT (Siding Spring Observatory, Australia)
siding_spring_obs = EarthLocation(
    lon=149.0644 * u.deg,  # (149° 03' 52" E)
    lat=-31.2733 * u.deg,  # (31° 16' 24" S)
    height=1165 * u.m
)

# 7DT (el Sauce Observatory, Chile)
el_sauce_obs = EarthLocation(
    lon=-70.763 * u.deg,
    lat=-30.4725 * u.deg,
    height=1600 * u.m
)

# LOAO (Mount Lemmon Observatory, USA)
lemmon_obs = EarthLocation(
    lon=-110.7893 * u.deg,
    lat=32.4420 * u.deg,
    height=2791 * u.m
)

dict_observatory = {
    "gemini_north": "gemini_north",
    "gemini_south": "gemini_south",
    "lemmon"      : lemmon_obs,
    "lsgt"        : siding_spring_obs,
    "sevendt"     : el_sauce_obs
}

for obsname in dict_observatory:
    print(obsname)
    print(sxobsplan.resolve_location(dict_observatory[obsname]))
    
for idx, row in df_sbdb.iterrows():

    pdes = row.pdes
    fpath_eph = EPHDIR / f"{''.join(pdes.split())}.csv"
    fpath_vis = VISDIR / f"{''.join(pdes.split())}.csv"

    if not fpath_eph.exists():
        print(f"{fpath_eph.name} does not exists. Skip the rows.")
        continue

    # target info.
    target = df_sbdb[df_sbdb["pdes"] == pdes].iloc[0]

    # ephemeris info.
    eph = pd.read_csv(fpath_eph)

    # Skip if all eph.r > 9 or all eph.Tmag > 22
    if (eph.r > 10).all() or (eph.Tmag > 22).all():
        print(f"{fpath_eph.name}: all r > 9 or all Tmag > 22. Skip the rows.")
        continue

    # Visibility
    df_visible = pd.DataFrame()

    for idx, row in eph.iterrows():

        df_visible.at[idx, "pdes"] = target["pdes"]
        df_visible.at[idx, "full_name"] = target["full_name"]
        df_visible.at[idx, "name"] = target["name"]
        df_visible.at[idx, "class"] = target["class"].upper()

        df_visible.at[idx, "datetime_jd"] = row.datetime_jd
        df_visible.at[idx, "ra"] = row.RA
        df_visible.at[idx, "dec"] = row.DEC
        df_visible.at[idx, "r"] = row.r
        df_visible.at[idx, "delta"] = row.delta
        df_visible.at[idx, "tmag"] = row.Tmag
        df_visible.at[idx, "elong"] = row.elong
        df_visible.at[idx, "lunar_elong"] = row.lunar_elong
        df_visible.at[idx, "glxlat"] = row.GlxLat
        df_visible.at[idx, "alpha"] = row.alpha
        
        for observatory, location in dict_observatory.items():

            is_visible, blocks = sxobsplan.is_target_visible(
                ra=row.RA*u.deg, dec=row.DEC*u.deg, date=Time(row.datetime_jd, format="jd"),
                location=location,
                elev_min=30*u.deg, duration=1*u.hour
            )

            df_visible.at[idx, f"is_visible_{observatory}"] = is_visible

            if is_visible:
                df_visible.at[idx, f"duration_{observatory}"] = blocks[0]["duration"].to_value("hour")
            else:
                df_visible.at[idx, f"duration_{observatory}"] = None

    df_visible.to_csv(fpath_vis)
    print(f"Visibility saved. {fpath_vis}")