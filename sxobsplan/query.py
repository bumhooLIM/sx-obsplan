#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query.py — Library for batch-fetching JPL Horizons observer ephemerides.

Public API
----------
- batch_query(designations, epochs, location, quantities, ...)
- fetch_with_fallback(designation, epochs, location, quantities, ...)
- read_designations(input_csv, column="pdes")

Notes
-----
- `epochs` is a dict like {"start": "YYYY-MM-DD", "stop": "YYYY-MM-DD", "step": "3d"}.
- `location` is an observer code (e.g., '500' geocenter, 'T15' Gemini North, 'I11' Gemini South).
- `quantities` is a comma-separated string per Horizons manual (e.g., "1,9,19,20").

Example (library use)
---------------------
from query import batch_query, read_designations, fetch_with_fallback

epochs = {"start": "2026-01-01", "stop": "2026-12-31", "step": "2d"}
designations = read_designations("./data/sbdb_query_results_allcomet.csv")

# High-Performance Batch Fetch (Concurrent)
# Fetches data for many objects simultaneously using multiple threads
eph_table = batch_query(
    designations,
    epochs=epochs,
    quantities="1,3,9,18,19,20,24",
    location="500",
    max_workers=8,     # Number of parallel requests (default 5)
    progress=True,     # Show progress bar
    limit=100          # Only test the first 100 targets
)

# Single Object Fetch (Sequential)
one_target = fetch_with_fallback(
    "1P",
    epochs=epochs,
    quantities="1,9",
    location="500"
)
"""

from __future__ import annotations

import logging
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Union, Any, Tuple

import pandas as pd
from tqdm import tqdm
from astropy.table import Table, vstack
from astroquery.jplhorizons import Horizons

__all__ = [
    "batch_query",
    "fetch_with_fallback",
    "read_designations",
]

# -------------------------
# Logging Configuration
# -------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _extract_last_record_number(error_message: str) -> Optional[str]:
    """
    Parse a Horizons error message to get the last 'Record #' token (8-digit).
    Returns None if not found.
    """
    candidates: List[str] = []
    for line in error_message.splitlines():
        s = line.strip()
        if re.match(r"^\d{8}\b", s):
            candidates.append(s)
            
    if not candidates:
        return None
    return candidates[-1].split()[0]


def _fetch_ephemerides(
    target_id: str,
    epochs: Dict[str, str],
    location: str,
    *,
    quantities: Optional[Union[str, int]] = None,
    max_retries: int = 2,
    sleep_s: float = 1.0,
    id_type: Optional[str] = None,
) -> Table:
    """
    Low-level fetch with simple retry/backoff.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            obj = Horizons(
                id=target_id, 
                id_type=id_type, 
                location=location, 
                epochs=epochs
            )
            return obj.ephemerides(quantities=quantities) if quantities else obj.ephemerides()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(sleep_s)
            else:
                raise
                
    assert last_exc is not None
    raise last_exc


def read_designations(input_csv: Union[str, Path], desig_col: str = "pdes") -> List[str]:
    """
    Read a CSV and return the list of primary designations (as strings).
    """
    df = pd.read_csv(input_csv)
    if desig_col not in df.columns:
        raise ValueError(f"Column '{desig_col}' not found in {input_csv}. Columns: {list(df.columns)}")
    return df[desig_col].dropna().astype(str).tolist()


def fetch_with_fallback(
    target_id: str,
    epochs: Dict[str, str],
    location: str,
    *,
    quantities: Optional[Union[str, int]] = None,
    id_type: Optional[str] = None,
    max_retries: int = 2,
    sleep_s: float = 1.0,
) -> Optional[Table]:
    """
    Fetch ephemerides for a single designation. Fallback to 'Record #' if non-unique.
    """
    fetch_kwargs: Dict[str, Any] = {
        "epochs": epochs,
        "location": location,
        "quantities": quantities,
        "max_retries": max_retries,
        "sleep_s": sleep_s
    }
    
    try:
        return _fetch_ephemerides(target_id, id_type=id_type, **fetch_kwargs)
        
    except ValueError as e:
        record = _extract_last_record_number(str(e))
        if record:
            logger.info(f"Non-unique id for '{target_id}'. Retrying with record #{record} ...")
            try:
                return _fetch_ephemerides(record, id_type=None, **fetch_kwargs)
            except Exception as e2:
                logger.warning(f"Failed with record #{record} for '{target_id}': {e2}")
                return None
                
        logger.warning(f"No record id found in error for '{target_id}'. Skipping.")
        return None
        
    except Exception as e:
        logger.warning(f"Failed for '{target_id}': {e}")
        return None


def batch_query(
    designations: Iterable[str],
    epochs: Dict[str, str],
    location: str = "500",
    *,
    quantities: Optional[Union[str, int]] = None,
    limit: int = 0,
    progress: bool = False,
    start_index: int = 1,
    max_retries: int = 2,
    sleep_s: float = 1.0,
    max_workers: int = 5,
) -> Table:
    """
    Fetch and stack ephemerides for many designations concurrently.

    Parameters
    ----------
    designations : Iterable[str]
        Comet/asteroid primary designations.
    epochs : dict
        {'start': 'YYYY-MM-DD', 'stop': 'YYYY-MM-DD', 'step': '3d'}
    location : str
        Observer code (e.g., '500', 'T15', 'I11'). Default is '500'.
    quantities : str or int, optional
        Comma-separated Horizons quantities string.
    limit : int
        If >0, process only the first N designations (useful for testing).
    progress : bool
        Show a tqdm progress bar if True (default False).
    start_index : int
        Starting value for the 'ID' column (default 1).
    max_retries : int
        Max retries per target.
    sleep_s : float
        Seconds to sleep between retries.
    max_workers : int
        Number of parallel threads to use for querying JPL Horizons.
        Keep between 5-10 to avoid rate-limiting. Default is 5.

    Returns
    -------
    astropy.table.Table
        Stacked table containing results for all successfully queried targets.
    """
    items = list(designations)
    if limit > 0:
        items = items[:limit]

    # Helper function for the thread pool
    def _worker(i: int, name: str) -> Tuple[int, str, Optional[Table]]:
        tab = fetch_with_fallback(
            name,
            epochs=epochs,
            location=location,
            quantities=quantities,
            id_type=None,
            max_retries=max_retries,
            sleep_s=sleep_s
        )
        return i, name, tab

    results = []

    # Execute queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track their original order (i)
        futures = {executor.submit(_worker, i, name): (i, name) for i, name in enumerate(items)}
        
        # Setup progress bar
        iterator = concurrent.futures.as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=len(items), desc="Batch Querying (Parallel)")

        # Collect results as they finish
        for future in iterator:
            i, name, tab = future.result()
            
            if tab is not None and len(tab) > 0:
                tab["ID"] = start_index + i
                tab["Name"] = name
                results.append((i, tab))

    # Because threads complete out of order, sort by the original index 
    # to ensure the final table matches the input list's order.
    results.sort(key=lambda x: x[0])
    
    # Extract just the tables
    table_chunks = [r[1] for r in results]

    return vstack(table_chunks, metadata_conflicts="silent") if table_chunks else Table()