#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query.py — Library for batch-fetching JPL Horizons observer ephemerides.

Public API
----------
- batch_query(designations, epochs, quantities, location, ...)
- fetch_with_fallback(designation, epochs, quantities, location, ...)
- read_designations(input_csv, column="pdes")

Notes
-----
- `epochs` is a dict like {"start": "YYYY-MM-DD", "stop": "YYYY-MM-DD", "step": "3d"}.
- `location` is an observer code (e.g., '500' geocenter, 'T15' Gemini North, 'I11' Gemini South).
- `quantities` is a comma-separated string per Horizons manual, e.g.:
  "1,3,9,18,19,20,22,23,24,25,27,28,29,33,43"

Example (library use)
---------------------
from query import batch_query, read_designations, fetch_with_fallback

epochs = {"start": "2025-08-01", "stop": "2026-01-31", "step": "3d"}
designations = read_designations("./__data__/sbdb_query_results_allcomet_ver25B.csv")

# Many objects
eph = batch_query(
    designations,
    epochs=epochs,
    quantities="1,3,9,18,19,20,22,23,24,25,27,28,29,33,43",
    location="500",
    limit=5
)

# One object
one = fetch_with_fallback(
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
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Union

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
# Logging
# -------------------------
logger = logging.getLogger("query")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

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
    return candidates[-1].split()[0]  # first token on the last matching line


def _fetch_ephemerides(
    target_id: str,
    id_type: Optional[str],
    epochs: Dict[str, str],
    quantities: str,
    location: str,
    *,
    max_retries: int = 2,
    sleep_s: float = 1.0,
) -> Table:
    """
    Low-level fetch with simple retry/backoff.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            obj = Horizons(id=target_id, id_type=id_type, location=location, epochs=epochs)
            return obj.ephemerides(quantities=quantities)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(sleep_s)
            else:
                raise
    assert last_exc is not None
    raise last_exc

def read_designations(input_csv: Union[str, Path], column: str = "pdes") -> List[str]:
    """
    Read a CSV and return the list of primary designations (as strings).

    Parameters
    ----------
    input_csv : str | Path
        Path to input CSV.
    column : str
        Column name containing designations (default: 'pdes').

    Returns
    -------
    list[str]
    """
    df = pd.read_csv(input_csv)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {input_csv}. Columns: {list(df.columns)}")
    return df[column].dropna().astype(str).tolist()

def fetch_with_fallback(
    designation: str,
    epochs: Dict[str, str],
    quantities: str,
    location: str,
    *,
    max_retries: int = 2,
    sleep_s: float = 1.0,
) -> Optional[Table]:
    """
    Fetch ephemerides for a single designation. If Horizons raises a ValueError
    due to non-unique solutions (common for periodic comets), the function tries
    to parse the last 'Record #' from the error and retries with that numeric ID.

    Returns
    -------
    astropy.table.Table | None
        Ephemerides table, or None if both attempts fail.
    """
    try:
        return _fetch_ephemerides(
            designation, "designation", epochs, quantities, location, max_retries=max_retries, sleep_s=sleep_s
        )
    except ValueError as e:
        record = _extract_last_record_number(str(e))
        if record:
            logger.info(f"Non-unique id for '{designation}'. Retrying with record #{record} ...")
            try:
                return _fetch_ephemerides(
                    record, None, epochs, quantities, location, max_retries=max_retries, sleep_s=sleep_s
                )
            except Exception as e2:
                logger.warning(f"Failed with record #{record} for '{designation}': {e2}")
                return None
        logger.warning(f"No record id found in error for '{designation}'. Skipping.")
        return None
    except Exception as e:
        logger.warning(f"Failed for '{designation}': {e}")
        return None

def batch_query(
    designations: Iterable[str],
    epochs: Dict[str, str],
    quantities: str,
    location: str = "500",
    *,
    limit: int = 0,
    progress: bool = True,
    start_index: int = 1,
    max_retries: int = 2,
    sleep_s: float = 1.0,
) -> Table:
    """
    Fetch and stack ephemerides for many designations.

    Parameters
    ----------
    designations : Iterable[str]
        Comet/asteroid primary designations.
    epochs : dict
        {'start': 'YYYY-MM-DD', 'stop': 'YYYY-MM-DD', 'step': '3d'}
    quantities : str
        Comma-separated Horizons quantities string.
    location : str
        Observer code (e.g., '500', 'T15', 'I11').
    limit : int
        If >0, process only the first N designations (useful for testing).
    progress : bool
        Show a tqdm progress bar if True.
    start_index : int
        Starting value for the 'ID' column (default 1).
    max_retries : int
        Max retries per target.
    sleep_s : float
        Seconds to sleep between retries.

    Returns
    -------
    astropy.table.Table
        Stacked table (empty if nothing succeeded).
    """
    items = list(designations)
    if limit > 0:
        items = items[:limit]

    iterator = enumerate(items, start=0)
    if progress:
        iterator = tqdm(iterator, total=len(items), desc="Querying ephemerides...")

    out_table: Optional[Table] = None
    for i, name in iterator:
        tab = fetch_with_fallback(
            name, epochs=epochs, quantities=quantities, location=location,
            max_retries=max_retries, sleep_s=sleep_s
        )
        if tab is None or len(tab) == 0:
            continue

        tab["ID"] = start_index + i   # 1-based index by default
        tab["Name"] = name

        out_table = tab if out_table is None else vstack(
            [out_table, tab], metadata_conflicts="silent"
        )

    return out_table if out_table is not None else Table()
