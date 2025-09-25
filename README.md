
# sxobsplan

**sxobsplan** is a Python library for planning follow-up observations of Solar System Objects (SSOs), especially in support of the **SPHEREx** mission.  
It provides utilities to batch-fetch ephemerides from JPL Horizons and prepare them for downstream analysis.

---

## Features

- Query **JPL Horizons** for many comets/asteroids in batch.
- Automatic fallback for periodic comets with ambiguous designations (uses last `Record #`).
- Append object ID and name to each ephemeris table.
- Output results as a single stacked `astropy.table.Table` (and easily export to CSV).
- Clean API with three main functions:
  - `batch_query`
  - `fetch_with_fallback`
  - `read_designations`

---

## Installation

Clone the repository and install in **editable** mode:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/sx-obsplan.git
cd sx-obsplan
pip install -e .
