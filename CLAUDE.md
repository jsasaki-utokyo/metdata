# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**metdata** is a Python library for processing meteorological data from the Japan Meteorological Agency (JMA), specifically:
- **GWO (Ground Weather Observation)** dataset from Japan Meteorological Business Support Center (JMBSC)
- **whpView** data (JMA-compatible format) from weather observation stations

The library provides data extraction, unit conversion, missing value handling, time series interpolation, visualization, and export capabilities to various formats including GOTM (General Ocean Turbulence Model) input files.

## Installation and Setup

Install in development mode (editable install):
```bash
pip install -e .
```

Requirements: `numpy`, `pandas`, `matplotlib`, `jupyterlab`

**External System Dependencies:**
- `nkf` (Network Kanji Filter) - Command-line tool for CSV encoding conversion to UTF-8 with Linux LF line endings

**Environment Variables (Linux only):**
- `$DATA_DIR` - **Required on Linux systems**. Points to the base data directory.
  ```bash
  export DATA_DIR=/mnt/c/Data
  ```
  The library will automatically construct paths:
  - GWO data: `$DATA_DIR/met/JMA_DataBase/GWO/`
  - whpView data: `$DATA_DIR/met/JMA_DataBase/whpView/wwwRoot/data/`

  **Note**: On non-Linux systems, default Windows-style paths are used instead.

**Version Information:**
- Package version is defined in `metdata/__init__.py` as `__version__`

**Testing:**
- No automated tests are included in this repository
- Testing is done through example scripts in the `examples/` directory

## Architecture

### Module Structure

The codebase consists of two main modules in `metdata/`:

1. **`gwo.py`** - Core module for GWO dataset processing
2. **`whp.py`** - Conversion module for whpView (JMA-compatible format) to GWO format

### Key Classes and Inheritance Hierarchy

#### gwo.py

```
Met (base class)
├── Hourly (main class for hourly data extraction)
│   ├── Check (debugging/validation class)
│   └── Daily (for daily data - under construction)
├── Stn (station metadata)
├── Data1D / Data1Ds (data organization for plotting)
├── Data1D_PlotConfig (plot configuration)
└── Plot1D (visualization)
```

**Met** (base class in gwo.py:76-199):
- Provides shared functionality for time series extraction
- Handles missing value detection using RMK (remark) codes
- `set_missing_values()` method replaces values with `np.nan` based on RMK columns

**Hourly** (main class in gwo.py:201-655):
- Extracts hourly meteorological data from yearly CSV files at each station
- Data format: Station files organized as `{dirpath}{stn}/{stn}{year}.csv`
- Handles two time intervals:
  - Before 1991: 3-hour intervals
  - After 1990: 1-hour intervals
- Properties:
  - `df_org`: Original data with missing rows filled
  - `df_interp`: Data at original intervals with interpolation
  - `df`: Fully processed 1-hour interval data (recommended)
- Critical preprocessing: GWO CSV files must be created using [GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD) tool first

**Stn** (gwo.py:37-74):
- Manages station metadata (location, altitude, instrument heights)
- Reads from `gwo_stn.csv` (note: this file is referenced in code but not present in repo)

**Data structures for plotting** (gwo.py:908-1081):
- `Data1Ds/Data1D`: Organize scalar or vector time series data
- `Data1D_PlotConfig`: Configure plot appearance
- `Plot1D`: Generate and save matplotlib figures with rolling mean support

#### whp.py

**Whp** (whp.py:41-172):
- Inherits from `gwo.Hourly`
- Converts whpView (JMA-compatible) format to GWO format
- Handles special symbols in JMA data (see https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html)
- Data cleaning: removes/replaces symbols like `×`, `///`, `#`, `)`, `]`, `0+`, `10-`
- File organization: `{data_dir_base}{year}/{stn}_{year}.csv`

**StationDict** (whp.py:15-38):
- Dictionary-like container for multiple Whp instances
- Access pattern: `whp_data[station_name, year]`

### Data Column Definitions

GWO hourly data columns (see gwo.py:209-228 for full definitions):
- Pressure: `lhpa` (local), `shpa` (sea level)
- Temperature: `kion` (air), `humd` (dewpoint)
- Humidity: `rhum` (relative), `stem` (vapor pressure)
- Wind: `muki` (direction, 0-360°), `sped` (speed), `u`/`v` (vector components)
- Radiation: `slht` (global horizontal irradiance), `lght` (sunshine duration)
- Precipitation: `kous`
- Cloud: `clod` (cover), `tnki` (current weather)

Each meteorological variable has a corresponding RMK (remark) column indicating data quality.

### RMK (Remark) Codes

RMK values indicate data quality (see gwo.py:80-138):
- 0: Not created
- 1: Missing
- 2: Not observed
- 3: Estimated value / extreme value ≤ true value
- 4: Extreme value ≥ true value / regional data used
- 5: Contains estimated values / 24-hour average with missing data
- 6: No phenomenon occurred (precipitation, sunshine, etc.)
- 7: Extreme time is previous day
- 8: Normal observation (default for whpView conversions)
- 9: Extreme time is next day / automatic acquisition (pre-1990)

Missing value designation (gwo.py:267-275):
- Most variables: RMK 0, 1, 2 → `np.nan`
- `lght` and `slht`: Only RMK 0, 1 → `np.nan` (RMK=2 indicates nighttime, not missing)

### Critical Data Processing Details

**Time handling**:
- GWO uses hours 1-24; converted to Python datetime 0-23 (hour 24 → next day 00:00)
- Missing datetime rows are inserted with RMK=0 and values=`np.nan`
- Time interpolation: Uses pandas `interpolate(method='time')` for numeric columns

**Unit conversions** (gwo.py:554-591):
- All units converted to SI-like: hPa, °C, m/s, W/m², etc.
- Wind direction: Converted from 0-16 scale to degrees (anticlockwise from east)
- Radiation: 0.01 MJ/m²/h → W/m²
- Precipitation: 0.1 mm/h → m/s

**3-hour to 1-hour resampling** (gwo.py:593-654):
- Forward fill for categorical/RMK columns
- Time interpolation for continuous variables
- Creates uniform 1-hour datasets from pre-1991 3-hour data

### Export Functions

Two GOTM export functions (gwo.py:1084-1117):

**`export_gotm(file, df, var, fmt=None)`**:
- Exports single variable to GOTM input format
- Format: `{datetime} {value}\n`

**`export_gotm_meteo(file, df)`**:
- Exports standard meteorological variables: u, v, shpa, kion, humd, clod
- Fixed format: 10.5f for winds/temp/humidity, 12.5f for pressure

## Common Development Commands

### Running Examples

```bash
# Set up environment (Linux only)
export DATA_DIR=/mnt/c/Data

# Export GWO data to GOTM format (copy sample first)
cp examples/export_gwo_gotm.sample.py export_gwo_gotm.py
# Edit export_gwo_gotm.py with your station/date parameters
python export_gwo_gotm.py
```

### Converting whpView to GWO Format

```python
from metdata.whp import Whp, StationDict

whp_data = StationDict()
whp_data['Tokyo', 2022] = Whp(year=2022, stn='Tokyo',
                               data_dir_base='/path/to/whpView/data/')
whp_data['Tokyo', 2022].to_csv(data_dir_base='/path/to/GWO/Hourly/')
```

## Important Implementation Notes

### Missing Data Handling Strategy

1. First check for missing rows (entire datetime not present)
2. Fill missing rows with fill_value=9999, then replace with `np.nan` and RMK=0
3. Then process RMK columns to identify missing values in existing rows
4. Apply interpolation only after all missing values are identified

The order matters: row completion must precede value interpolation.

### CSV File I/O

All CSV operations use UTF-8 encoding and Linux LF line endings:
```python
df.to_csv(path, encoding='utf-8')
subprocess.call(['nkf', '-w', '-Lu', '--overwrite', path])
```

**Note:** The `nkf` command is a system-level dependency. Ensure it's installed:
- Linux/WSL: `sudo apt install nkf`
- macOS: `brew install nkf`

### whpView Special Cases

When converting whpView data:
- `0+` in cloud cover → `0`
- `10-` in cloud cover → `10`
- Nighttime radiation/sunshine: Set to 0 with RMK=2
- Missing "現在天気" (current weather): Set to `np.NaN` with RMK=0
- Station ID adjustment: Subtract 47000 from whpView IDs

### Data Prerequisites

GWO data requires preprocessing with [GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD) to create yearly CSV files at each station before using this library.

### Typical Data Access Pattern

```python
from metdata import gwo

# Extract data for a period
# On Linux: Uses $DATA_DIR environment variable automatically
# On other platforms: Uses default Windows paths
met = gwo.Hourly(
    datetime_ini="2019-12-01 00:00:00",
    datetime_end="2020-12-30 00:00:00",
    stn="Tokyo"
    # dirpath parameter is optional - defaults to $DATA_DIR/met/JMA_DataBase/GWO/ on Linux
)

# Or specify custom path explicitly:
# met = gwo.Hourly(
#     datetime_ini="2019-12-01 00:00:00",
#     datetime_end="2020-12-30 00:00:00",
#     stn="Tokyo",
#     dirpath="/custom/path/to/GWO/"
# )

# Access processed 1-hour interval data
df = met.df  # Recommended: fully interpolated 1-hour data
# df_org = met.df_org  # Original intervals, missing rows filled
# df_interp = met.df_interp  # Original intervals, interpolated
```

## Data Attribution

All data from JMA requires attribution: "気象庁提供" (Provided by Japan Meteorological Agency)
