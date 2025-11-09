#!/usr/bin/env python3
"""
Inspect a CF NetCDF file created by `gwo_to_cf_netcdf.py` and list missing values.

For each variable that carries the `time` dimension, the script reports every
missing sample together with the closest previous and next valid values. When
the original GWO CSV files are available, the corresponding RMK code is shown.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

from metdata import gwo


CF_TO_RMK = {
    "air_pressure": "lhpaRMK",
    "air_pressure_at_sea_level": "shpaRMK",
    "air_temperature": "kionRMK",
    "dew_point_temperature": "humdRMK",
    "water_vapor_partial_pressure": "stemRMK",
    "relative_humidity": "rhumRMK",
    "wind_from_direction": "mukiRMK",
    "wind_speed": "spedRMK",
    "cloud_area_fraction": "clodRMK",
    "duration_of_sunshine": "lghtRMK",
    "surface_downwelling_shortwave_flux_in_air": "slhtRMK",
    "precipitation_flux": "kousRMK",
    "eastward_wind": "spedRMK",
    "northward_wind": "spedRMK",
}


class RawHourly(gwo.Hourly):
    """Hourly reader that bypasses unit conversion for RMK extraction."""

    def _unit_conversion(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report missing values inside a CF-compliant NetCDF file."
    )
    parser.add_argument(
        "netcdf",
        type=Path,
        help="Path to the NetCDF file (e.g., tokyo_2019.nc).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help="Subset of variables to inspect; defaults to all time-dependent variables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of missing samples to display per variable (default: 50).",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for reporting timestamps (e.g., Asia/Tokyo). Default: UTC.",
    )
    parser.add_argument(
        "--time-var",
        default="time",
        help="Name of the time coordinate variable (default: time).",
    )
    parser.add_argument(
        "--station",
        help="Station name override for RMK lookup (falls back to NetCDF global attribute).",
    )
    parser.add_argument(
        "--start",
        dest="start_datetime",
        help="Start datetime override for RMK lookup; defaults to NetCDF coverage.",
    )
    parser.add_argument(
        "--end",
        dest="end_datetime",
        help="End datetime override for RMK lookup; defaults to NetCDF coverage.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Directory containing station folders for RMK lookup "
            "(defaults to $DATA_DIR/met/JMA_DataBase/GWO/Hourly/)."
        ),
    )
    return parser.parse_args()


def load_time_index(ds: Dataset, var_name: str, timezone: str) -> pd.DatetimeIndex:
    if var_name not in ds.variables:
        raise ValueError(f"Time variable '{var_name}' not found.")
    time_var = ds.variables[var_name]
    times = num2date(
        time_var[:],
        units=getattr(time_var, "units", "seconds since 1970-01-01 00:00:00 UTC"),
        calendar=getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    timestamp_index = pd.to_datetime(times)
    timestamp_index = timestamp_index.tz_localize("UTC").tz_convert(timezone)
    return timestamp_index


def candidate_variables(ds: Dataset, time_dim: str) -> List[str]:
    names: List[str] = []
    for name, var in ds.variables.items():
        if name == time_dim:
            continue
        if time_dim in var.dimensions:
            names.append(name)
    return names


def stringify_timestamp(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "N/A"
    return ts.isoformat()


def resolve_hourly_dir(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        return explicit.expanduser()
    data_root = os.environ.get("DATA_DIR")
    if not data_root:
        return None
    return (
        Path(data_root).expanduser() / "met" / "JMA_DataBase" / "GWO" / "Hourly"
    )


def ensure_trailing_sep(path: Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else f"{text}{os.sep}"


def infer_station(args: argparse.Namespace, ds: Dataset) -> Optional[str]:
    if args.station:
        return args.station
    return getattr(ds, "station_id", None)


def infer_range(
    args: argparse.Namespace, ds: Dataset, time_index: pd.DatetimeIndex
) -> Tuple[str, str]:
    start = args.start_datetime or getattr(ds, "time_coverage_start", None)
    end = args.end_datetime or getattr(ds, "time_coverage_end", None)

    naive_index = time_index.tz_localize(None)
    if start is None:
        start = naive_index[0].strftime("%Y-%m-%d %H:%M:%S")
    if end is None:
        if len(naive_index) > 1:
            delta = naive_index[1] - naive_index[0]
        else:
            delta = pd.Timedelta(hours=1)
        end_ts = naive_index[-1] + delta
        end = end_ts.strftime("%Y-%m-%d %H:%M:%S")
    return start, end


def load_rmk_lookup(
    station: Optional[str],
    start: str,
    end: str,
    hourly_dir: Optional[Path],
) -> Dict[str, pd.Series]:
    if not station or hourly_dir is None or not hourly_dir.exists():
        return {}
    reader = RawHourly(
        datetime_ini=start,
        datetime_end=end,
        stn=station,
        dirpath=ensure_trailing_sep(hourly_dir),
    )
    df = reader.df_org.copy()
    if df.empty:
        return {}
    lookup: Dict[str, pd.Series] = {}
    for var, rmk_col in CF_TO_RMK.items():
        if rmk_col in df.columns:
            series = pd.Series(df[rmk_col].values, index=pd.DatetimeIndex(df.index))
            lookup[var] = series
    return lookup


def missing_report(
    var_name: str,
    data: np.ndarray,
    time_index: pd.DatetimeIndex,
    naive_index: pd.DatetimeIndex,
    limit: int,
    rmk_series: Optional[pd.Series],
) -> Tuple[pd.DataFrame, int]:
    masked = np.ma.array(data)
    fill_value = getattr(data, "fill_value", np.nan)
    display_values = masked.filled(fill_value)
    numeric_values = masked.filled(np.nan)
    series = pd.Series(numeric_values, index=time_index, dtype="float64")
    if not series.isna().any():
        return pd.DataFrame(), 0

    display_series = pd.Series(display_values, index=time_index)
    idx_series = pd.Series(time_index, index=time_index)
    prev_time = idx_series.where(series.notna()).ffill()
    next_time = idx_series.where(series.notna()).bfill()
    prev_value = series.ffill()
    next_value = series.bfill()

    df = pd.DataFrame(
        {
            "variable": var_name,
            "value": display_series,
            "prev_time": prev_time,
            "prev_value": prev_value,
            "next_time": next_time,
            "next_value": next_value,
            "naive_time": naive_index,
        },
        index=time_index,
    )
    missing_df = df[series.isna()].copy()
    total_missing = missing_df.shape[0]
    if total_missing == 0:
        return pd.DataFrame(), 0

    if rmk_series is not None:
        target_index = pd.DatetimeIndex(missing_df["naive_time"].values)
        aligned_rmks = rmk_series.reindex(target_index)
        missing_df["rmk"] = aligned_rmks.values
    else:
        missing_df["rmk"] = np.nan

    if limit and limit > 0:
        missing_df = missing_df.iloc[:limit]
    missing_df.index.name = "time"
    return missing_df.drop(columns=["naive_time"]), total_missing


def print_report(df: pd.DataFrame) -> None:
    for time, row in df.iterrows():
        msg = (
            f"{stringify_timestamp(time)} | {row['variable']}: value={row['value']} | "
            f"prev={row['prev_value']} | "
            f"next={row['next_value']} | "
            f"RMK={row.get('rmk', 'N/A')}"
        )
        print(msg)


def main() -> None:
    args = parse_args()
    if not args.netcdf.exists():
        raise SystemExit(f"{args.netcdf} does not exist.")

    with Dataset(args.netcdf, mode="r") as ds:
        time_index = load_time_index(ds, args.time_var, args.timezone)
        naive_index = time_index.tz_localize(None)
        names = args.variables or candidate_variables(ds, args.time_var)
        if not names:
            print("No variables with the time dimension were found.", file=sys.stderr)
            return

        hourly_dir = resolve_hourly_dir(args.data_dir)
        station = infer_station(args, ds)
        start, end = infer_range(args, ds, time_index)
        rmk_lookup = load_rmk_lookup(station, start, end, hourly_dir)

        for name in names:
            var = ds.variables.get(name)
            if var is None:
                print(f"[WARN] Variable '{name}' not found; skipping.")
                continue
            if args.time_var not in var.dimensions:
                print(f"[WARN] Variable '{name}' lacks '{args.time_var}' dimension; skipping.")
                continue
            data = var[:]
            rmk_series = rmk_lookup.get(name)
            report, total_missing = missing_report(
                name, data, time_index, naive_index, args.limit, rmk_series
            )
            if total_missing == 0:
                print(f"{name}: no missing values.")
                continue
            print(f"{name}: showing up to {args.limit} of {total_missing} missing samples")
            print_report(report)
            print()


if __name__ == "__main__":
    main()
