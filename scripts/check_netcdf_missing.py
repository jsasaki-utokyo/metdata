#!/usr/bin/env python3
"""
Inspect a CF NetCDF file created by `gwo_to_cf_netcdf.py` and list missing values.

For each variable that carries the `time` dimension, the script reports every
missing sample together with the closest previous and next valid values to help
diagnose gaps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


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


from typing import Tuple


def missing_report(
    var_name: str,
    data: np.ndarray,
    time_index: pd.DatetimeIndex,
    limit: int,
) -> Tuple[pd.DataFrame, int]:
    masked = np.ma.array(data)
    fill_value = getattr(data, "fill_value", np.nan)
    display_values = masked.filled(fill_value)
    numeric_values = masked.filled(np.nan)
    series = pd.Series(numeric_values, index=time_index, dtype="float64")
    missing_mask = series.isna()
    if not missing_mask.any():
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
        },
        index=time_index,
    )
    missing_df = df[series.isna()].copy()
    total_missing = missing_df.shape[0]
    if limit and limit > 0:
        missing_df = missing_df.iloc[:limit]
    missing_df.index.name = "time"
    return missing_df, total_missing


def stringify_timestamp(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "N/A"
    return ts.isoformat()


def print_report(df: pd.DataFrame) -> None:
    for time, row in df.iterrows():
        print(
            f"{stringify_timestamp(time)} | {row['variable']}: value={row['value']} | "
            f"prev=({stringify_timestamp(row['prev_time'])}, {row['prev_value']}) | "
            f"next=({stringify_timestamp(row['next_time'])}, {row['next_value']})"
        )


def main() -> None:
    args = parse_args()
    if not args.netcdf.exists():
        raise SystemExit(f"{args.netcdf} does not exist.")

    with Dataset(args.netcdf, mode="r") as ds:
        time_index = load_time_index(ds, args.time_var, args.timezone)
        names = args.variables or candidate_variables(ds, args.time_var)
        if not names:
            print("No variables with the time dimension were found.", file=sys.stderr)
            return

        for name in names:
            var = ds.variables.get(name)
            if var is None:
                print(f"[WARN] Variable '{name}' not found; skipping.")
                continue
            if args.time_var not in var.dimensions:
                print(f"[WARN] Variable '{name}' lacks '{args.time_var}' dimension; skipping.")
                continue
            data = var[:]
            report, total_missing = missing_report(name, data, time_index, args.limit)
            if total_missing == 0:
                print(f"{name}: no missing values.")
                continue
            print(f"{name}: showing up to {args.limit} of {total_missing} missing samples")
            print_report(report)
            print()


if __name__ == "__main__":
    main()
