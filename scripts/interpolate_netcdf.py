#!/usr/bin/env python3
"""
Interpolate missing values in CF-compliant NetCDF files.

This script reads a NetCDF file, identifies gaps in time series variables,
and fills them using interpolation when the gap size is within the specified
maximum allowable continuous missing values. Longer gaps trigger alerts.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

LOG = logging.getLogger("interpolate_netcdf")

# Valid pandas interpolation methods
VALID_METHODS = [
    "linear",
    "time",
    "index",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate missing values in NetCDF time series variables. "
            "Only fills gaps up to a specified maximum continuous length."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interpolate with default settings (linear, max_gap=3)
  %(prog)s tokyo_2019.nc

  # Use time-based interpolation with max gap of 5
  %(prog)s tokyo_2019.nc --method time --max-gap 5

  # Save to a different output file
  %(prog)s tokyo_2019.nc --output tokyo_2019_interp.nc

  # Use cubic interpolation for smoother results
  %(prog)s tokyo_2019.nc --method cubic --max-gap 2

Available interpolation methods:
  linear (default) - Linear interpolation
  time             - Time-weighted interpolation
  cubic            - Cubic spline interpolation
  nearest          - Nearest neighbor
  And more: quadratic, spline, pchip, akima, etc.
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input NetCDF file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output NetCDF file path (default: overwrite input file)",
    )
    parser.add_argument(
        "--method",
        "-m",
        default="linear",
        choices=VALID_METHODS,
        help="Interpolation method (default: linear)",
    )
    parser.add_argument(
        "--max-gap",
        "-g",
        type=int,
        default=3,
        help=(
            "Maximum number of continuous missing values to interpolate. "
            "Gaps larger than this will trigger alerts and remain unfilled. "
            "(default: 3)"
        ),
    )
    parser.add_argument(
        "--time-var",
        default="time",
        help="Name of the time coordinate variable (default: time)",
    )
    parser.add_argument(
        "--variables",
        "-v",
        nargs="+",
        help="Specific variables to interpolate (default: all time-dependent variables)",
    )
    parser.add_argument(
        "--backup",
        "-b",
        action="store_true",
        help="Create backup of input file before overwriting (adds .bak extension)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Suppress alerts for long gaps (>max-gap)",
    )
    return parser.parse_args()


def load_time_index(ds: Dataset, var_name: str) -> pd.DatetimeIndex:
    """Load time variable as pandas DatetimeIndex."""
    if var_name not in ds.variables:
        raise ValueError(f"Time variable '{var_name}' not found in dataset.")
    time_var = ds.variables[var_name]
    times = num2date(
        time_var[:],
        units=getattr(time_var, "units", "seconds since 1970-01-01 00:00:00 UTC"),
        calendar=getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return pd.to_datetime(times)


def get_time_variables(ds: Dataset, time_dim: str) -> List[str]:
    """Find all variables that depend on the time dimension."""
    names: List[str] = []
    for name, var in ds.variables.items():
        if name == time_dim:
            continue
        if time_dim in var.dimensions and len(var.dimensions) == 1:
            names.append(name)
    return names


def find_gaps(series: pd.Series) -> List[Tuple[int, int, int]]:
    """
    Find all continuous gaps (missing value sequences) in a series.

    Returns:
        List of tuples (start_idx, end_idx, gap_length)
    """
    is_missing = series.isna()
    if not is_missing.any():
        return []

    gaps: List[Tuple[int, int, int]] = []
    in_gap = False
    gap_start = 0

    for i, missing in enumerate(is_missing):
        if missing and not in_gap:
            # Start of a new gap
            gap_start = i
            in_gap = True
        elif not missing and in_gap:
            # End of gap
            gap_length = i - gap_start
            gaps.append((gap_start, i - 1, gap_length))
            in_gap = False

    # Handle gap extending to end of series
    if in_gap:
        gap_length = len(series) - gap_start
        gaps.append((gap_start, len(series) - 1, gap_length))

    return gaps


def interpolate_with_max_gap(
    series: pd.Series,
    method: str,
    max_gap: int,
    time_index: pd.DatetimeIndex | None = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Interpolate a series but only fill gaps up to max_gap length.

    Args:
        series: Series with missing values
        method: Interpolation method
        max_gap: Maximum gap length to fill
        time_index: DatetimeIndex for time-based interpolation

    Returns:
        Tuple of (interpolated_series, stats_dict)
    """
    if not series.isna().any():
        return series.copy(), {"total_missing": 0, "filled": 0, "unfilled": 0}

    gaps = find_gaps(series)
    total_missing = sum(gap[2] for gap in gaps)
    fillable_gaps = [g for g in gaps if g[2] <= max_gap]
    unfillable_gaps = [g for g in gaps if g[2] > max_gap]

    # Create a copy for interpolation
    result = series.copy()

    # Only interpolate if there are fillable gaps
    if fillable_gaps:
        # For gaps larger than max_gap, temporarily fill with a marker
        # so they won't be interpolated
        MARKER = -9.99999e35
        for start_idx, end_idx, _ in unfillable_gaps:
            result.iloc[start_idx : end_idx + 1] = MARKER

        # Perform interpolation
        if method == "time" and time_index is not None:
            result_with_time = pd.Series(result.values, index=time_index)
            result_with_time = result_with_time.interpolate(method=method, limit_area="inside")
            result = pd.Series(result_with_time.values, index=series.index)
        else:
            result = result.interpolate(method=method, limit_area="inside")

        # Restore NaN for unfillable gaps
        for start_idx, end_idx, _ in unfillable_gaps:
            result.iloc[start_idx : end_idx + 1] = np.nan

    filled = sum(gap[2] for gap in fillable_gaps)
    unfilled = sum(gap[2] for gap in unfillable_gaps)

    return result, {
        "total_missing": total_missing,
        "filled": filled,
        "unfilled": unfilled,
        "gaps": gaps,
        "fillable_gaps": fillable_gaps,
        "unfillable_gaps": unfillable_gaps,
    }


def format_timestamp(dt: pd.Timestamp) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def interpolate_netcdf(
    input_path: Path,
    output_path: Path,
    method: str,
    max_gap: int,
    time_var: str,
    target_vars: List[str] | None,
    show_alerts: bool,
) -> None:
    """
    Interpolate missing values in NetCDF file.

    Args:
        input_path: Input NetCDF file
        output_path: Output NetCDF file
        method: Interpolation method
        max_gap: Maximum gap size to fill
        time_var: Name of time variable
        target_vars: Variables to interpolate (None = all)
        show_alerts: Whether to show alerts for long gaps
    """
    # Load data
    LOG.info("Loading NetCDF file: %s", input_path)
    with Dataset(input_path, "r") as ds:
        time_index = load_time_index(ds, time_var)
        all_vars = get_time_variables(ds, time_var)

        if target_vars:
            # Validate requested variables
            invalid = set(target_vars) - set(all_vars)
            if invalid:
                raise ValueError(
                    f"Variables not found or not time-dependent: {', '.join(invalid)}"
                )
            vars_to_process = target_vars
        else:
            vars_to_process = all_vars

        if not vars_to_process:
            LOG.warning("No time-dependent variables found to interpolate.")
            return

        LOG.info("Variables to interpolate: %s", ", ".join(vars_to_process))
        LOG.info("Interpolation method: %s", method)
        LOG.info("Maximum gap size: %d", max_gap)

        # Load and interpolate each variable
        interpolated_data: Dict[str, np.ndarray] = {}
        total_filled = 0
        total_unfilled = 0

        for var_name in vars_to_process:
            var = ds.variables[var_name]
            data = var[:]

            # Convert masked array to series with NaN
            if hasattr(data, "filled"):
                numeric_data = data.filled(np.nan)
            else:
                numeric_data = np.array(data, dtype=np.float64)

            series = pd.Series(numeric_data, index=time_index)

            # Interpolate
            use_time_index = time_index if method == "time" else None
            result, stats = interpolate_with_max_gap(
                series, method, max_gap, use_time_index
            )

            interpolated_data[var_name] = result.values

            if stats["total_missing"] > 0:
                LOG.info(
                    "  %s: %d missing → %d filled, %d unfilled",
                    var_name,
                    stats["total_missing"],
                    stats["filled"],
                    stats["unfilled"],
                )
                total_filled += stats["filled"]
                total_unfilled += stats["unfilled"]

                # Show alerts for unfillable gaps
                if show_alerts and stats["unfillable_gaps"]:
                    for start_idx, end_idx, gap_len in stats["unfillable_gaps"]:
                        start_time = time_index[start_idx]
                        end_time = time_index[end_idx]
                        LOG.warning(
                            "    ALERT: Gap too long (%d > %d) in %s from %s to %s",
                            gap_len,
                            max_gap,
                            var_name,
                            format_timestamp(start_time),
                            format_timestamp(end_time),
                        )
            else:
                LOG.info("  %s: no missing values", var_name)

    # Write output
    LOG.info("Writing interpolated data to: %s", output_path)

    # If input and output are the same, modify in place
    # Otherwise, copy input to output first
    if input_path.resolve() != output_path.resolve():
        shutil.copy2(input_path, output_path)

    # Update variables with interpolated data
    with Dataset(output_path, "r+") as ds:
        for var_name, data in interpolated_data.items():
            # Convert NaN back to masked array with fill value
            var = ds.variables[var_name]
            fill_value = getattr(var, "_FillValue", None) or getattr(var, "missing_value", np.nan)

            mask = ~np.isfinite(data)
            masked_data = np.ma.masked_array(data, mask=mask)

            var[:] = masked_data

        # Update history
        history_entry = (
            f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} - "
            f"Interpolated using scripts/interpolate_netcdf.py "
            f"(method={method}, max_gap={max_gap}, filled={total_filled}, unfilled={total_unfilled})"
        )
        existing_history = getattr(ds, "history", "")
        if existing_history:
            ds.history = f"{history_entry}\n{existing_history}"
        else:
            ds.history = history_entry

    LOG.info("Done. Total: %d filled, %d unfilled", total_filled, total_unfilled)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )

    # Validate input file
    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    # Determine output file
    if args.output:
        output_path = args.output
    else:
        output_path = args.input
        if args.backup:
            backup_path = args.input.with_suffix(args.input.suffix + ".bak")
            LOG.info("Creating backup: %s", backup_path)
            shutil.copy2(args.input, backup_path)

    # Validate max_gap
    if args.max_gap < 1:
        raise SystemExit("--max-gap must be at least 1")

    # Perform interpolation
    try:
        interpolate_netcdf(
            input_path=args.input,
            output_path=output_path,
            method=args.method,
            max_gap=args.max_gap,
            time_var=args.time_var,
            target_vars=args.variables,
            show_alerts=not args.no_alerts,
        )
    except Exception as e:
        LOG.error("Error: %s", e)
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
