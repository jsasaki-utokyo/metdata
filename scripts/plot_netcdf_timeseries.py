#!/usr/bin/env python3
"""
Create time series plots of all variables in a CF-compliant NetCDF file.

This script reads a NetCDF file and generates a multi-panel figure with:
- Line plots for scalar variables (temperature, pressure, etc.)
- Vector plots for wind data (u, v components)
- Each variable in its own panel with appropriate labels and units
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import (
    DateFormatter,
    DayLocator,
    MonthLocator,
    YearLocator,
    WeekdayLocator,
    MO,
)
from matplotlib.ticker import AutoMinorLocator
from netCDF4 import Dataset, num2date

LOG = logging.getLogger("plot_netcdf")

# Variable metadata: NetCDF name -> (short_label, unit_latex, plot_type)
# plot_type: "scalar" or "wind_component" (skip individual components, plot as vector)
VARIABLE_CONFIG: Dict[str, Tuple[str, str, str]] = {
    "air_pressure": ("P$_{\\mathrm{sfc}}$", "hPa", "scalar"),
    "air_pressure_at_sea_level": ("P$_{\\mathrm{msl}}$", "hPa", "scalar"),
    "air_temperature": ("T$_{\\mathrm{air}}$", "°C", "scalar"),
    "dew_point_temperature": ("T$_{\\mathrm{dew}}$", "°C", "scalar"),
    "water_vapor_partial_pressure": ("e", "hPa", "scalar"),
    "relative_humidity": ("RH", "1", "scalar"),
    "wind_from_direction": ("Wind Dir", "°", "scalar"),
    "wind_speed": ("Wind Speed", "m s$^{-1}$", "scalar"),
    "cloud_area_fraction": ("Cloud", "1", "scalar"),
    "duration_of_sunshine": ("Sunshine", "s", "scalar"),
    "surface_downwelling_shortwave_flux_in_air": ("SW$_{\\downarrow}$", "W m$^{-2}$", "scalar"),
    "precipitation_flux": ("Precip", "kg m$^{-2}$ s$^{-1}$", "scalar"),
    "eastward_wind": ("Wind", "m s$^{-1}$", "wind_component"),
    "northward_wind": ("Wind", "m s$^{-1}$", "wind_component"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time series plots from CF-compliant NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot entire dataset
  %(prog)s tokyo_2019.nc

  # Plot specific time range
  %(prog)s tokyo_2019.nc --start "2019-06-01" --end "2019-08-31"

  # Apply 24-hour moving average
  %(prog)s tokyo_2019.nc --window 24

  # Plot every 3 hours (skip 2 out of 3)
  %(prog)s tokyo_2019.nc --skip 3

  # Custom output location
  %(prog)s tokyo_2019.nc --output-dir figures

  # Combine options
  %(prog)s tokyo_2019.nc --start "2019-01-01" --window 12 --skip 2
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input NetCDF file path",
    )
    parser.add_argument(
        "--start",
        help="Start date/time for plotting (e.g., '2019-01-01' or '2019-01-01 12:00:00')",
    )
    parser.add_argument(
        "--end",
        help="End date/time for plotting (e.g., '2019-12-31' or '2019-12-31 23:00:00')",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        help="Moving average window size in hours (default: no averaging)",
    )
    parser.add_argument(
        "--skip",
        "-s",
        type=int,
        default=1,
        help="Plot every N-th data point (default: 1, plot all points)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("PNG"),
        help="Output directory for PNG file (default: PNG/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PNG file (default: 300)",
    )
    parser.add_argument(
        "--time-var",
        default="time",
        help="Name of the time coordinate variable (default: time)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )
    parser.add_argument(
        "--no-wind-vector",
        action="store_true",
        help="Plot wind components as separate scalars instead of vectors",
    )
    return parser.parse_args()


def load_time_index(ds: Dataset, var_name: str, timezone: str = "UTC") -> pd.DatetimeIndex:
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
    dt_index = pd.to_datetime(times)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(timezone)
    return dt_index


def load_variable_data(
    ds: Dataset,
    var_name: str,
    time_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load a variable as a pandas Series with NaN for missing values."""
    var = ds.variables[var_name]
    data = var[:]

    # Convert masked array to numeric with NaN
    if hasattr(data, "filled"):
        numeric_data = data.filled(np.nan)
    else:
        numeric_data = np.array(data, dtype=np.float64)

    return pd.Series(numeric_data, index=time_index, name=var_name)


def filter_time_range(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Filter dataframe to specified time range."""
    if start is None and end is None:
        return df

    result = df.copy()
    if start:
        start_dt = pd.to_datetime(start)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(df.index.tz)
        result = result[result.index >= start_dt]

    if end:
        end_dt = pd.to_datetime(end)
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize(df.index.tz)
        result = result[result.index <= end_dt]

    return result


def apply_moving_average(series: pd.Series, window: int | None) -> pd.Series:
    """Apply moving average with specified window (in hours)."""
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def skip_data(df: pd.DataFrame, skip: int) -> pd.DataFrame:
    """Keep every N-th data point."""
    if skip <= 1:
        return df
    return df.iloc[::skip]


def convert_units_for_display(series: pd.Series, var_name: str) -> pd.Series:
    """Convert units from NetCDF (SI-like) to display units."""
    result = series.copy()

    # Temperature: K → °C
    if var_name in ["air_temperature", "dew_point_temperature"]:
        result = result - 273.15

    # Pressure: Pa → hPa
    if var_name in ["air_pressure", "air_pressure_at_sea_level", "water_vapor_partial_pressure"]:
        result = result / 100.0

    # Wind: already in m/s
    # Cloud fraction: already in 0-1
    # Precipitation: already in kg/m²/s

    return result


def plot_scalar_variable(
    ax: plt.Axes,
    series: pd.Series,
    label: str,
    unit: str,
    color: str = "b",
) -> None:
    """Plot a scalar time series on given axes."""
    # Remove NaN for plotting
    plot_data = series.dropna()

    if len(plot_data) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        ax.set_ylabel(f"{label} ({unit})", fontsize=10)
        return

    # Plot line
    ax.plot(plot_data.index, plot_data.values, color=color, linewidth=0.8, alpha=0.8)

    # Format axes
    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    # Grid
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.1, linewidth=0.3)

    # Auto minor locator for y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def plot_wind_vector(
    ax: plt.Axes,
    u_series: pd.Series,
    v_series: pd.Series,
    label: str,
    unit: str,
) -> None:
    """Plot wind vectors on given axes using metdata package style."""
    # Combine u and v into dataframe
    df = pd.DataFrame({"u": u_series, "v": v_series})
    df = df.dropna()

    if len(df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        ax.set_ylabel(f"{label} ({unit})", fontsize=10)
        return

    # Convert index to matplotlib date numbers for quiver
    times = df.index.to_pydatetime()
    time_nums = plt.matplotlib.dates.date2num(times)

    # Plot vectors using metdata package style:
    # - units='y', scale_units='y', scale=1 for proper vector scaling
    # - headlength=1, headaxislength=1 for minimal arrow heads
    # - width=0.1 for thin lines (in 'y' units, which makes them adaptive)
    # - alpha=0.5 for transparency
    ax.quiver(
        time_nums,
        0,  # All vectors originate from y=0
        df["u"].values,
        df["v"].values,
        color="b",
        units="y",
        scale_units="y",
        scale=1,
        headlength=1,
        headaxislength=1,
        width=0.1,
        alpha=0.5,
    )

    # Set y-axis limits based on wind magnitude
    speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
    max_speed = speed.max() if len(speed) > 0 else 1.0
    ax.set_ylim(-max_speed * 1.2, max_speed * 1.2)

    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Keep y-ticks to show wind magnitude scale
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Grid
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.1, linewidth=0.3)


def setup_time_axis(ax: plt.Axes, time_range: pd.DatetimeIndex) -> None:
    """Setup time axis formatting with adaptive tick spacing based on data range."""
    # Calculate time span in days
    time_span_days = (time_range[-1] - time_range[0]).days

    # Adaptive locator based on data range
    if time_span_days > 3650:  # > 10 years: yearly major ticks, quarterly minor
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    elif time_span_days > 730:  # > 2 years: yearly major, monthly minor
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(MonthLocator())
    elif time_span_days > 90:  # > 3 months: monthly major, weekly minor
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(WeekdayLocator())
    else:  # <= 3 months: weekly major, daily minor
        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(DayLocator())

    # Rotate labels for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def create_timeseries_plot(
    input_path: Path,
    time_var: str,
    start: str | None,
    end: str | None,
    window: int | None,
    skip: int,
    plot_wind_vectors: bool,
) -> plt.Figure:
    """Create multi-panel time series plot from NetCDF file."""
    LOG.info("Loading NetCDF file: %s", input_path)

    with Dataset(input_path, "r") as ds:
        # Load time index
        time_index = load_time_index(ds, time_var)
        LOG.info("Time range: %s to %s", time_index[0], time_index[-1])

        # Find available variables
        available_vars = [v for v in ds.variables.keys() if v != time_var]
        plot_vars = [v for v in available_vars if v in VARIABLE_CONFIG]

        if not plot_vars:
            raise ValueError("No recognized variables found in NetCDF file")

        LOG.info("Variables to plot: %s", ", ".join(plot_vars))

        # Load all data
        data_dict = {}
        for var_name in plot_vars:
            series = load_variable_data(ds, var_name, time_index)
            series = convert_units_for_display(series, var_name)
            data_dict[var_name] = series

    # Create dataframe
    df = pd.DataFrame(data_dict)

    # Apply filters
    LOG.info("Applying filters...")
    if start or end:
        df = filter_time_range(df, start, end)
        LOG.info("Filtered time range: %s to %s", df.index[0], df.index[-1])

    if window:
        LOG.info("Applying %d-hour moving average", window)
        for col in df.columns:
            df[col] = apply_moving_average(df[col], window)

    if skip > 1:
        LOG.info("Subsampling: keeping every %d-th point", skip)
        df = skip_data(df, skip)
        LOG.info("Data points after subsampling: %d", len(df))

    # Determine panels to plot
    panels: List[Tuple[str, str, str, str]] = []  # (type, var_name, label, unit)

    # Check if we should plot wind as vector
    has_wind_components = "eastward_wind" in df.columns and "northward_wind" in df.columns

    for var_name in df.columns:
        config = VARIABLE_CONFIG.get(var_name)
        if not config:
            continue

        label, unit, plot_type = config

        if plot_type == "wind_component" and has_wind_components and plot_wind_vectors:
            # Only add wind vector panel once
            if var_name == "eastward_wind" and "northward_wind" in df.columns:
                panels.append(("wind_vector", var_name, label, unit))
        elif plot_type == "wind_component" and (not has_wind_components or not plot_wind_vectors):
            # Plot wind components as separate scalars
            if var_name == "eastward_wind":
                panels.append(("scalar", var_name, "u", unit))
            elif var_name == "northward_wind":
                panels.append(("scalar", var_name, "v", unit))
        elif plot_type == "scalar":
            panels.append(("scalar", var_name, label, unit))

    n_panels = len(panels)
    LOG.info("Creating figure with %d panels", n_panels)

    # Create figure with dynamic height
    panel_height = 2.0  # inches per panel
    fig_height = panel_height * n_panels + 1.0  # +1 for spacing
    fig = plt.figure(figsize=(12, fig_height))

    # Create subplots
    axes = []
    for i in range(n_panels):
        ax = fig.add_subplot(n_panels, 1, i + 1)
        axes.append(ax)

    # Plot each panel
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for i, (plot_type, var_name, label, unit) in enumerate(panels):
        ax = axes[i]

        if plot_type == "wind_vector":
            plot_wind_vector(
                ax,
                df["eastward_wind"],
                df["northward_wind"],
                label,
                unit,
            )
        else:  # scalar
            color = colors[color_idx % len(colors)]
            plot_scalar_variable(ax, df[var_name], label, unit, color=color)
            color_idx += 1

        # Setup time axis for all panels
        setup_time_axis(ax, df.index)

        # Only show x-label on bottom panel
        if i < n_panels - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Date", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    return fig


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    # Validate input file
    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    # Validate skip
    if args.skip < 1:
        raise SystemExit("--skip must be at least 1")

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Output directory: %s", output_dir)

    # Generate output filename
    output_filename = f"{args.input.stem}_timeseries.png"
    output_path = output_dir / output_filename

    # Create plot
    try:
        fig = create_timeseries_plot(
            input_path=args.input,
            time_var=args.time_var,
            start=args.start,
            end=args.end,
            window=args.window,
            skip=args.skip,
            plot_wind_vectors=not args.no_wind_vector,
        )

        # Save figure
        LOG.info("Saving figure to: %s", output_path)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        LOG.info("Done!")

    except Exception as e:
        LOG.error("Error: %s", e)
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
