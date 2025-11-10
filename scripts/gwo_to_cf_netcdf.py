#!/usr/bin/env python3
"""
Convert hourly GWO CSV files into a CF-compliant NetCDF file.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num, default_fillvals
from pvlib import solarposition

from metdata import gwo

LOG = logging.getLogger("gwo_to_cf")
FILL_VALUE = np.float32(default_fillvals["f4"])
UTC_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
CALENDAR = "gregorian"
RMK_RULES = {
    "default": {"missing": {"0", "1", 0, 1}, "rmk2_zero": False},
    "clod": {"missing": {"0", "1", "2", 0, 1, 2}, "rmk2_zero": False},
    "muki": {"missing": {"0", "1", "2", 0, 1, 2}, "rmk2_zero": False},
    "sped": {"missing": {"0", "1", "2", 0, 1, 2}, "rmk2_zero": False},
    "lght": {"missing": {"0", "1", 0, 1}, "rmk2_zero": True},
    "slht": {"missing": {"0", "1", 0, 1}, "rmk2_zero": True},
    "kous": {"missing": {"0", "1", 0, 1}, "rmk2_zero": True},
}
RMK_MAP = {
    "lhpa": "lhpaRMK",
    "shpa": "shpaRMK",
    "kion": "kionRMK",
    "stem": "stemRMK",
    "rhum": "rhumRMK",
    "muki": "mukiRMK",
    "sped": "spedRMK",
    "clod": "clodRMK",
    "tnki": "tnkiRMK",
    "humd": "humdRMK",
    "lght": "lghtRMK",
    "slht": "slhtRMK",
    "kous": "kousRMK",
}


@dataclass(frozen=True)
class CFVariable:
    """Metadata for a CF variable derived from the GWO dataframe."""

    source: str
    standard_name: str
    long_name: str
    units: str
    transform: Callable[[pd.Series], pd.Series]
    dtype: str = "f4"
    additional_attributes: Dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CF-compliant NetCDF file from GWO hourly CSV files by "
            "station and date range."
        )
    )
    parser.add_argument("--station", required=True, help="Station name (e.g., Tokyo)")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime (e.g., 2019-01-01 00:00:00)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime (exclusive, e.g., 2019-12-31 23:00:00)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NetCDF path. Defaults to ./<station>_<start>_<end>.nc",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Directory containing station folders (defaults to "
            "$DATA_DIR/met/JMA_DataBase/GWO/Hourly/)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target NetCDF file if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    return parser.parse_args()


def resolve_hourly_dir(explicit: Path | None) -> Path:
    if explicit:
        hourly_dir = explicit.expanduser()
    else:
        data_root = os.environ.get("DATA_DIR")
        if not data_root:
            raise RuntimeError(
                "DATA_DIR is not set. Specify --data-dir or export DATA_DIR."
            )
        hourly_dir = Path(data_root).expanduser() / "met" / "JMA_DataBase" / "GWO" / "Hourly"
    if not hourly_dir.exists():
        raise FileNotFoundError(f"Hourly directory does not exist: {hourly_dir}")
    return hourly_dir


def ensure_trailing_sep(path: Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else f"{text}{os.sep}"


def load_raw_dataframe(
    station: str, start: str, end: str, hourly_dir: Path
) -> pd.DataFrame:
    reader = gwo.Hourly(
        datetime_ini=start,
        datetime_end=end,
        stn=station,
        dirpath=ensure_trailing_sep(hourly_dir),
    )
    df = reader.df_org.copy()
    if df.empty:
        raise RuntimeError("No rows returned for the requested period.")
    return _mask_rmk_values(df)


def _mask_rmk_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply RMK-specific masking and zero-filling rules to the raw dataframe."""
    masked = df.copy()
    for value_col, rmk_col in RMK_MAP.items():
        if value_col not in masked.columns or rmk_col not in masked.columns:
            continue
        rules = RMK_RULES.get(value_col, RMK_RULES["default"])
        rmk_series = masked[rmk_col]
        missing_mask = rmk_series.isin(rules["missing"])
        masked.loc[missing_mask, value_col] = np.nan
        if rules.get("rmk2_zero"):
            zero_mask = rmk_series.isin({"2", 2})
            # Check if there are ANY valid observations (RMK=8)
            # If not, the instrument doesn't exist - treat all as missing
            has_valid_obs = rmk_series.isin({"8", 8}).any()
            if not has_valid_obs:
                # No valid observations - instrument doesn't exist
                masked.loc[zero_mask, value_col] = np.nan
            else:
                # Has valid observations - RMK=2 is nighttime, set to zero
                masked.loc[zero_mask, value_col] = 0.0

    # Wind-specific rules for calm and missing conditions
    if (
        "muki" in masked.columns
        and "mukiRMK" in masked.columns
        and "sped" in masked.columns
        and "spedRMK" in masked.columns
    ):
        # Rule 1: When muki=0 (direction N/A) with mukiRMK=8 and very low speed,
        # this indicates calm conditions where direction is undefined.
        # Set both direction and speed to 0 to indicate calm.
        # Speed threshold: ≤ 3 units (0.3 m/s after conversion)
        calm_by_data = (
            (masked["muki"] == 0)
            & masked["mukiRMK"].isin({"8", 8})
            & (masked["sped"] <= 3)
            & masked["spedRMK"].isin({"8", 8})
        )
        if calm_by_data.any():
            masked.loc[calm_by_data, "sped"] = 0.0
            # muki=0 will be converted to NaN by _deg_from_code later

        # Rule 2: When direction RMK is 2 or 6 (not observed) and speed RMK is also not observed,
        # set both to NaN (truly missing data)
        dir_missing = masked["mukiRMK"].isin({"2", "6", 2, 6})
        speed_missing = masked["spedRMK"].isin({"0", "1", "2", 0, 1, 2})
        both_missing = dir_missing & speed_missing
        if both_missing.any():
            masked.loc[both_missing, ["muki", "sped"]] = np.nan

        # Rule 3: When direction RMK is 2/6 but speed RMK is valid (8),
        # this also indicates calm - set speed to 0
        calm_by_rmk = dir_missing & masked["spedRMK"].isin({"8", 8})
        if calm_by_rmk.any():
            masked.loc[calm_by_rmk, "muki"] = 0  # Will become NaN in _deg_from_code
            masked.loc[calm_by_rmk, "sped"] = 0.0
    return masked


def _deg_from_code(series: pd.Series) -> pd.Series:
    code_to_deg = {
        0: np.nan,  # calm
        1: 22.5,
        2: 45.0,
        3: 67.5,
        4: 90.0,
        5: 112.5,
        6: 135.0,
        7: 157.5,
        8: 180.0,
        9: 202.5,
        10: 225.0,
        11: 247.5,
        12: 270.0,
        13: 292.5,
        14: 315.0,
        15: 337.5,
        16: 0.0,
    }
    mapped = series.map(code_to_deg)
    return mapped


def _no_transform(series: pd.Series) -> pd.Series:
    """Identity transform - returns series as-is (already converted by gwo.Hourly)."""
    return series.astype(float)


def _kelvin_from_c(series: pd.Series) -> pd.Series:
    """Convert °C to K (gwo.Hourly already converted from tenths)."""
    return series.astype(float) + 273.15


def _pa_from_hpa(series: pd.Series) -> pd.Series:
    """Convert hPa to Pa (gwo.Hourly already converted from tenths)."""
    return series.astype(float) * 100.0


def _vapor_pressure_pa(series: pd.Series) -> pd.Series:
    return _pa_from_hpa(series)


def _relative_humidity(series: pd.Series) -> pd.Series:
    """gwo.Hourly already converted from % to 0-1."""
    return series.astype(float)


def _wind_speed(series: pd.Series) -> pd.Series:
    """gwo.Hourly already converted to m/s."""
    return series.astype(float)


def _cloud_fraction(series: pd.Series) -> pd.Series:
    """gwo.Hourly already converted to 0-1."""
    return series.astype(float)


def _seconds_from_hours(series: pd.Series) -> pd.Series:
    """Convert hours to seconds (gwo.Hourly already converted from tenths)."""
    return series.astype(float) * 3600.0


def _solar_irradiance(series: pd.Series) -> pd.Series:
    """gwo.Hourly already converted to W/m²."""
    return series.astype(float)


def _precipitation_rate(series: pd.Series) -> pd.Series:
    """gwo.Hourly already converted to m/s."""
    return series.astype(float)


CF_VARIABLES: Dict[str, CFVariable] = {
    "air_pressure": CFVariable(
        source="lhpa",
        standard_name="air_pressure",
        long_name="Station level air pressure",
        units="Pa",
        transform=_pa_from_hpa,
    ),
    "air_pressure_at_sea_level": CFVariable(
        source="shpa",
        standard_name="air_pressure_at_sea_level",
        long_name="Sea level air pressure",
        units="Pa",
        transform=_pa_from_hpa,
    ),
    "air_temperature": CFVariable(
        source="kion",
        standard_name="air_temperature",
        long_name="Dry-bulb air temperature",
        units="K",
        transform=_kelvin_from_c,
    ),
    "dew_point_temperature": CFVariable(
        source="humd",
        standard_name="dew_point_temperature",
        long_name="Dew point temperature",
        units="K",
        transform=_kelvin_from_c,
    ),
    "water_vapor_partial_pressure": CFVariable(
        source="stem",
        standard_name="water_vapor_partial_pressure",
        long_name="Water vapor partial pressure",
        units="Pa",
        transform=_vapor_pressure_pa,
    ),
    "relative_humidity": CFVariable(
        source="rhum",
        standard_name="relative_humidity",
        long_name="Relative humidity",
        units="1",
        transform=_relative_humidity,
    ),
    "wind_from_direction": CFVariable(
        source="muki",
        standard_name="wind_from_direction",
        long_name="Wind from direction",
        units="degree",
        transform=_no_transform,  # Already converted by gwo.Hourly
    ),
    "wind_speed": CFVariable(
        source="sped",
        standard_name="wind_speed",
        long_name="Wind speed",
        units="m s-1",
        transform=_wind_speed,
    ),
    "cloud_area_fraction": CFVariable(
        source="clod",
        standard_name="cloud_area_fraction",
        long_name="Total cloud cover",
        units="1",
        transform=_cloud_fraction,
    ),
    "duration_of_sunshine": CFVariable(
        source="lght",
        standard_name="duration_of_sunshine",
        long_name="Sunshine duration per interval",
        units="s",
        transform=_seconds_from_hours,
    ),
    "surface_downwelling_shortwave_flux_in_air": CFVariable(
        source="slht",
        standard_name="surface_downwelling_shortwave_flux_in_air",
        long_name="Global horizontal irradiance",
        units="W m-2",
        transform=_solar_irradiance,
    ),
    "precipitation_flux": CFVariable(
        source="kous",
        standard_name="precipitation_flux",
        long_name="Liquid precipitation rate",
        units="kg m-2 s-1",
        transform=_precipitation_rate,
        additional_attributes={"comment": "Derived from hourly accumulation reported in 0.1 mm/h."},
    ),
}


def convert_to_cf(df: pd.DataFrame) -> pd.DataFrame:
    cf_data = {}
    for target, meta in CF_VARIABLES.items():
        if meta.source not in df.columns:
            LOG.warning("Column %s missing; skipping %s", meta.source, target)
            continue
        cf_data[target] = meta.transform(df[meta.source])
    cf_df = pd.DataFrame(cf_data, index=df.index)
    cf_df.sort_index(inplace=True)

    # Derive wind components when direction and speed exist.
    if {"wind_from_direction", "wind_speed"}.issubset(cf_df.columns):
        direction_deg = cf_df["wind_from_direction"].to_numpy(dtype=np.float64)
        speed = cf_df["wind_speed"].to_numpy(dtype=np.float64)
        eastward = np.full_like(speed, np.nan, dtype=np.float64)
        northward = np.full_like(speed, np.nan, dtype=np.float64)

        # Handle normal conditions: both direction and speed are valid
        valid = np.isfinite(direction_deg) & np.isfinite(speed)
        if valid.any():
            direction_rad = np.deg2rad(direction_deg[valid])
            eastward[valid] = -speed[valid] * np.sin(direction_rad)
            northward[valid] = -speed[valid] * np.cos(direction_rad)

        # Handle calm conditions: direction is NaN but speed is finite and near zero
        # This occurs when mukiRMK=2 or 6 (calm/not observed) but spedRMK=8 (valid)
        calm_mask = (
            np.isnan(direction_deg)
            & np.isfinite(speed)
            & (np.abs(speed) <= 1e-6)
        )
        if calm_mask.any():
            eastward[calm_mask] = 0.0
            northward[calm_mask] = 0.0

        cf_df["eastward_wind"] = eastward.astype(np.float32)
        cf_df["northward_wind"] = northward.astype(np.float32)
    return cf_df.astype(np.float32)


def enforce_nighttime_zero(
    cf_df: pd.DataFrame,
    original_index: pd.Index,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Tokyo",
) -> pd.DataFrame:
    target_cols = [
        "duration_of_sunshine",
        "surface_downwelling_shortwave_flux_in_air",
    ]
    present = [col for col in target_cols if col in cf_df.columns]
    if not present or cf_df.empty:
        return cf_df

    dt_index = pd.DatetimeIndex(original_index)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(timezone)
    else:
        dt_index = dt_index.tz_convert(timezone)
    solpos = solarposition.get_solarposition(dt_index, latitude, longitude)
    night_mask = pd.Series(
        (solpos["apparent_elevation"].to_numpy() <= 0),
        index=cf_df.index,
    )

    # Only fill nighttime zeros if there are actual daytime measurements
    # This prevents filling when the instrument doesn't exist at all
    if night_mask.any():
        for col in present:
            # Check if there are any valid (non-NaN) daytime values
            day_mask = ~night_mask
            has_daytime_data = cf_df.loc[day_mask, col].notna().any()

            # Only fill nighttime with zeros if daytime has measurements
            if has_daytime_data:
                cf_df.loc[night_mask, col] = 0.0

    return cf_df


def summarize_missing(cf_df: pd.DataFrame) -> List[Tuple[str, int, float]]:
    summaries = []
    total = len(cf_df)
    for col in cf_df.columns:
        missing = int(cf_df[col].isna().sum())
        pct = (missing / total * 100.0) if total else 0.0
        summaries.append((col, missing, pct))
    return summaries


def time_to_utc_numbers(index: pd.Index) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex for NetCDF export.")
    localized = index.tz_localize("Asia/Tokyo")
    utc_index = localized.tz_convert("UTC").tz_localize(None)
    return date2num(
        utc_index.to_pydatetime(),
        units=UTC_UNITS,
        calendar=CALENDAR,
    )


def fetch_station_metadata(station: str) -> Dict[str, float | str]:
    catalog = gwo.Stn()
    try:
        return catalog.values(station)
    except ValueError as exc:  # pandas raises ValueError when missing
        raise RuntimeError(f"Station {station!r} not found in gwo_stn.csv") from exc


def write_netcdf(
    output_path: Path,
    cf_df: pd.DataFrame,
    station_meta: Dict[str, float | str],
    requested: Dict[str, str],
) -> None:
    times = time_to_utc_numbers(cf_df.index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_path, mode="w") as nc:
        nc.createDimension("time", len(times))
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = times
        time_var.units = UTC_UNITS
        time_var.calendar = CALENDAR
        time_var.standard_name = "time"

        lat_var = nc.createVariable("latitude", "f4")
        lat_var[:] = station_meta["latitude"]
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"

        lon_var = nc.createVariable("longitude", "f4")
        lon_var[:] = station_meta["longitude"]
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"

        alt_var = nc.createVariable("altitude", "f4")
        alt_var[:] = station_meta["altitude"]
        alt_var.standard_name = "altitude"
        alt_var.units = "m"

        def to_masked(series: pd.Series) -> np.ma.MaskedArray:
            arr = series.to_numpy(dtype=np.float32, copy=True)
            mask = ~np.isfinite(arr)
            return np.ma.masked_array(arr, mask=mask)

        for name in cf_df.columns:
            meta = CF_VARIABLES.get(name)
            attrs: Dict[str, str] = {}
            if meta:
                attrs = {
                    "standard_name": meta.standard_name,
                    "long_name": meta.long_name,
                    "units": meta.units,
                }
                attrs.update(meta.additional_attributes)
            elif name in {"eastward_wind", "northward_wind"}:
                attrs = {
                    "standard_name": name,
                    "long_name": name.replace("_", " "),
                    "units": "m s-1",
                }
            var = nc.createVariable(name, "f4", ("time",), fill_value=FILL_VALUE)
            var[:] = to_masked(cf_df[name])
            for key, value in attrs.items():
                setattr(var, key, value)
            var.missing_value = FILL_VALUE
            var.coordinates = "latitude longitude"

        nc.Conventions = "CF-1.10"
        nc.title = f"GWO hourly observations for {requested['station']}"
        nc.institution = "metdata contributors"
        nc.source = "GWO CSV archive"
        nc.history = (
            f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} - Created by scripts/gwo_to_cf_netcdf.py"
        )
        nc.station_id = requested["station"]
        nc.time_coverage_start = requested["start"]
        nc.time_coverage_end = requested["end"]
        nc.comment = (
            "Converted from raw GWO CSV files with missing values preserved as CF _FillValue."
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    hourly_dir = resolve_hourly_dir(args.data_dir)
    output = (
        args.output
        if args.output
        else Path(
            f"{args.station}_{args.start.replace(' ', 'T')}_{args.end.replace(' ', 'T')}.nc"
        )
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists. Use --overwrite to replace.")

    station_meta = fetch_station_metadata(args.station)

    LOG.info("Loading GWO data for %s from %s", args.station, hourly_dir)
    df = load_raw_dataframe(args.station, args.start, args.end, hourly_dir)
    cf_df = convert_to_cf(df)
    cf_df = enforce_nighttime_zero(
        cf_df,
        df.index,
        latitude=station_meta["latitude"],
        longitude=station_meta["longitude"],
    )
    summaries = summarize_missing(cf_df)
    LOG.info("Missing values per variable:")
    for name, missing, pct in summaries:
        LOG.info("  %-45s %5d (%.2f%%)", name, missing, pct)

    LOG.info("Writing NetCDF to %s", output)
    write_netcdf(
        output_path=output,
        cf_df=cf_df,
        station_meta=station_meta,
        requested={"station": args.station, "start": args.start, "end": args.end},
    )
    LOG.info("Done.")


if __name__ == "__main__":
    main()
