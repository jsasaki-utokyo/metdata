#!/usr/bin/env python3
"""
Convert hourly GWO CSV files into a CF-compliant NetCDF file.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num

from metdata import gwo

LOG = logging.getLogger("gwo_to_cf")
FILL_VALUE = np.float32(-9999.0)
UTC_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
CALENDAR = "gregorian"


class RawHourly(gwo.Hourly):
    """Hourly reader that skips TEEM-specific unit conversion."""

    def _unit_conversion(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return df


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
    reader = RawHourly(
        datetime_ini=start,
        datetime_end=end,
        stn=station,
        dirpath=ensure_trailing_sep(hourly_dir),
    )
    df = reader.df_org.copy()
    if df.empty:
        raise RuntimeError("No rows returned for the requested period.")
    return df


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


def _kelvin_from_tenths_c(series: pd.Series) -> pd.Series:
    return series.astype(float) / 10.0 + 273.15


def _pa_from_tenths_hpa(series: pd.Series) -> pd.Series:
    return series.astype(float) * 10.0


def _vapor_pressure_pa(series: pd.Series) -> pd.Series:
    return _pa_from_tenths_hpa(series)


def _relative_humidity(series: pd.Series) -> pd.Series:
    return series.astype(float) / 100.0


def _wind_speed(series: pd.Series) -> pd.Series:
    return series.astype(float) / 10.0


def _cloud_fraction(series: pd.Series) -> pd.Series:
    return series.astype(float) / 10.0


def _seconds_from_tenths_hours(series: pd.Series) -> pd.Series:
    return (series.astype(float) / 10.0) * 3600.0


def _solar_irradiance(series: pd.Series) -> pd.Series:
    return series.astype(float) * (1.0e4 / 3.6e3)


def _precipitation_rate(series: pd.Series) -> pd.Series:
    return series.astype(float) * (1.0e-4 / 3600.0)


CF_VARIABLES: Dict[str, CFVariable] = {
    "air_pressure": CFVariable(
        source="lhpa",
        standard_name="air_pressure",
        long_name="Station level air pressure",
        units="Pa",
        transform=_pa_from_tenths_hpa,
    ),
    "air_pressure_at_sea_level": CFVariable(
        source="shpa",
        standard_name="air_pressure_at_sea_level",
        long_name="Sea level air pressure",
        units="Pa",
        transform=_pa_from_tenths_hpa,
    ),
    "air_temperature": CFVariable(
        source="kion",
        standard_name="air_temperature",
        long_name="Dry-bulb air temperature",
        units="K",
        transform=_kelvin_from_tenths_c,
    ),
    "dew_point_temperature": CFVariable(
        source="humd",
        standard_name="dew_point_temperature",
        long_name="Dew point temperature",
        units="K",
        transform=_kelvin_from_tenths_c,
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
        transform=_deg_from_code,
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
        transform=_seconds_from_tenths_hours,
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
        direction = np.deg2rad(cf_df["wind_from_direction"].to_numpy(dtype=float))
        speed = cf_df["wind_speed"].to_numpy(dtype=float)
        mask = np.isnan(direction) | np.isnan(speed)
        eastward = np.full_like(speed, np.nan)
        northward = np.full_like(speed, np.nan)
        valid = ~mask
        eastward[valid] = -speed[valid] * np.sin(direction[valid])
        northward[valid] = -speed[valid] * np.cos(direction[valid])
        cf_df["eastward_wind"] = eastward
        cf_df["northward_wind"] = northward
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
            var[:] = np.ma.masked_invalid(cf_df[name].to_numpy(dtype=np.float32))
            for key, value in attrs.items():
                setattr(var, key, value)
            var.coordinates = "latitude longitude"

        nc.Conventions = "CF-1.10"
        nc.title = f"GWO hourly observations for {requested['station']}"
        nc.institution = "metdata contributors"
        nc.source = "GWO CSV archive"
        nc.history = (
            f"{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} - Created by scripts/gwo_to_cf_netcdf.py"
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

    LOG.info("Loading GWO data for %s from %s", args.station, hourly_dir)
    df = load_raw_dataframe(args.station, args.start, args.end, hourly_dir)
    cf_df = convert_to_cf(df)
    summaries = summarize_missing(cf_df)
    LOG.info("Missing values per variable:")
    for name, missing, pct in summaries:
        LOG.info("  %-45s %5d (%.2f%%)", name, missing, pct)

    station_meta = fetch_station_metadata(args.station)
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
