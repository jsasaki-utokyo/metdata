"""Public accessors for JMA station coordinate tables.

The bundled ``gwo_stn.csv`` stores latitude and longitude as JMA-style
sexagesimal-encoded-as-decimal numbers (``DD.MMSS``): the integer part
holds the degrees, and the four-digit fractional part packs minutes
(first two digits) and seconds (next two digits). For example
``35.2612`` means 35°26'12" (≈ 35.4367° N), and ``139.3918`` means
139°39'18" (≈ 139.6550° E).

The public API of this module — ``load_gwo`` and (transitively)
``gwo.Stn.values`` — returns **true decimal degrees**, matching the
CF-1.x ``degrees_north`` / ``degrees_east`` units that downstream
NetCDF writers apply. The raw on-disk CSV is left untouched as the
single source of truth.
"""

from __future__ import annotations

import math
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd

_GWO_CSV = files("metdata").joinpath("gwo_stn.csv")


def _ddmmss_to_decimal(value: float) -> float:
    """Convert a JMA-style ``DD.MMSS`` float to true decimal degrees.

    The integer part is degrees; the fractional part — formatted to
    four digits — is read as ``MMSS`` (minutes and seconds, zero-padded
    on the right). Trailing zeros that may have been elided by the CSV
    are restored by formatting with ``f"{abs(v):.4f}"``; this keeps the
    conversion robust against the IEEE-754 noise that bites naive
    arithmetic (e.g. ``44.01 - 44 == 0.010000000000005116``).

    Parameters
    ----------
    value : float
        Latitude or longitude in JMA ``DD.MMSS`` form. NaN passes through.

    Returns
    -------
    float
        Decimal degrees (signed). NaN if the input is NaN.

    Raises
    ------
    ValueError
        If the encoded minutes or seconds field is not in ``[0, 60)``.
    """
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    sign = -1.0 if value < 0 else 1.0
    text = f"{abs(float(value)):.4f}"
    int_part, frac_part = text.split(".")
    degrees = int(int_part)
    minutes = int(frac_part[0:2])
    seconds = int(frac_part[2:4])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(
            f"Invalid DD.MMSS coordinate {value!r}: parsed as "
            f"deg={degrees}, min={minutes}, sec={seconds}"
        )
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def _ddmmss_series_to_decimal(series: pd.Series) -> pd.Series:
    """Vectorised wrapper for :func:`_ddmmss_to_decimal`."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(
        lambda v: float("nan") if pd.isna(v) else _ddmmss_to_decimal(v)
    )


def load_gwo(name: Optional[str] = None) -> pd.DataFrame:
    """Return the JMA Ground Weather Observation (GWO) station table.

    Parameters
    ----------
    name : str, optional
        If given, return the row for this station (matched against
        ``name_en``, the romanized name that also matches the on-disk
        directory name). Returns ``None`` if not found.

    Returns
    -------
    pandas.DataFrame
        Columns: ``kanid`` (int), ``name_en`` (str), ``name_ja`` (str),
        ``lat`` (float, decimal degrees N), ``lon`` (float, decimal
        degrees E), ``elev_m`` (float), ``baro_height_m`` (float),
        ``anemo_height_m`` (float), ``temp_height_m`` (float,
        JMA-standard 1.5 m for all stations).

    Notes
    -----
    Coordinates are converted from the on-disk ``DD.MMSS`` form to true
    decimal degrees by :func:`_ddmmss_to_decimal`.
    """
    df = pd.read_csv(_GWO_CSV, skiprows=1).rename(
        columns={
            "Kname":              "name_en",
            "station_id":         "kanid",
            "name_jp":            "name_ja",
            "latitude":           "lat",
            "longitude":          "lon",
            "altitude":           "elev_m",
            "barometer_height":   "baro_height_m",
            "anemometer_height":  "anemo_height_m",
            "temperature_height": "temp_height_m",
        }
    )
    df = df[["kanid", "name_en", "name_ja", "lat", "lon",
             "elev_m", "baro_height_m", "anemo_height_m",
             "temp_height_m"]]
    df["kanid"] = df["kanid"].astype(int)
    df["lat"] = _ddmmss_series_to_decimal(df["lat"])
    df["lon"] = _ddmmss_series_to_decimal(df["lon"])
    if name is not None:
        sub = df[df["name_en"] == name]
        return sub.iloc[0] if len(sub) else None
    return df.copy()
