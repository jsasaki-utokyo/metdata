"""Public accessors for JMA station coordinate tables."""

from __future__ import annotations

from importlib.resources import files
from typing import Optional

import pandas as pd

_GWO_CSV = files("metdata").joinpath("gwo_stn.csv")


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
        ``lat`` (float, deg N), ``lon`` (float, deg E), ``elev_m``
        (float), ``baro_height_m`` (float), ``anemo_height_m`` (float),
        ``temp_height_m`` (float, JMA-standard 1.5 m for all stations).
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
    if name is not None:
        sub = df[df["name_en"] == name]
        return sub.iloc[0] if len(sub) else None
    return df.copy()
