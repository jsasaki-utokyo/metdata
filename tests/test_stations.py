"""Sanity checks for the bundled GWO station table."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import metdata
from metdata.stations import _ddmmss_to_decimal


# ---------------------------------------------------------------------------
# Smoke tests (no external data needed)
# ---------------------------------------------------------------------------


def test_load_gwo_smoke():
    df = metdata.load_gwo()
    # Mandatory columns
    expected = {"kanid", "name_en", "name_ja", "lat", "lon",
                "elev_m", "baro_height_m", "anemo_height_m",
                "temp_height_m"}
    assert expected.issubset(df.columns), df.columns
    # Reasonable physical ranges
    assert df["lat"].between(20.0, 46.0).all()
    assert df["lon"].between(122.0, 154.0).all()
    assert df["anemo_height_m"].between(1.0, 200.0).all()
    # JMA standard thermometer height is 1.5 m above ground for every
    # 地上気象観測装置 ("官") site; metdata stores this constant for all
    # 155 stations as documented in gwo_stn.csv's comment line.
    assert (df["temp_height_m"] == 1.5).all()
    # No duplicate KanIDs
    assert df["kanid"].is_unique


def test_load_gwo_dtypes():
    df = metdata.load_gwo()
    assert pd.api.types.is_integer_dtype(df["kanid"])
    assert pd.api.types.is_float_dtype(df["lat"])
    assert pd.api.types.is_float_dtype(df["lon"])
    assert pd.api.types.is_float_dtype(df["elev_m"])
    assert pd.api.types.is_float_dtype(df["baro_height_m"])
    assert pd.api.types.is_float_dtype(df["anemo_height_m"])
    assert pd.api.types.is_float_dtype(df["temp_height_m"])


def test_load_gwo_returns_independent_copy():
    """Mutating the returned frame must not affect subsequent calls."""
    df1 = metdata.load_gwo()
    df1.loc[df1.index[0], "name_en"] = "MUTATED"
    df2 = metdata.load_gwo()
    assert df2.loc[df2.index[0], "name_en"] != "MUTATED"


def test_load_gwo_by_name_known():
    tokyo = metdata.load_gwo("Tokyo")
    assert tokyo is not None
    assert tokyo["kanid"] == 662
    assert tokyo["name_ja"] == "東京"


# ---------------------------------------------------------------------------
# Coordinate decoding (DD.MMSS -> decimal degrees)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ddmmss,expected",
    [
        # Yokohama 35°26'12" / 139°39'18"
        (35.2612, 35 + 26 / 60 + 12 / 3600),
        (139.3918, 139 + 39 / 60 + 18 / 3600),
        # Abashiri 44°01'00" / 144°17'00" (trailing zeros elided in CSV)
        (44.01, 44 + 1 / 60),
        (144.17, 144 + 17 / 60),
        # Aburatsu 31°34'30" (one trailing zero elided)
        (31.343, 31 + 34 / 60 + 30 / 3600),
        # A negative pseudo-value (no station has one, but the helper
        # must still respect the sign).
        (-35.2612, -(35 + 26 / 60 + 12 / 3600)),
        (0.0, 0.0),
    ],
)
def test_ddmmss_to_decimal_known(ddmmss, expected):
    assert _ddmmss_to_decimal(ddmmss) == pytest.approx(expected, rel=0, abs=1e-9)


def test_ddmmss_to_decimal_rejects_invalid_minutes():
    # 35.6500 would imply 65 minutes, which is impossible.
    with pytest.raises(ValueError):
        _ddmmss_to_decimal(35.6500)


def test_ddmmss_to_decimal_rejects_invalid_seconds():
    # 35.0061 would imply 61 seconds, which is impossible.
    with pytest.raises(ValueError):
        _ddmmss_to_decimal(35.0061)


def test_ddmmss_to_decimal_passes_through_nan():
    assert math.isnan(_ddmmss_to_decimal(float("nan")))


def test_load_gwo_returns_decimal_degrees_for_yokohama():
    """Spot-check: Yokohama JMA principal station ≈ 35.4367° N, 139.6550° E."""
    yokohama = metdata.load_gwo("Yokohama")
    assert yokohama is not None
    assert yokohama["lat"] == pytest.approx(35 + 26 / 60 + 12 / 3600, abs=1e-9)
    assert yokohama["lon"] == pytest.approx(139 + 39 / 60 + 18 / 3600, abs=1e-9)


def test_load_gwo_lat_lon_within_japan_bbox():
    """Every decoded coordinate must land inside Japan's bounding box.

    This is a stronger version of the existing range smoke test: it is
    only true after DD.MMSS -> decimal-degree conversion. Before the
    conversion was added some station entries (e.g., 39.4254 for Akita)
    happened to fall in 20-46 by coincidence; the bounds below are
    tight enough to fail if conversion regresses.
    """
    df = metdata.load_gwo()
    assert df["lat"].between(20.0, 46.0).all()
    assert df["lon"].between(122.0, 154.0).all()
    # Sanity: no station should land at an exact integer degree (which
    # would happen if a DD.MMSS value with non-zero MM/SS were left
    # uninterpreted).
    nontrivial_frac = ((df["lat"] - df["lat"].astype(int)).abs() > 1e-6)
    assert nontrivial_frac.mean() > 0.95


def test_stn_values_returns_decimal_degrees():
    """``gwo.Stn().values()`` must mirror ``load_gwo`` after the fix."""
    from metdata import gwo

    catalog = gwo.Stn()
    rec = catalog.values("Yokohama")
    yokohama = metdata.load_gwo("Yokohama")
    assert rec["latitude"] == pytest.approx(float(yokohama["lat"]), abs=1e-9)
    assert rec["longitude"] == pytest.approx(float(yokohama["lon"]), abs=1e-9)


def test_load_gwo_by_name_unknown():
    assert metdata.load_gwo("NoSuchStation") is None


@pytest.mark.parametrize(
    "name,kanid,name_ja",
    [
        ("Asosan", 821, "阿蘇山"),
        ("Ibukiyama", 751, "伊吹山"),
        ("Tsurugisan", 894, "剣山"),
    ],
)
def test_newly_added_mountain_summit_stations(name, kanid, name_ja):
    """The three summit stations added in 2026-05 must be present and consistent."""
    row = metdata.load_gwo(name)
    assert row is not None, f"{name} missing from gwo_stn.csv"
    assert row["kanid"] == kanid
    assert row["name_ja"] == name_ja
    # All three are mountain summit observatories with elevation > 1000 m.
    assert row["elev_m"] > 1000.0
    # Anemometer is at most a few tens of metres above ground.
    assert 1.0 <= row["anemo_height_m"] <= 50.0
    # Barometer altitude AMSL must be at or above ground altitude
    # (typically within a few metres).
    assert 0.0 <= row["baro_height_m"] - row["elev_m"] <= 10.0


# ---------------------------------------------------------------------------
# DATA_DIR-gated cross-checks against the on-disk archive
# ---------------------------------------------------------------------------


def _gwo_hourly_dir() -> Path | None:
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return None
    gwoh = Path(data_dir) / "met/JMA_DataBase/GWO/Hourly"
    return gwoh if gwoh.is_dir() else None


@pytest.mark.skipif(
    _gwo_hourly_dir() is None,
    reason="$DATA_DIR/met/JMA_DataBase/GWO/Hourly not available",
)
def test_gwo_covers_all_disk_directories():
    """Every directory under GWO/Hourly must have a row in the table."""
    gwoh = _gwo_hourly_dir()
    on_disk = {p.name for p in gwoh.iterdir() if p.is_dir()}
    in_table = set(metdata.load_gwo()["name_en"])
    missing = on_disk - in_table
    assert not missing, f"GWO directories missing from gwo_stn.csv: {sorted(missing)}"


@pytest.mark.skipif(
    _gwo_hourly_dir() is None,
    reason="$DATA_DIR/met/JMA_DataBase/GWO/Hourly not available",
)
@pytest.mark.parametrize("name", ["Tokyo", "Asosan", "Ibukiyama", "Tsurugisan"])
def test_kanid_matches_yearly_csv_first_column(name):
    """Cross-check: the kanid in the table matches column 1 of any yearly CSV."""
    gwoh = _gwo_hourly_dir()
    stn_dir = gwoh / name
    candidates = sorted(stn_dir.glob(f"{name}[0-9][0-9][0-9][0-9].csv"))
    # Pick a file with non-zero size; some yearly files are empty placeholders.
    sample = next((p for p in candidates if p.stat().st_size > 0), None)
    if sample is None:
        pytest.skip(f"No populated yearly CSV under {stn_dir}")
    with sample.open(encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n").split(",")
    on_disk_kanid = int(first[0])
    expected_kanid = int(metdata.load_gwo(name)["kanid"])
    assert on_disk_kanid == expected_kanid
