from pathlib import Path

import pandas as pd
import pytest

from gnssdl import ngl

HOLDINGS = """\
Sta  Lat(deg)   Long(deg) Hgt(m)  X(m)           Y(m)         Z(m)          Dtbeg      Dtend      Dtmod      NumSol StaOrigName
ANKR  39.8874   32.7585  976.045  4121948.0  2652187.0  4069023.0 1995-06-25 2022-01-04 2025-09-25   9000
FAR1  37.2000  217.0000  100.000  0 0 0 2010-01-01 2024-01-01 2025-09-25   5000
"""

STEPS = """\
ANKR  08MAY06  1  Antenna_and_Radome_Type_Changed
GOL2  94JAN17  2   363.078   201.965  6.7 ci3144585
"""

TENV3 = """\
site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)
ANKR 95JUN25 1995.4798 49893  807 0   32.8  -3552 -0.235481   4417023  0.323651   976  0.045141  0.0600 0.000771 0.001002 0.003364 -0.017130  0.014898 -0.175585  39.8873700200 -327.2415301033   976.04514
ANKR 95JUN26 1995.4825 49894  807 1   32.8  -3552 -0.237141   4417023  0.327936   976  0.040824  0.0600 0.000768 0.001013 0.003416 -0.005760  0.006089 -0.134956  39.8873700586 -327.2415301227   976.04082
"""


@pytest.fixture
def files(tmp_path: Path):
    (tmp_path / "h.txt").write_text(HOLDINGS)
    (tmp_path / "s.txt").write_text(STEPS)
    (tmp_path / "ANKR.tenv3").write_text(TENV3)
    return tmp_path


def test_holdings_wraps_longitude(files):
    h = ngl.read_holdings(files / "h.txt")
    assert list(h.sta) == ["ANKR", "FAR1"]
    assert h.loc[1, "lon"] == pytest.approx(-143.0)
    assert h.loc[0, "start"] == pd.Timestamp("1995-06-25")


def test_steps_kinds(files):
    s = ngl.read_steps(files / "s.txt")
    assert list(s.kind) == ["equipment", "quake"]
    q = s.iloc[1]
    assert q.date == pd.Timestamp("1994-01-17") and q.mag == 6.7 and q["info"] == "ci3144585"


def test_tenv3_dates_and_components(files):
    d = ngl.read_tenv3(files / "ANKR.tenv3")
    assert d.index[0] == pd.Timestamp("1995-06-25")  # MJD 49893
    assert d["e"].iloc[0] == pytest.approx(-3552.235481)
    assert d["n"].iloc[1] == pytest.approx(4417023.327936)
    assert d["u"].iloc[0] == pytest.approx(976.045141)


def test_stations_near_filters_distance_and_window(files):
    h = ngl.read_holdings(files / "h.txt")
    near = ngl.stations_near(h, 39.9, 32.8, 50)
    assert list(near.sta) == ["ANKR"]
    # ANKR's data ends 2022, so it cannot cover a window around Feb 2023
    assert ngl.stations_near(
        h, 39.9, 32.8, 50, start=pd.Timestamp("2022-02-06"), end=pd.Timestamp("2023-05-06")
    ).empty


def test_station_id_validation():
    assert ngl.is_station_id("P595") and ngl.is_station_id("ANKR")
    assert not ngl.is_station_id("P595 P594")
    assert not ngl.is_station_id("../x")
    assert not ngl.is_station_id("ankr")


def test_stations_in_bbox(files):
    h = ngl.read_holdings(files / "h.txt")
    assert list(ngl.stations_in_bbox(h, 30, 45, 25, 40).sta) == ["ANKR"]
    assert list(ngl.stations_in_bbox(h, 30, 45, -150, -140).sta) == ["FAR1"]
