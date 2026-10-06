import numpy as np
import pandas as pd
import pytest

from gnssdl import model


def _synthetic():
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    t = dates.year + (dates.dayofyear - 0.5) / np.where(dates.is_leap_year, 366, 365)
    x = 0.012 * (t - 2010) + 0.002 * np.sin(2 * np.pi * t)   # 12 mm/yr + 2 mm annual
    x = x + 0.005 * (dates >= "2012-06-01")                   # in-window step: 5 mm
    x = x + 0.030 * (dates >= "2018-03-01")                   # post-window step: 30 mm
    df = pd.DataFrame({"decyear": t, "e": x, "n": -x, "u": 0 * x}, index=dates)
    return df, [pd.Timestamp("2012-06-01"), pd.Timestamp("2018-03-01")]


def test_departure_recovers_velocity_and_exposes_later_offset():
    df, steps = _synthetic()
    res, vel = model.departure(df, "2010-01-01", "2015-12-31", steps)
    assert vel["e"] == pytest.approx(12.0, abs=1e-6)
    assert vel["n"] == pytest.approx(-12.0, abs=1e-6)
    assert res.loc[:"2015-12-31", "e"].abs().max() < 1e-6
    # The 2018 step was outside the fit window, so it must survive in the residual.
    assert res.loc["2019", "e"].mean() == pytest.approx(30.0, abs=1e-6)
    assert res.loc["2017", "e"].abs().max() < 1e-6


def test_departure_refuses_short_window():
    df, _ = _synthetic()
    with pytest.raises(ValueError):
        model.departure(df, "2010-01-01", "2010-06-01")


def test_equipment_jump_after_window_is_removed_but_quake_jump_is_not():
    df, steps = _synthetic()
    antenna = pd.Timestamp("2017-05-01")
    df.loc[antenna:, "e"] += 0.008
    res, _ = model.departure(df, "2010-01-01", "2015-12-31", steps + [antenna], [antenna])
    assert res.loc["2017-06":"2017-12", "e"].abs().max() < 1e-6
    assert res.loc["2019", "e"].mean() == pytest.approx(30.0, abs=1e-6)
