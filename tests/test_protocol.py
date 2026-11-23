"""Nominal laser settings per frame (celldform.acquisition.protocol)."""
import numpy as np
import pandas as pd
import pytest

from celldform import analysis_config as ac
from celldform.acquisition.protocol import LASER_COLUMNS, LaserProtocol, add_laser_columns, laser_settings


def test_levels_and_currents_follow_the_protocol():
    p = LaserProtocol()
    # 22 fps x 30 s = 660 frames per level
    s = laser_settings([0, 659, 660, 1319, 1320, 9239, 9240 - 1], p)
    assert s.power_level.tolist() == [0, 0, 1, 1, 2, 13, 13]
    assert s.current_mA.tolist() == [500, 500, 550, 550, 600, 1150, 1150]
    assert p.currents_mA[0] == 500 and p.currents_mA[-1] == 1200 and len(p.currents_mA) == 15


def test_last_level_and_frames_beyond_the_ramp():
    s = laser_settings([9240, 9899, 9900, 10357])          # level 14 = 9240..9899; the ramp ends at 450 s
    assert s.power_level.tolist() == [14, 14, 14, 14]
    assert s.current_mA.tolist() == [1200, 1200, 1200, 1200]
    assert s.beyond_ramp.tolist() == [False, False, True, True]


def test_power_matches_the_calibration_points():
    p = LaserProtocol()
    assert p.power_mW(500) == pytest.approx(270)
    assert p.power_mW(1200) == pytest.approx(685)
    assert p.power_mW(850) == pytest.approx(477.5)                 # midway
    s = laser_settings(np.arange(0, 9900, 660), p)
    assert np.all(np.diff(s.power_mW) > 0)                        # rises with every level


def test_calibration_table_is_interpolated():
    p = LaserProtocol(calibration_mA=(500, 800, 1200), calibration_mW=(270, 400, 685))
    assert p.power_mW(650) == pytest.approx(335)
    assert p.power_mW(1000) == pytest.approx(542.5)
    with pytest.raises(ValueError):
        LaserProtocol(calibration_mA=(800, 500), calibration_mW=(1, 2)).power_mW(600)


def test_add_laser_columns_places_and_replaces():
    f = pd.DataFrame({"stem": ["a", "b"], "frame_index": [20, 700], "x": [1.0, 2.0]})
    out = add_laser_columns(f)
    assert list(out.columns) == ["stem", "frame_index", *LASER_COLUMNS, "x"]
    again = add_laser_columns(out)                                # idempotent: columns replaced, not doubled
    assert list(again.columns) == list(out.columns)
    assert out.time_s.tolist() == pytest.approx([20 / 22, 700 / 22])
    assert f.columns.tolist() == ["stem", "frame_index", "x"]     # input untouched


def test_negative_index_is_an_error():
    with pytest.raises(ValueError):
        laser_settings([-1])


def test_config_defaults_match_the_module():
    lp = ac.load("features").laser_protocol
    p = LaserProtocol()
    assert (lp.fps, lp.hold_s, lp.start_mA, lp.step_mA, lp.n_levels) == \
           (p.fps, p.hold_s, p.start_mA, p.step_mA, p.n_levels)
    assert tuple(lp.calibration_mA) == p.calibration_mA and tuple(lp.calibration_mW) == p.calibration_mW
