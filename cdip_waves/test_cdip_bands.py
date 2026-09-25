"""Tests for cdip_bands.py.

Run from anywhere:
    python3 -m pytest /home/hwalker/mbari_wec_ws/cdip_waves/test_cdip_bands.py -v

Most tests build small synthetic de/dd files with known values, so the
expected answers are exact. The tests marked "real data" read the CDIP 156
files in cdip_bands.DATA_DIR and are skipped if that folder is missing.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdip_bands as cb  # noqa: E402

N = len(cb.BANDS)


# ------------------------------------------------------------ synthetic files

def _write_year(folder, records):
    """records: list of (timestamp 'YYYYMMDDHHMM', Hs cm, Tp str, Dp,
    energies[9], directions[9]). Writes files in the CDIP fixed-width layout."""
    os.makedirs(folder, exist_ok=True)
    de = ["ENERGY (CM^2)", "UTC Hs Tp BAND PERIOD LIMITS (SECS)",
          "YYYYMMDDHHMM (CM) (SEC) 22+ 22-18 18-16 16-14 14-12 12-10 10-8 8-6 6-2"]
    dd = ["ANGULAR DISTRIBUTION", "UTC Dp BAND PERIOD LIMITS (SECS)",
          "YYYYMMDDHHMM (DEG) +22 22-18 18-16 16-14 14-12 12-10 10-8 8-6 6-2"]
    for t, hs, tp, dp, E, D in records:
        de.append(f"{t}{hs:5d}{tp:>3}" + "".join(f"{v:7d}" for v in E))
        dd.append(f"{t}{dp:5d}   " + "".join(f"{v:7d}" for v in D))
    year = os.path.basename(folder)
    with open(os.path.join(folder, f"de156p101_{year}01-{year}12"), "w") as f:
        f.write("\n".join(de) + "\n")
    with open(os.path.join(folder, f"dd156p101_{year}01-{year}12"), "w") as f:
        f.write("\n".join(dd) + "\n")


@pytest.fixture
def data_dir(tmp_path):
    """2016: eight half-hourly records on 2016-03-08 from 00:23 to 03:53.
       Band 0 ('22+') is zero energy in every record.
       Band 1 directions alternate 350/10 deg with equal energy.
       Band 2 alternates 0 deg (E=300) and 90 deg (E=100).
       Record 3 has the merged Hs/Tp field '34522+'.
    2017: two records on 2017-01-01."""
    recs = []
    for i in range(8):
        t = (pd.Timestamp("2016-03-08 00:23") + pd.Timedelta(minutes=30 * i))
        E = [0, 100, 300 if i % 2 == 0 else 100] + [10 * (i + 1)] * (N - 3)
        D = [270, 350 if i % 2 == 0 else 10, 0 if i % 2 == 0 else 90] + [280] * (N - 3)
        tp = "22+" if i == 3 else "12"
        recs.append((t.strftime("%Y%m%d%H%M"), 345 if i == 3 else 200, tp, 280, E, D))
    _write_year(tmp_path / "2016", recs)
    _write_year(tmp_path / "2017", [
        ("201701010023", 150, "10", 270, [1] * N, [270] * N),
        ("201701010053", 150, "10", 270, [3] * N, [270] * N),
    ])
    return str(tmp_path)


# ------------------------------------------------------------------ band table

def test_band_table_edges():
    bt = cb.band_table()
    assert list(bt.index) == cb.BANDS
    assert (bt["f_low"] < bt["f_high"]).all()
    # bands are contiguous: each band's upper frequency is the next one's lower
    np.testing.assert_allclose(bt["f_high"].values[:-1], bt["f_low"].values[1:])
    assert bt["T_max"].iloc[0] == cb.LONGEST_PERIOD
    assert bt["T_min"].iloc[-1] == 2
    np.testing.assert_allclose(bt["df"], bt["f_high"] - bt["f_low"])


def test_band_table_longest_period():
    assert cb.band_table(longest_period=40)["f_low"].iloc[0] == pytest.approx(1 / 40)


# ---------------------------------------------------------------- file reading

def test_load_year_parses_merged_hs_tp(data_dir):
    e, d = cb.load_year(2016, data_dir)
    assert len(e) == len(d) == 8
    row = e.loc["2016-03-08 01:53"]
    assert row["Hs"] == 345 and row["Tp"] == 22       # from '34522+'
    assert list(e.columns) == ["Hs", "Tp"] + cb.BANDS
    assert list(d.columns) == ["Dp"] + cb.BANDS
    assert e.index.is_monotonic_increasing


def test_available_years(data_dir):
    assert cb.available_years(data_dir) == ["2016", "2017"]


def test_missing_year_raises(data_dir):
    with pytest.raises(FileNotFoundError):
        cb.load_year(2015, data_dir)


# ------------------------------------------------------------- extract_records

def test_extract_records_half_open_window(data_dir):
    # 00:23 and 00:53 are inside; 01:23 is exactly `end`, so excluded
    e, d = cb.extract_records("2016-03-08 00:00", "2016-03-08 01:23", data_dir)
    assert list(e.index.strftime("%H:%M")) == ["00:23", "00:53"]
    assert e.index.equals(d.index)


def test_extract_records_across_years(data_dir):
    e, _ = cb.extract_records("2016-03-08 03:00", "2017-01-02", data_dir)
    assert len(e) == 2 + 2   # 03:23, 03:53 in 2016 plus two 2017 records


def test_extract_records_empty_raises(data_dir):
    with pytest.raises(ValueError, match="no records"):
        cb.extract_records("2016-07-01", "2016-07-02", data_dir)


# -------------------------------------------------------------- average_window

@pytest.fixture
def spec(data_dir):
    # four records: 00:23, 00:53, 01:23, 01:53
    return cb.average_window("2016-03-08 00:00", "2016-03-08 02:00", data_dir)


def test_average_window_attrs(spec):
    assert spec.attrs["n_records"] == 4
    assert spec.attrs["start"] == pd.Timestamp("2016-03-08 00:23")
    assert spec.attrs["end"] == pd.Timestamp("2016-03-08 01:53")
    assert spec.attrs["Hs_m"] == pytest.approx(4 * np.sqrt(spec["energy_m2"].sum()))


def test_average_window_energy(spec):
    # bands 3+ have energy 10, 20, 30, 40 in the four records -> mean 25
    assert spec["energy_cm2"].iloc[3:].tolist() == [25.0] * (N - 3)
    assert spec.loc["22-18", "energy_cm2"] == 100
    assert spec.loc["18-16", "energy_cm2"] == 200    # mean of 300, 100, 300, 100
    np.testing.assert_allclose(spec["energy_m2"], spec["energy_cm2"] / 1e4)
    np.testing.assert_allclose(spec["Szz"], spec["energy_m2"] / spec["df"])


def test_direction_wraps_through_north(spec):
    # 350 and 10 deg with equal energy average to 0 deg, not 180
    assert spec.loc["22-18", "direction_deg"] == pytest.approx(0, abs=1e-9)
    assert spec.loc["22-18", "spread_deg"] > 0


def test_direction_is_energy_weighted(spec):
    # 0 deg with E=300 and 90 deg with E=100 -> atan2(100, 300)
    assert spec.loc["18-16", "direction_deg"] == pytest.approx(
        np.degrees(np.arctan2(100, 300)))


def test_zero_energy_band_still_has_direction(spec):
    assert spec.loc["22+", "energy_cm2"] == 0
    assert spec.loc["22+", "direction_deg"] == pytest.approx(270)
    assert not spec.isna().any().any()


def test_constant_direction_has_zero_spread(spec):
    assert spec.loc["6-2", "direction_deg"] == pytest.approx(280)
    assert spec.loc["6-2", "spread_deg"] == pytest.approx(0, abs=1e-6)


def test_energy_std(spec):
    # bands 3+ have energy 10, 20, 30, 40 across the four records
    expected = np.std([10, 20, 30, 40], ddof=1)
    np.testing.assert_allclose(spec["energy_std_cm2"].iloc[3:], expected)
    assert spec.loc["22-18", "energy_std_cm2"] == 0            # constant 100
    assert spec.loc["18-16", "energy_std_cm2"] == pytest.approx(np.std([300, 100] * 2, ddof=1))
    np.testing.assert_allclose(spec["energy_std_m2"], spec["energy_std_cm2"] / 1e4)
    np.testing.assert_allclose(spec["Szz_std"], spec["energy_std_m2"] / spec["df"])


def test_szz_values_m2_per_hz(spec):
    # band 6-2: mean energy 25 cm^2 = 0.0025 m^2 over df = 1/2 - 1/6 Hz
    df = 1 / 2 - 1 / 6
    assert spec.loc["6-2", "Szz"] == pytest.approx(0.0025 / df)                  # m^2/Hz
    std_m2 = np.std([10, 20, 30, 40], ddof=1) / 1e4
    assert spec.loc["6-2", "Szz_std"] == pytest.approx(std_m2 / df)              # m^2/Hz


def test_direction_std_degrees(spec):
    # 22-18: 350 and 10 deg, equal energy -> R = cos(10 deg),
    # spread = sqrt(2(1 - R)) rad = 2 sin(5 deg) rad, about 9.99 deg
    assert spec.loc["22-18", "spread_deg"] == pytest.approx(
        np.degrees(2 * np.sin(np.radians(5))))
    # 18-16: two records at 0 deg (E=300) and two at 90 deg (E=100)
    R = np.hypot(600, 200) / 800
    assert spec.loc["18-16", "spread_deg"] == pytest.approx(np.degrees(np.sqrt(2 * (1 - R))))
    assert spec.loc["18-16", "spread_deg"] == pytest.approx(37.08, abs=0.01)     # deg


def test_single_record_std_is_nan(data_dir):
    spec = cb.average_window("2016-03-08 00:00", "2016-03-08 00:30", data_dir)
    assert spec.attrs["n_records"] == 1
    assert spec["energy_std_cm2"].isna().all()


def test_directions_in_range(spec):
    assert spec["direction_deg"].between(0, 360, inclusive="left").all()


# ------------------------------------------------------------- average_windows

def test_average_windows_splits_and_skips_empty(data_dir):
    many = cb.average_windows("2016-03-08 00:00", "2016-03-08 06:00", "1h", data_dir)
    # data only covers 00:23-03:53, so windows starting 04:00 and 05:00 are skipped
    starts = many["window_start"].unique()
    assert len(starts) == 4
    assert len(many) == 4 * N
    assert (many["n_records"] == 2).all()


def test_average_windows_matches_average_window(data_dir):
    many = cb.average_windows("2016-03-08 00:00", "2016-03-08 04:00", "2h", data_dir)
    first = many[many["window_start"] == pd.Timestamp("2016-03-08 00:00")]
    single = cb.average_window("2016-03-08 00:00", "2016-03-08 02:00", data_dir)
    np.testing.assert_allclose(first["energy_cm2"], single["energy_cm2"])
    np.testing.assert_allclose(first["direction_deg"], single["direction_deg"])


def test_average_windows_too_short_raises(data_dir):
    with pytest.raises(ValueError):
        cb.average_windows("2016-03-08 00:00", "2016-03-08 01:00", "2h", data_dir)


# ---------------------------------------------------------- to_custom_spectrum

def test_custom_spectrum_format(spec):
    c = cb.to_custom_spectrum(spec)
    assert len(c["f"]) == len(c["Szz"]) == N + 2
    assert c["Szz"][0] == 0 and c["Szz"][-1] == 0
    assert all(np.diff(c["f"]) > 0)                       # increasing frequency
    assert all(type(v) is float for v in c["f"] + c["Szz"])  # YAML-friendly


def test_custom_spectrum_no_pad(spec):
    c = cb.to_custom_spectrum(spec, pad=False)
    assert len(c["f"]) == N
    # BANDS run longest period first, i.e. already in increasing frequency
    np.testing.assert_allclose(c["Szz"], spec["Szz"].values, atol=1e-5)


# ------------------------------------------------------------------ real data

real = pytest.mark.skipif(not os.path.isdir(os.path.join(cb.DATA_DIR, "2016")),
                          reason="CDIP 2016 data folder not found")


@real
def test_real_2016_loads():
    e, d = cb.load_year(2016)
    assert len(e) == len(d) == 6943
    assert e.index[0] == pd.Timestamp("2016-01-01 00:23")
    assert not e.isna().any().any() and not d.isna().any().any()
    assert (e[cb.BANDS] >= 0).all().all()
    assert d[cb.BANDS].stack().between(0, 360).all()


@real
def test_real_band_energies_match_file_hs():
    # Hs in the file should equal 4*sqrt(total band energy) to rounding
    e, _ = cb.load_year(2016)
    hs_from_bands = 4 * np.sqrt(e[cb.BANDS].sum(axis=1))
    assert np.median(np.abs(hs_from_bands - e["Hs"])) < 2      # cm
    assert np.percentile(np.abs(hs_from_bands - e["Hs"]), 99) < 5


@real
def test_real_storm_window():
    spec = cb.average_window("2016-03-08 01:00", "2016-03-08 03:00")
    assert spec.attrs["n_records"] == 4
    assert spec.attrs["Hs_m"] == pytest.approx(5.30, abs=0.01)
    assert spec["energy_cm2"].idxmax() == "14-12"
    assert spec.loc["14-12", "direction_deg"] == pytest.approx(277, abs=1)


# -------------------------------------------------------------- daily_spectrum

def test_daily_spectrum_arrays_line_up(data_dir):
    d = cb.daily_spectrum(2016, 3, 8, data_dir)
    assert d["n_records"] == 8                               # whole day
    assert len(d["f"]) == len(d["Szz"]) == len(d["energy"]) == len(d["direction"]) == N
    assert all(np.diff(d["f"]) > 0)                          # increasing frequency
    spec = cb.average_window("2016-03-08", "2016-03-09", data_dir).sort_values("f_center")
    np.testing.assert_allclose(d["Szz"], spec["Szz"], atol=1e-5)
    np.testing.assert_allclose(d["energy"], spec["energy_m2"], atol=1e-5)
    np.testing.assert_allclose(d["direction"], spec["direction_deg"], atol=1e-5)
    assert all(type(v) is float for k in ("f", "Szz", "energy", "direction") for v in d[k])
    np.testing.assert_allclose(d["Szz_std"], spec["Szz_std"], atol=1e-5)          # m^2/Hz
    np.testing.assert_allclose(d["energy_std"], spec["energy_std_m2"], atol=1e-5)  # m^2
    np.testing.assert_allclose(d["direction_std"], spec["spread_deg"], atol=1e-5)  # deg


def test_daily_spectrum_accepts_strings(data_dir):
    assert cb.daily_spectrum("2017", "1", "1", data_dir)["n_records"] == 2


def test_daily_spectrum_no_data_raises(data_dir):
    with pytest.raises(ValueError, match="no records"):
        cb.daily_spectrum(2016, 7, 1, data_dir)


def test_print_and_plot_daily(data_dir, tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    d = cb.daily_spectrum(2016, 3, 8, data_dir)
    cb.print_daily(d, "2016-03-08 UTC")
    out = capsys.readouterr().out
    assert out.startswith("# 2016-03-08 UTC: n = 8, Hs = ")
    assert len([l for l in out.splitlines() if l[:2].strip().isdigit()]) == N  # table rows
    assert "Szz ± std (m2/Hz)" in out and "dir ± std (deg)" in out
    assert "Szz_std:" in out and "direction_std:" in out
    png = tmp_path / "day.png"
    fig = cb.plot_daily(d, "2016-03-08", save_path=png, show=False)
    assert png.stat().st_size > 0 and len(fig.axes) == 3   # Szz, energy, direction


@real
def test_real_daily_output():
    """Prints the 2016-03-08 daily arrays; see them with `pytest -s`."""
    d = cb.daily_spectrum(2016, 3, 8)
    print()
    cb.print_daily(d, "2016-03-08 UTC")
    assert d["n_records"] == 48
    assert all(v > 0 for v in d["Szz_std"])                     # m^2/Hz
    assert all(0 < v < 30 for v in d["direction_std"])          # deg


# ------------------------------------------------------- spectrum / sea_states

def test_spectrum_range_matches_average_window(data_dir):
    d = cb.spectrum("2016-03-08 00:00", "2016-03-08 02:00", data_dir)
    spec = cb.average_window("2016-03-08 00:00", "2016-03-08 02:00",
                             data_dir).sort_values("f_center")
    assert d["n_records"] == 4
    assert d["start"] == pd.Timestamp("2016-03-08 00:23")
    assert d["end"] == pd.Timestamp("2016-03-08 01:53")
    np.testing.assert_allclose(d["Szz"], spec["Szz"], atol=1e-5)
    np.testing.assert_allclose(d["direction_std"], spec["spread_deg"], atol=1e-5)
    # wave_dir is the direction of the band with the highest Szz
    assert d["wave_dir"] == pytest.approx(d["direction"][int(np.argmax(d["Szz"]))], abs=1e-3)


def test_spectrum_single_record(data_dir):
    d = cb.spectrum("2016-03-08 01:20", data_dir=data_dir)
    assert d["n_records"] == 1
    assert d["start"] == d["end"] == pd.Timestamp("2016-03-08 01:23")
    # record i=2: bands 3+ hold 30 cm^2 = 0.003 m^2 at 280 deg
    assert d["energy"][3:] == [0.003] * (N - 3)              # m^2
    assert d["direction"][3:] == [280.0] * (N - 3)
    for k in ("Szz_std", "energy_std", "direction_std"):
        assert np.isnan(d[k]).all()                           # no spread from one record


def test_spectrum_single_record_picks_nearest(data_dir):
    # 01:40 is 13 min from 01:53 and 17 min from 01:23
    assert cb.spectrum("2016-03-08 01:40", data_dir=data_dir)["start"] == \
        pd.Timestamp("2016-03-08 01:53")


def test_spectrum_single_record_outside_tolerance_raises(data_dir):
    with pytest.raises(ValueError, match="no record within"):
        cb.spectrum("2016-03-08 05:00", data_dir=data_dir)    # last record 03:53


def test_sea_states_one_per_window(data_dir):
    states = cb.sea_states("2016-03-08 00:00", "2016-03-08 06:00", "1h", data_dir=data_dir)
    # data covers 00:23-03:53: four 1 h windows with two records each
    assert [s["window_start"].hour for s in states] == [0, 1, 2, 3]
    assert all(s["n_records"] == 2 for s in states)
    assert all(s["f"] == states[0]["f"] for s in states)


def test_sea_states_random_pick_is_repeatable(data_dir):
    kw = dict(window="30min", n=3, seed=7, data_dir=data_dir)
    a = cb.sea_states("2016-03-08 00:00", "2016-03-08 04:00", **kw)
    b = cb.sea_states("2016-03-08 00:00", "2016-03-08 04:00", **kw)
    assert len(a) == 3 and all(s["n_records"] == 1 for s in a)
    assert [s["start"] for s in a] == [s["start"] for s in b]
    assert [s["start"] for s in a] == sorted(s["start"] for s in a)   # time order


def test_sea_states_n_too_large_raises(data_dir):
    with pytest.raises(ValueError, match="only 4 windows"):
        cb.sea_states("2016-03-08 00:00", "2016-03-08 06:00", "1h", n=5, data_dir=data_dir)


def test_print_sea_states(data_dir, capsys):
    states = cb.sea_states("2016-03-08 00:00", "2016-03-08 04:00", "2h", data_dir=data_dir)
    cb.print_sea_states(states)
    out = capsys.readouterr().out
    assert out.count("Szz: ") == 2
    assert f"wave_dir: {[s['wave_dir'] for s in states]}" in out
