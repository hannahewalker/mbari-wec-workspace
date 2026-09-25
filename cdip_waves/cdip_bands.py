"""Extract banded wave spectra from CDIP 156 historic data over time windows.

The data are half-hourly records in nine period bands:
    de156p101_*  energy per band (cm^2), plus Hs (cm) and Tp (s)
    dd156p101_*  mean direction per band (deg, direction waves come from), plus Dp (deg)
stored in one folder per year under DATA_DIR (e.g. DATA_DIR/2016/).

Usage from elsewhere in the workspace:

    import sys
    sys.path.insert(0, "/home/hwalker/mbari_wec_ws/cdip_waves")
    import cdip_bands as cb

    # every record in a window
    energy, direction = cb.extract_records("2016-03-08 00:00", "2016-03-08 06:00")

    # one averaged spectrum for a window
    spec = cb.average_window("2016-03-08 01:00", "2016-03-08 03:00")

    # consecutive 2 h averages across a longer span
    specs = cb.average_windows("2016-03-01", "2016-03-03", window="2h")

    # f / Szz lists for an mbari_wec 'Custom' IncidentWaveSpectrumType
    custom = cb.to_custom_spectrum(spec)

    # matching f / Szz / energy / direction (+ std dev) arrays and wave_dir:
    d = cb.spectrum("2016-03-08 01:00", "2016-03-08 03:00")   # mean over a range
    d = cb.spectrum("2016-03-08 01:23")                       # one record
    d = cb.daily_spectrum(2016, 3, 8)                         # mean over a UTC day

    # one real spectrum per 2 h window (optionally n of them at random)
    states = cb.sea_states("2016-03-08", "2016-03-09", window="2h", n=4, seed=1)

Run this file directly for a quick demo, or see `python3 cdip_bands.py -h`
to print and plot a day, a range, a single record or a set of sea states.
"""
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "/home/hwalker/CDIP_156_historic_data"

BANDS = ["22+", "22-18", "18-16", "16-14", "14-12", "12-10", "10-8", "8-6", "6-2"]
# Period limits (s) of each band. The file's "22+" band has no upper limit;
# LONGEST_PERIOD closes it so it has a finite frequency width.
LONGEST_PERIOD = 30.0
_PERIOD_EDGES = [LONGEST_PERIOD, 22, 18, 16, 14, 12, 10, 8, 6, 2]


def band_table(longest_period=LONGEST_PERIOD):
    """Period and frequency limits of the nine bands, longest period first.

    Columns: T_max, T_min (s); f_low, f_high, f_center, df (Hz).
    f_center is the midpoint in frequency."""
    edges = np.array([longest_period] + _PERIOD_EDGES[1:], dtype=float)
    T_max, T_min = edges[:-1], edges[1:]
    f_low, f_high = 1 / T_max, 1 / T_min
    return pd.DataFrame({"T_max": T_max, "T_min": T_min, "f_low": f_low,
                         "f_high": f_high, "f_center": (f_low + f_high) / 2,
                         "df": f_high - f_low}, index=pd.Index(BANDS, name="band"))


# ---------------------------------------------------------------- file reading

def _find_file(folder, prefix):
    matches = [m for m in sorted(glob.glob(os.path.join(folder, prefix + "156*")))
               if not m.endswith(".png")]
    if not matches:
        raise FileNotFoundError(f"no {prefix}156* file in {folder}")
    return matches


def _read_energy(path):
    # Fixed-width: Hs and Tp can run together ('34522+' = Hs 345, Tp 22+),
    # so slice columns instead of splitting on whitespace. Tp '22+' -> 22.
    rows = []
    with open(path) as f:
        for line in f.readlines()[3:]:
            if line.strip():
                rows.append([line[:12], float(line[12:17]),
                             float(line[17:20].strip().rstrip("+"))]
                            + [float(v) for v in line[20:].split()])
    df = pd.DataFrame(rows, columns=["time", "Hs", "Tp"] + BANDS)
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M")
    return df.set_index("time")


def _read_direction(path):
    df = pd.read_csv(path, sep=r"\s+", skiprows=3, header=None,
                     names=["time", "Dp"] + BANDS, dtype={"time": str})
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M")
    return df.set_index("time").astype(float)


_cache = {}


def load_year(year, data_dir=DATA_DIR):
    """Return (energy, direction) DataFrames for one year folder, indexed by
    UTC time. energy: Hs (cm), Tp (s), then band energies (cm^2).
    direction: Dp (deg), then band directions (deg). Results are cached."""
    key = (str(year), os.path.abspath(data_dir))
    if key not in _cache:
        folder = os.path.join(data_dir, str(year))
        e = pd.concat([_read_energy(p) for p in _find_file(folder, "de")])
        d = pd.concat([_read_direction(p) for p in _find_file(folder, "dd")])
        _cache[key] = (e.sort_index(), d.sort_index())
    return _cache[key]


def available_years(data_dir=DATA_DIR):
    return sorted(n for n in os.listdir(data_dir)
                  if n.isdigit() and os.path.isdir(os.path.join(data_dir, n)))


# -------------------------------------------------------------- extraction API

def extract_records(start, end, data_dir=DATA_DIR):
    """All half-hourly records with start <= time < end (UTC).

    Returns (energy, direction) DataFrames; the window may span years.
    Raises ValueError if there are no records in the window."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    years = [y for y in available_years(data_dir) if start.year <= int(y) <= end.year]
    parts = [load_year(y, data_dir) for y in years]
    if not parts:
        raise ValueError(f"no year folders in {data_dir} cover {start} to {end}")
    e = pd.concat([p[0] for p in parts])
    d = pd.concat([p[1] for p in parts])
    e = e[(e.index >= start) & (e.index < end)]
    d = d[(d.index >= start) & (d.index < end)]
    common = e.index.intersection(d.index)
    if common.empty:
        raise ValueError(f"no records between {start} and {end}")
    return e.loc[common], d.loc[common]


def average_window(start, end, data_dir=DATA_DIR, longest_period=LONGEST_PERIOD):
    """One averaged spectrum for start <= time < end (UTC).

    Returns a DataFrame indexed by band (longest period first) with the
    band_table() columns plus:
        energy_cm2     mean band energy (cm^2)
        energy_std_cm2 sample std dev of band energy across records (cm^2)
        energy_std_m2  the same in m^2
        Szz_std        energy_std_m2 / df (m^2/Hz)
        energy_m2      mean band energy (m^2)
        Szz            spectral density, energy_m2 / df (m^2/Hz)
        direction_deg  energy-weighted circular mean direction (deg, from)
        spread_deg     circular spread of the per-record directions (deg);
                       the direction's std dev (energy-weighted angular deviation)
    and attrs: start, end, n_records, Hs_m (4*sqrt(sum energy)).
    Circular averaging means 350 deg and 10 deg average to 0 deg, not 180."""
    e, d = extract_records(start, end, data_dir)
    E = e[BANDS].values
    D = np.deg2rad(d[BANDS].values)
    # weight directions by energy; fall back to equal weights if a band is all zero
    w = np.where(E.sum(axis=0) > 0, E, 1.0)
    C, S = (w * np.cos(D)).sum(0), (w * np.sin(D)).sum(0)
    R = np.hypot(C, S) / w.sum(0)  # mean resultant length, 0..1

    out = band_table(longest_period)
    out["energy_cm2"] = E.mean(axis=0)
    out["energy_m2"] = out["energy_cm2"] / 1e4
    out["Szz"] = out["energy_m2"] / out["df"]
    # sample std dev (ddof=1); NaN if the window holds a single record
    out["energy_std_cm2"] = E.std(axis=0, ddof=1) if len(E) > 1 else np.nan
    out["energy_std_m2"] = out["energy_std_cm2"] / 1e4
    out["Szz_std"] = out["energy_std_m2"] / out["df"]
    mean_dir = np.rad2deg(np.arctan2(S, C)) % 360
    # a tiny negative angle rounds to 360.0 after % 360; report it as 0
    out["direction_deg"] = np.where(mean_dir >= 360, 0.0, mean_dir)
    out["spread_deg"] = np.rad2deg(np.sqrt(np.clip(2 * (1 - R), 0, None)))
    out.attrs.update(start=e.index[0], end=e.index[-1], n_records=len(e),
                     Hs_m=4 * np.sqrt(out["energy_m2"].sum()))
    return out


def average_windows(start, end, window="2h", data_dir=DATA_DIR,
                    longest_period=LONGEST_PERIOD):
    """Consecutive averaged spectra of length `window` (any pandas offset,
    e.g. '2h', '30min', '1D') from start to end.

    Returns a long-format DataFrame with one row per (window_start, band)
    and the same columns as average_window(), plus n_records and Hs_m.
    Windows with no data are skipped."""
    edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=window)
    if len(edges) < 2:
        raise ValueError(f"window {window} does not fit between {start} and {end}")
    frames = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        try:
            spec = average_window(t0, t1, data_dir, longest_period)
        except ValueError:
            continue
        spec = spec.reset_index()
        spec.insert(0, "window_start", t0)
        spec["n_records"] = spec.attrs["n_records"]
        spec["Hs_m"] = spec.attrs["Hs_m"]
        frames.append(spec)
    if not frames:
        raise ValueError(f"no records between {start} and {end}")
    return pd.concat(frames, ignore_index=True)


def to_custom_spectrum(spec, pad=True):
    """Convert an average_window() result to the f / Szz lists used by the
    mbari_wec 'Custom' IncidentWaveSpectrumType (f in Hz, Szz in m^2/Hz),
    ordered by increasing frequency at band centres.

    With pad=True, zero-energy points are added just outside the lowest and
    highest band so the spectrum tapers to zero at the ends.
    Note: the band shape is coarse (nine points), so the simulated Hs will
    only approximately match spec.attrs['Hs_m']."""
    s = spec.sort_values("f_center")
    f = s["f_center"].tolist()
    Szz = s["Szz"].tolist()
    if pad:
        f = [s["f_low"].iloc[0]] + f + [s["f_high"].iloc[-1]]
        Szz = [0.0] + Szz + [0.0]
    return {"f": [round(float(v), 5) for v in f], "Szz": [round(float(v), 5) for v in Szz]}


def _arrays(spec):
    """average_window() result -> plain-float per-band arrays in increasing f."""
    s = spec.sort_values("f_center")

    def as_list(col):
        return [round(float(v), 5) for v in s[col]]

    return {"f": as_list("f_center"), "Szz": as_list("Szz"),
            "energy": as_list("energy_m2"), "direction": as_list("direction_deg"),
            "Szz_std": as_list("Szz_std"), "energy_std": as_list("energy_std_m2"),
            "direction_std": as_list("spread_deg"),
            # direction at the Szz peak: a single value for the sim's wave_dir
            "wave_dir": round(float(s.loc[s["Szz"].idxmax(), "direction_deg"]), 3),
            "start": spec.attrs["start"], "end": spec.attrs["end"],
            "n_records": spec.attrs["n_records"],
            "Hs_m": round(float(spec.attrs["Hs_m"]), 3)}


def spectrum(start, end=None, data_dir=DATA_DIR, longest_period=LONGEST_PERIOD,
             tolerance="15min"):
    """Per-band arrays for one sea state, ordered by increasing frequency
    (element i of each list is the same band):
        f          band centre frequency (Hz)
        Szz        spectral density (m^2/Hz)   -> Custom 'Szz'
        energy     band energy (m^2)
        direction  energy-weighted mean direction (deg, from, compass)
        Szz_std, energy_std, direction_std
                   std dev of each across the records (direction_std is the
                   circular spread); NaN for a single record
        wave_dir   direction of the band with the highest Szz (deg)   -> 'wave_dir'
    plus start / end (first and last record used), n_records and Hs_m.

    With end given: the average of all records with start <= time < end (UTC).
    With end=None: the single record nearest to `start`, which must be within
    `tolerance` (records are every 30 min, so 15 min always finds one unless
    the data has a gap)."""
    if end is not None:
        return _arrays(average_window(start, end, data_dir, longest_period))
    t, tol = pd.Timestamp(start), pd.Timedelta(tolerance)
    try:
        e, _ = extract_records(t - tol, t + tol + pd.Timedelta("1s"), data_dir)
    except ValueError:
        raise ValueError(f"no record within {tolerance} of {t}") from None
    t_rec = e.index[np.abs(e.index - t).argmin()]
    d = _arrays(average_window(t_rec, t_rec + pd.Timedelta("1s"), data_dir,
                               longest_period))
    d["direction_std"] = [float("nan")] * len(d["f"])  # no spread from one record
    return d


def daily_spectrum(year, month, day, data_dir=DATA_DIR,
                   longest_period=LONGEST_PERIOD):
    """spectrum() averaged over one whole UTC day."""
    t0 = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    return spectrum(t0, t0 + pd.Timedelta(days=1), data_dir, longest_period)


def sea_states(start, end, window="2h", n=None, seed=None, data_dir=DATA_DIR,
               longest_period=LONGEST_PERIOD):
    """Split start..end into consecutive windows (any pandas offset: '2h',
    '30min' for every individual record, '1D', ...) and return a list of
    spectrum() dicts, one per window with data, each with a 'window_start'.

    n: keep only n windows, drawn at random without replacement (seed makes
    the draw repeatable) and returned in time order. Each state is a real
    measured spectrum, so band-to-band structure is kept."""
    edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=window)
    if len(edges) < 2:
        raise ValueError(f"window {window} does not fit between {start} and {end}")
    states = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        try:
            d = spectrum(t0, t1, data_dir, longest_period)
        except ValueError:
            continue
        d["window_start"] = t0
        states.append(d)
    if not states:
        raise ValueError(f"no records between {start} and {end}")
    if n is not None:
        if n > len(states):
            raise ValueError(f"asked for {n} sea states but only {len(states)} "
                             f"windows have data")
        keep = np.random.default_rng(seed).choice(len(states), size=n, replace=False)
        states = [states[i] for i in sorted(keep)]
    return states


def print_sea_states(states):
    """Print sea_states() output: one Szz array per state (f is the same for
    all) and the matching wave_dir list, in the same order."""
    print(f"# {len(states)} sea states")
    print(f"f: {states[0]['f']}")
    for i, d in enumerate(states):
        print(f"\n# [{i}] {d['window_start']:%Y-%m-%d %H:%M} UTC, "
              f"{d['n_records']} records, Hs = {d['Hs_m']} m, wave_dir = {d['wave_dir']}")
        print(f"Szz: {d['Szz']}")
    print(f"\nwave_dir: {[d['wave_dir'] for d in states]}")
    print(f"Hs_m: {[d['Hs_m'] for d in states]}")


def label(d):
    """Short description of a spectrum() result for printing and plot titles."""
    if d["n_records"] == 1:
        return f"{d['start']:%Y-%m-%d %H:%M} UTC record"
    return f"{d['start']:%Y-%m-%d %H:%M} to {d['end']:%Y-%m-%d %H:%M} UTC mean"


def print_spectrum(d, label=""):
    """Print spectrum() output as the raw arrays and a per-band table."""
    print(f"# {label + ': ' if label else ''}n = {d['n_records']}, "
          f"Hs = {d['Hs_m']} m, wave_dir = {d['wave_dir']}")
    for k in ("f", "Szz", "Szz_std", "energy", "energy_std", "direction", "direction_std"):
        print(f"{k}: {d[k]}")
    print(f"\n{'i':>2} {'f (Hz)':>8} {'T (s)':>7} {'Szz ± std (m2/Hz)':>22} "
          f"{'energy ± std (m2)':>20} {'dir ± std (deg)':>16}")
    rows = zip(d["f"], d["Szz"], d["Szz_std"], d["energy"], d["energy_std"],
               d["direction"], d["direction_std"])
    for i, (f, S, Ss, E, Es, D, Ds) in enumerate(rows):
        print(f"{i:>2} {f:>8.5f} {1 / f:>7.2f} {S:>10.5f} ± {Ss:<9.5f} "
              f"{E:>8.5f} ± {Es:<9.5f} {D:>6.1f} ± {Ds:<6.1f}")


def plot_spectrum(d, label="", save_path=None, show=True):
    """Plot spectrum() output: Szz, energy and direction against f.
    Saves to save_path if given; shows the window if show is True."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    def band(ax, y, std, floor=None):
        # mean line with a shaded +-1 std dev band (clipped at `floor`)
        y, std = np.asarray(y), np.asarray(std)
        lo = y - std if floor is None else np.maximum(y - std, floor)
        line, = ax.plot(d["f"], y, "o-")
        ax.fill_between(d["f"], lo, y + std, color=line.get_color(), alpha=0.25,
                        linewidth=0, label="±1 std dev")

    band(axs[0], d["Szz"], d["Szz_std"], floor=0)
    axs[0].set_ylabel("Szz (m$^2$/Hz)")
    axs[0].legend(loc="upper right")
    band(axs[1], d["energy"], d["energy_std"], floor=0)
    axs[1].set_ylabel("energy (m$^2$)")
    band(axs[2], d["direction"], d["direction_std"])
    axs[2].set_ylabel("direction (deg, from)")
    # zoom to the data (+-1 std, 10 deg margin) so the band is visible
    std = np.nan_to_num(d["direction_std"])  # NaN for a single record
    lo = np.min(np.subtract(d["direction"], std))
    hi = np.max(np.add(d["direction"], std))
    axs[2].set_ylim(max(lo - 10, 0), min(hi + 10, 360))
    axs[2].set_xlabel("frequency (Hz)")
    for ax in axs:
        ax.grid(alpha=0.3)
    top = axs[0].secondary_xaxis("top", functions=(lambda f: 1 / np.maximum(f, 1e-6),
                                                   lambda T: 1 / np.maximum(T, 1e-6)))
    top.set_xticks([22, 16, 12, 10, 8, 6, 4, 3])
    top.set_xlabel("period (s)")
    fig.suptitle(f"CDIP 156 {label}  (n = {d['n_records']}, "
                 f"Hs = {d['Hs_m']} m; shading = ±1 std dev)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


# names used before spectrum() existed
print_daily, plot_daily = print_spectrum, plot_spectrum


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Print and plot CDIP 156 banded spectra.",
        epilog="examples:\n"
               "  cdip_bands.py 2016 3 8                                 whole-day mean\n"
               "  cdip_bands.py 2016-03-08T01:00 2016-03-08T03:00        mean over a range\n"
               "  cdip_bands.py 2016-03-08T01:23                         single record\n"
               "  cdip_bands.py 2016-03-08 2016-03-09 --window 2h        one per 2 h window\n"
               "  cdip_bands.py 2016-03-08 2016-03-09 --window 30min --n 5 --seed 1",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("when", nargs="*", help="YEAR MONTH DAY, or START [END] (UTC)")
    ap.add_argument("--window", help="split START..END into sea states of this length")
    ap.add_argument("--n", type=int, help="with --window: pick N windows at random")
    ap.add_argument("--seed", type=int, help="with --n: random seed")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if len(args.when) == 3 and all(w.isdigit() for w in args.when):
        t0 = pd.Timestamp(year=int(args.when[0]), month=int(args.when[1]),
                          day=int(args.when[2]))
        args.when = [str(t0), str(t0 + pd.Timedelta(days=1))]

    if args.window:
        if len(args.when) != 2:
            ap.error("--window needs START and END")
        print_sea_states(sea_states(*args.when, window=args.window, n=args.n,
                                    seed=args.seed))
    elif args.when:
        if len(args.when) > 2:
            ap.error("give YEAR MONTH DAY, or START [END]")
        d = spectrum(*args.when)
        print_spectrum(d, label(d))
        if not args.no_plot:
            stamp = f"{d['start']:%Y%m%d%H%M}"
            if d["n_records"] > 1:
                stamp += f"-{d['end']:%Y%m%d%H%M}"
            png = f"spectrum_{stamp}.png"
            plot_spectrum(d, label(d), save_path=png)
            print(f"\nplot saved to {os.path.abspath(png)}")
    else:
        pd.set_option("display.width", 140)
        spec = average_window("2016-03-08 01:00", "2016-03-08 03:00")
        print(f"{spec.attrs['start']} to {spec.attrs['end']}, "
              f"{spec.attrs['n_records']} records, Hs = {spec.attrs['Hs_m']:.2f} m")
        print(spec.round(3))
        print("\nCustom spectrum:", to_custom_spectrum(spec))
        many = average_windows("2016-03-08", "2016-03-09", window="6h")
        print(f"\naverage_windows: {many['window_start'].nunique()} windows, "
              f"{len(many)} rows")
        print(many.groupby("window_start")[["Hs_m", "n_records"]].first())
