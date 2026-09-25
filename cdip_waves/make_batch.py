"""Write an mbari_wec batch folder whose sim_params yaml uses CDIP 156 spectra.

Each sea state from cdip_bands becomes one 'Custom' IncidentWaveSpectrumType
(f / Szz), and wave_dir is set from the CDIP peak-band direction. See
https://osrf.github.io/mbari_wec/main/Tutorials/Simulation/SimulatorParameters/

examples:
    make_batch.py 2016-03-08T01:00 2016-03-08T03:00       mean over a range
    make_batch.py 2016-03-08T01:23                        single record
    make_batch.py 2016 3 8                                whole-day mean
    make_batch.py 2016-03-08 2016-03-09 --window 6h       one spectrum per 6 h

Then run the batch from inside the folder:
    cd foofoo && ros2 launch buoy_gazebo mbari_wec_batch.launch.py \\
        sim_params_yaml:=foofoo.yaml
"""
import argparse
import os

import pandas as pd

import cdip_bands as cb

WS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _list(values):
    return "[" + ", ".join(repr(v) for v in values) + "]"


def batch_yaml(states, duration=300, seed=42, physics_rtf=11, physics_step=0.01,
               door_state="closed", scale_factor=1.0, battery_soc=0.5,
               enable_gui=False):
    """Batch yaml text for a list of cb.spectrum() / cb.sea_states() dicts."""
    lines = [
        "#",
        "# Batch-Specific Scalar Parameters",
        "#",
        f"duration: {duration}",
        f"seed: {seed}",
        f"physics_rtf: {physics_rtf}",
        f"enable_gui: {enable_gui}",
        "#",
        "# Run-Specific Parameters (Test Matrix)",
        "#",
        f"physics_step: {physics_step}",
        f"door_state: ['{door_state}']",
        f"scale_factor: [{scale_factor}]",
        f"battery_soc: {battery_soc}",
        "# Incident wave direction: compass degrees True, waves coming FROM",
        "# (direction of the CDIP band with the highest Szz)",
    ]
    dirs = [s["wave_dir"] for s in states]
    if len(set(dirs)) > 1:
        # wave_dir is crossed with every spectrum, not paired with it
        lines.append("# NOTE: every wave_dir runs with every spectrum below "
                     f"({len(dirs)} x {len(states)} runs)")
    lines.append(f"wave_dir: {_list(sorted(set(dirs), key=dirs.index))}")
    lines.append("IncidentWaveSpectrumType:")
    for s in states:
        spec = cb.average_window(s["start"], s["end"] + pd.Timedelta("1s"))
        custom = cb.to_custom_spectrum(spec)
        lines += [
            f"  # CDIP 156 {cb.label(s)}: n = {s['n_records']}, "
            f"Hs = {s['Hs_m']} m, wave_dir = {s['wave_dir']}",
            "  - Custom:",
            f"      f: {_list(custom['f'])}",
            f"      Szz: {_list(custom['Szz'])}",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Make an mbari_wec batch folder + yaml from CDIP 156 spectra.",
        epilog=__doc__.split("examples:")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("when", nargs="+", help="YEAR MONTH DAY, or START [END] (UTC)")
    ap.add_argument("--name", default="foofoo",
                    help="batch folder (and yaml) name, created in the workspace")
    ap.add_argument("--window", help="split START..END into sea states of this length")
    ap.add_argument("--n", type=int, help="with --window: pick N windows at random")
    ap.add_argument("--seed", type=int, default=42,
                    help="sim seed, also used for the --n draw")
    ap.add_argument("--duration", type=float, default=300, help="sim duration (s)")
    ap.add_argument("--physics-rtf", type=float, default=11)
    ap.add_argument("--physics-step", type=float, default=0.01)
    ap.add_argument("--door-state", default="closed", choices=["closed", "open"])
    ap.add_argument("--scale-factor", type=float, default=1.0)
    ap.add_argument("--battery-soc", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="overwrite an existing yaml")
    args = ap.parse_args()

    when = args.when
    if len(when) == 3 and all(w.isdigit() for w in when):
        t0 = pd.Timestamp(year=int(when[0]), month=int(when[1]), day=int(when[2]))
        when = [str(t0), str(t0 + pd.Timedelta(days=1))]
    if len(when) > 2:
        ap.error("give YEAR MONTH DAY, or START [END]")

    if args.window:
        if len(when) != 2:
            ap.error("--window needs START and END")
        states = cb.sea_states(*when, window=args.window, n=args.n, seed=args.seed)
    else:
        states = [cb.spectrum(*when)]

    folder = os.path.join(WS_DIR, args.name)
    path = os.path.join(folder, args.name + ".yaml")
    if os.path.exists(path) and not args.force:
        ap.error(f"{path} exists; use --force to overwrite")
    os.makedirs(folder, exist_ok=True)
    text = batch_yaml(states, duration=args.duration, seed=args.seed,
                      physics_rtf=args.physics_rtf, physics_step=args.physics_step,
                      door_state=args.door_state, scale_factor=args.scale_factor,
                      battery_soc=args.battery_soc)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
