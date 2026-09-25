#!/usr/bin/env bash
# Run an mbari_wec batch simulation with an external controller.
#
# usage:
#   ./run_batch_controller.sh FOLDER [YAML] [-p PACKAGE] [-l LAUNCH_FILE] [-w CONTROLLER_WS]
#
#   FOLDER  batch folder, relative to this workspace or absolute (e.g. foofoo)
#   YAML    yaml inside FOLDER; optional if FOLDER holds exactly one .yaml
#   -p      controller ROS 2 package      (default: mbari_wec_temp_9_15)
#   -l      launch file in that package   (default: controller.launch.py)
#   -w      workspace the controller is built in (default: ~/controller_ws)
#
# The batch yaml is run with a 'controller:' entry added, so the controller is
# started alongside every run in the test matrix. Your yaml is not modified.
# Results (batch_results_<timestamp>/) are written inside FOLDER, as with
# run_batch.sh. Rebuild the controller first after editing it:
#   cd ~/controller_ws && colcon build --packages-select mbari_wec_temp_9_15
set -eo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

usage() { sed -n '2,18s/^# \{0,1\}//p' "$0"; exit "${1:-1}"; }

[[ ${1:-} == -h || ${1:-} == --help ]] && usage 0
[[ $# -ge 1 && $1 != -* ]] || usage
folder="$1"; shift
yaml=""
if [[ $# -ge 1 && $1 != -* ]]; then
    yaml="$1"; shift
fi
ctl_pkg="mbari_wec_temp_9_15"
ctl_launch="controller.launch.py"
ctl_ws="$HOME/controller_ws"
while getopts ':p:l:w:h' opt; do
    case "$opt" in
        p) ctl_pkg="$OPTARG" ;;
        l) ctl_launch="$OPTARG" ;;
        w) ctl_ws="$OPTARG" ;;
        h) usage 0 ;;
        *) echo "error: unknown option -$OPTARG" >&2; usage ;;
    esac
done

[[ $folder = /* ]] || folder="$WS_DIR/$folder"
if [[ ! -d $folder ]]; then
    echo "error: no folder $folder" >&2
    exit 1
fi

if [[ -z $yaml ]]; then
    shopt -s nullglob
    yamls=("$folder"/*.yaml "$folder"/*.yml)
    if [[ ${#yamls[@]} -ne 1 ]]; then
        echo "error: expected one .yaml in $folder, found ${#yamls[@]}; name it:" >&2
        echo "  $0 $(basename "$folder") <file.yaml>" >&2
        exit 1
    fi
    yaml="$(basename "${yamls[0]}")"
fi
if [[ ! -f $folder/$yaml ]]; then
    echo "error: no file $folder/$yaml" >&2
    exit 1
fi
if [[ ! -f $ctl_ws/install/local_setup.bash ]]; then
    echo "error: $ctl_ws/install/local_setup.bash not found; build the controller workspace" >&2
    exit 1
fi

# ROS setup scripts reference unset variables, so source them before set -u.
# The controller workspace goes last so its packages overlay mbari_wec_ws.
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$WS_DIR/install/setup.bash"
source "$ctl_ws/install/local_setup.bash"
set -u

if ! ros2 pkg prefix "$ctl_pkg" >/dev/null 2>&1; then
    echo "error: package '$ctl_pkg' not found in $ctl_ws/install" >&2
    exit 1
fi
launch_path="$(ros2 pkg prefix "$ctl_pkg")/share/$ctl_pkg/launch/$ctl_launch"
if [[ ! -f $launch_path ]]; then
    echo "error: no launch file $launch_path" >&2
    exit 1
fi

# Batch yaml + controller entry, deleted when the run ends. The batch runner
# copies it into batch_results_*/ under this name, recording the controller used.
stem="${yaml%.*}"
ctl_yaml="${stem}_controller.yaml"
if [[ -e $folder/$ctl_yaml ]]; then
    echo "error: $folder/$ctl_yaml exists (left from another run?); remove it first" >&2
    exit 1
fi
python3 - "$folder/$yaml" "$folder/$ctl_yaml" "$ctl_pkg" "$ctl_launch" <<'EOF'
import sys
import yaml
src, dst, pkg, launch = sys.argv[1:]
with open(src) as f:
    params = yaml.safe_load(f)
if "controller" in params:
    sys.exit(f"error: {src} already has a 'controller:' entry; use run_batch.sh for it")
with open(src) as f:
    text = f.read().rstrip("\n")
with open(dst, "w") as f:
    f.write(text + "\n"
            f"# added by run_batch_controller.sh\n"
            f"controller:\n  - package: '{pkg}'\n    launch_file: '{launch}'\n")
EOF
trap 'rm -f "$folder/$ctl_yaml"' EXIT

cd "$folder"
echo "running $folder/$yaml"
echo "controller: $ctl_pkg/$ctl_launch ($ctl_ws)"
ros2 launch buoy_gazebo mbari_wec_batch.launch.py sim_params_yaml:="$ctl_yaml"
