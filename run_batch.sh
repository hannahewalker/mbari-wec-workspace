#!/usr/bin/env bash
# Run an mbari_wec batch simulation from a batch folder.
#
# usage:
#   ./run_batch.sh FOLDER [YAML]
#
#   FOLDER  batch folder, relative to this workspace or absolute (e.g. foofoo)
#   YAML    yaml inside FOLDER; optional if FOLDER holds exactly one .yaml
#
# Results (batch_results_<timestamp>/) are written inside FOLDER.
set -eo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

if [[ $# -lt 1 || $# -gt 2 || $1 == -h || $1 == --help ]]; then
    sed -n '2,10s/^# \{0,1\}//p' "$0"
    exit 1
fi

folder="$1"
[[ $folder = /* ]] || folder="$WS_DIR/$folder"
if [[ ! -d $folder ]]; then
    echo "error: no folder $folder" >&2
    exit 1
fi

if [[ $# -eq 2 ]]; then
    yaml="$2"
else
    shopt -s nullglob
    yamls=("$folder"/*.yaml "$folder"/*.yml)
    if [[ ${#yamls[@]} -ne 1 ]]; then
        echo "error: expected one .yaml in $folder, found ${#yamls[@]}; name it:" >&2
        echo "  $0 $1 <file.yaml>" >&2
        exit 1
    fi
    yaml="$(basename "${yamls[0]}")"
fi
if [[ ! -f $folder/$yaml ]]; then
    echo "error: no file $folder/$yaml" >&2
    exit 1
fi

# ROS setup scripts reference unset variables, so source them before set -u
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$WS_DIR/install/setup.bash"
set -u

cd "$folder"
echo "running $folder/$yaml"
ros2 launch buoy_gazebo mbari_wec_batch.launch.py sim_params_yaml:="$yaml"
