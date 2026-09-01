#!/usr/bin/env bash
# Start the Gazebo social-navigation demo in four dedicated terminals.
#
# Usage:
#   ./social_navigation/scripts/run_social_demo.sh /absolute/path/to/yolo-pose.pt [talking|gathering] [map.yaml]

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${script_dir}/../.." && pwd)"
pose_model_path="${1:-}"
scenario="${2:-talking}"
map_path="${3:-}"

usage() {
    echo "Usage: $0 /absolute/path/to/yolo-pose.pt [talking|gathering] [map.yaml]" >&2
}

if [[ -z "${pose_model_path}" ]]; then
    usage
    exit 2
fi
if [[ ! -f "${pose_model_path}" ]]; then
    echo "YOLO Pose checkpoint does not exist: ${pose_model_path}" >&2
    exit 2
fi
if [[ "${scenario}" != "talking" && "${scenario}" != "gathering" ]]; then
    echo "scenario must be talking or gathering, got: ${scenario}" >&2
    exit 2
fi
if [[ -n "${map_path}" && ! -f "${map_path}" ]]; then
    echo "Map file does not exist: ${map_path}" >&2
    exit 2
fi
if [[ ! -f "${workspace_dir}/install/setup.bash" ]]; then
    echo "Workspace has not been built: ${workspace_dir}/install/setup.bash is missing" >&2
    exit 2
fi
if ! command -v gnome-terminal >/dev/null; then
    echo "gnome-terminal is required to open the four terminals." >&2
    exit 2
fi

open_terminal() {
    local title="$1"
    shift
    gnome-terminal --title="${title}" -- bash -lc '
        source /opt/ros/humble/setup.bash
        source "$1"
        cd "$2"
        shift 2
        "$@"
        status=$?
        printf "\nProcess exited with status %s. Press Enter to close this terminal.\n" "$status"
        read -r
        exit "$status"
    ' bash "${workspace_dir}/install/setup.bash" "${workspace_dir}" "$@"
}

echo "[1/4] Starting Gazebo..."
open_terminal '1 - Gazebo' \
    ros2 launch linorobot2_gazebo gazebo.launch.py run_ekf:=false

# Gazebo needs a moment to create /clock, the camera and its TF tree.
sleep 5

echo "[2/4] Starting YOLO Pose and RViz..."
open_terminal '2 - YOLO Pose and RViz' \
    ros2 launch social_navigation social_bringup.launch.py \
    sim:=true rviz:=true "yolo_model_path:=${pose_model_path}"

sleep 2

echo "[3/4] Starting actors (${scenario}); they wait for perception readiness..."
open_terminal '3 - Animated people' \
    ros2 launch social_navigation social_sim.launch.py \
    "scenario:=${scenario}" wait_for_perception:=true ready_timeout:=600.0

sleep 1

nav_command=(ros2 launch linorobot2_navigation navigation.launch.py sim:=true)
if [[ -n "${map_path}" ]]; then
    nav_command+=("map:=${map_path}")
fi

echo "[4/4] Starting Nav2..."
open_terminal '4 - Nav2' "${nav_command[@]}"

echo "All terminals were opened. Stop each component with Ctrl+C in its own terminal."
