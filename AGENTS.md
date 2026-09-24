# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 workspace for a linorobot2 platform and social navigation.
Core robot packages live in `linorobot2_*`: `*_bringup` contains launch and
sensor configuration, `*_description` contains URDF/Xacro and meshes,
`*_gazebo` contains worlds/models, and `*_navigation` contains Nav2 maps,
configuration, and behavior trees. `social_perception/` owns RGB-D, YOLO pose,
tracking, and social-region grounding; `social_navigation/` provides the Nav2
costmap plugin, velocity filter, scenarios, and models; `social_rl/` contains
the PPO training and runtime agent. Keep package-local code in `src/`,
`scripts/`, or the Python package directory, and installable resources in
`launch/`, `config/`, `rviz/`, `maps/`, or `models/` as appropriate.

## Build, Test, and Development Commands

Source ROS before working, then build only the affected dependency chain:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_navigation
source install/setup.bash
python3 -m unittest discover -s social_perception/test
```

Use `--packages-select <package>` for an isolated C++ rebuild. Launch the
social simulation in the documented order in `social_navigation/README.md`;
for example, `ros2 launch social_navigation social_bringup.launch.py rviz:=true`.
Do not hard-code workstation paths or introduce runtime downloads.

## Coding Style & Naming Conventions

Use four spaces for Python and conventional PEP 8 naming: `snake_case` for
functions, files, and ROS parameters; `PascalCase` for classes. Keep launch
files named `<feature>.launch.py` and YAML configuration under the owning
package's `config/` directory. C++ targets use C++17 with `-Wall -Wextra
-Wpedantic`; follow the existing ROS/C++ style, use `snake_case` members and
methods, and keep public interfaces in `include/`. Extend existing YAML knobs
for tunables instead of embedding machine-specific values in code.

## Testing Guidelines

Add deterministic regression coverage with behavior changes. Python geometry
tests are `unittest` files named `test_*.py` in `social_perception/test/` and
should run without a live ROS graph where possible. Run the command above
after perception or grounding changes; build changed C++ packages before
testing. No repository-wide coverage target is configured, so state the tests
and simulation scenario exercised in the PR.

## Commit & Pull Request Guidelines

History uses short imperative English or Vietnamese summaries, often scoped
with a branch-style prefix (for example, `Integrate/rl block e`) and PR number.
Write a concise subject describing the subsystem and outcome. PRs should
explain the behavior change, list build/test commands, link the relevant issue,
and include RViz/Gazebo screenshots or logs for visible navigation changes.
Call out any sim-versus-real configuration difference explicitly.

## Change Control & Safety

- Default to analysis, investigation, and explanation only. Do not modify files unless the user explicitly asks to implement, edit, fix, or apply a change.
- Before editing, state the planned approach and the exact files expected to change.
- Make the smallest focused change needed for the requested task.
- Do not refactor, reformat, rename, move, delete, or regenerate unrelated files.
- Do not modify dependency versions, lockfiles, CI, Docker files, ROS environment setup, package manifests, or launch/config files unless explicitly requested.
- Do not alter URDF/Xacro, Nav2 parameters, costmap behavior, robot safety limits, or simulation worlds without explicit confirmation.
- Do not run destructive commands, including `rm -rf`, `git reset --hard`, `git clean -fd`, or force-push.
- Do not overwrite user changes or revert existing work.
- If the request is ambiguous, affects multiple packages, or could change robot behavior, ask for clarification before editing.
- After making changes, list modified files and the build/tests actually run. Do not claim tests passed if they were not run.

## Code Comments

- When adding non-trivial code, include concise comments explaining:
  - the purpose of the code;
  - why the chosen approach is necessary;
  - ROS topic/frame/unit assumptions, when applicable;
  - safety or behavioral consequences for navigation and robot motion.
- Comment non-obvious algorithms, coordinate transforms, social-distance calculations, costmap logic, and velocity constraints.
- Do not add comments that merely restate obvious code syntax.
- Keep comments accurate, concise, and in English to match the codebase.
- Update or remove outdated comments whenever the related behavior changes.
- For each new ROS parameter, document its unit, valid range, default behavior, and practical effect in the owning YAML/config file or nearby code comment.
