#!/usr/bin/env python3
"""Test replay after matching the complete initial left-right base vector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_TRAJECTORY_DIR = Path(
    "/home/kernel/Desktop/data/unified_episode/episode_20260818_0009"
)
R_GRID_TO_ROBOT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
)
R_SLAM_WORLD_TO_ROBOT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_dir", nargs="?", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def source_initial_base_vector(metadata_path: Path) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    direct_vector = metadata.get("initial_control_point_vector_robot_m")
    if direct_vector is not None:
        vector = np.asarray(direct_vector, dtype=float)
        if vector.shape == (3,) and np.all(np.isfinite(vector)):
            return vector

    # Compatibility fallback for metadata written by older converters. This
    # uses nearest source samples and is less precise than the synchronized
    # vector written by convert_episode_to_robot.py.
    source = Path(metadata["source"])
    stats = metadata["stats"]
    positions = {}
    for side in ("left", "right"):
        samples = np.genfromtxt(
            source / "pose_data" / f"pose_data_{side}.csv",
            delimiter=",",
            names=True,
            dtype=float,
        )
        index = int(stats[side]["source_initial_index"])
        position = np.array([samples["X"][index], samples["Y"][index], samples["Z"][index]])
        positions[side] = position
    source_frame = metadata.get("source_frame", "aprilgrid")
    source_to_robot = R_SLAM_WORLD_TO_ROBOT if source_frame == "slam_world" else R_GRID_TO_ROBOT
    return source_to_robot @ (positions["left"] - positions["right"])


def main() -> int:
    args = parse_args()
    from convert_episode_to_robot import resolve_trajectory_input

    input_path = args.trajectory_dir.expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    trajectory_dir = resolve_trajectory_input(input_path)
    print(f"[TEST] replay trajectory directory: {trajectory_dir}")
    os.environ["SCORE_FOLDER_PATH"] = str(trajectory_dir)

    import replay

    replay.FILE_SEARCH_LEVEL = 2
    replay.START_VIEWER = int(args.viewer)
    replay.LOG_FLAG = False

    metadata_path = trajectory_dir / "conversion_metadata.json"
    target_vector = source_initial_base_vector(metadata_path)
    robot = replay.RbtKin()
    home = robot.solve_fk()
    home_vector = np.asarray(home[0][:3]) - np.asarray(home[1][:3])
    # replay() restores the complete left-right vector automatically.
    adjustment = np.zeros(6, dtype=float)

    print(f"[TEST] target base vector (left-right) = {target_vector.tolist()}")
    print(f"[TEST] home base vector (left-right)   = {home_vector.tolist()}")
    print(f"[TEST] automatic vector alignment; manual correction = {adjustment.tolist()}")
    result = replay.replay(trajectory_index=args.index, adjust_ee_dist=adjustment)
    collision = result["detail"].get("collision") or {"colliding": False, "score": 100.0}
    print(f"[TEST] collision_min_score = {result['collision_min_score']}")
    print(f"[TEST] collision_min_index = {result['collision_min_index']}")
    print(f"[TEST] collision_min_time  = {result['collision_min_timestamp']}")
    print(f"[TEST] collision_first_time = {result.get('collision_first_timestamp')}")
    print(f"[TEST] collision_last_time  = {result.get('collision_last_timestamp')}")
    print(f"[TEST] collision_duration_sec = {result.get('collision_duration_sec', 0.0):.6f}")
    for interval_index, interval in enumerate(result.get("collision_intervals", []), start=1):
        print(
            f"[TEST] collision_interval_{interval_index}: "
            f"{interval['start_timestamp']:.6f}s -> "
            f"{interval['end_timestamp']:.6f}s "
            f"(duration={interval['duration_sec']:.6f}s, "
            f"samples={interval['sample_count']})"
        )
    print(f"[TEST] collision_detail    = {collision}")
    print(f"[TEST] continuous_min_score = {result['continuous_min_score']}")
    print(f"[TEST] out_ws_min_score     = {result['out_ws_min_score']}")
    return 0 if not collision["colliding"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
